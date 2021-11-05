"""
SQN network, reproduced based on the SQN paper, check https://arxiv.org/abs/2104.04891
Author: Chao YIN
Email: cyinac@connect.ust.hk

history: 
- Oct. 15, 2021, create the file

difference from the codebase (RandLANet.py of Official RandLA-Net) 
- add weak_labels relevant attributes
- delete the decoder part in its inference() and add query network
- add three_nearest_interpolation() based on tf_ops from official PointNet2 for the query network
- adjust the losses, modify the training() and evaluate() function correspondingly
"""

import sys
import time
import os, json
from os import makedirs
from os.path import exists, join
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from helper_tool import DataProcessing as DP, log_out
import helper_tf_util

# custom tf ops based on PointNet++ (https://github.com/charlesq34/pointnet2)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/3d_interpolation'))
from tf_interpolate import three_nn, three_interpolate


class SqnNet:
    """SQNetwork class based RandLA-Net's encoder and its query network
    - __init__(): set the config, flat_inputs(all those batched data), inputs, logits, loss, optimizer, results and log using TF summarywriter.
    - inference(): implement the network logic with encoder-decoder structure, need the follow function as core components:
        - dilated_res_block(): the dilated residual block
        - building_block(): build 1 simple block 
        - relative_pos_encoding(): relative position encoding for the LocSE
        - random_sample(): RS
        - nearest_interpolation(): nearest interpolation with inverse weighted distance
        - gather neighbour(): gather nearest neighbours
        - att_pooling(): attentive pooling
        - three_nearest_interpolation(): three nearest interpolation for each weak point in the query network
    - train(): training with an optimizer by running session for ops (following tensorflow 1.x pattern)
    - evaluate(): evaluate on the val/test set.
    """
    def __init__(self, dataset, config):

        # obtain the dataset iterator's next element under the hood
        flat_inputs = dataset.flat_inputs
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                # self.saving_path = time.strftime('{}/Log_%Y-%m-%d_%H-%M-%S'.format(config.results_dir), time.gmtime())
                self.saving_path = time.strftime('{}/Log_weak_{}_%Y-%m-%d_%H-%M-%S'.format(self.config.results_dir,dataset.weak_label_ratio), time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None
            
        # use inputs(a dict) variable to map the flat_inputs
        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers

            # correspond to the flat_inputs defined in get_tf_mapping2() in main_S3DIS_SQN.py
            # HACK: for encoder, it needs the original points, so add it to the first element of this array.
            self.inputs['original_xyz'] = flat_inputs[4 * num_layers] # features containing xyz and feature, (B,N,3+C)
            self.inputs['xyz'] = (self.inputs['original_xyz'],) + flat_inputs[:num_layers] # xyz_original plus xyz(points) of sub_pc at all the sub_sampling stages, containing num_layers items
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers] # neighbour id, containing num_layers items
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers] # sub_sampled idx, containing num_layers items
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers] # interpolation idx (nearest idx in the sub_pc for all raw pts), containing num_layers items
            self.inputs['features'] = flat_inputs[4 * num_layers + 1] # features containing xyz and feature, (B,N,3+C)
            self.inputs['labels'] = flat_inputs[4 * num_layers + 2]
            self.inputs['weak_label_masks'] = flat_inputs[4 * num_layers + 3]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 4] # input_inds for each batch 's point in the sub_pc
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 5] # cloud_inds for each batch

            self.points = self.inputs['original_xyz'] # (B,N,3)
            self.labels = self.inputs['labels'] # (B,N)
            self.weak_label_masks = self.inputs['weak_label_masks'] # weak label mask for weakly semseg, (B,N)
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            if self.config.saving:
                # put the training log to the resutls dir
                self.Log_file = open(os.path.join(self.saving_path,'log_train_' + dataset.name + str(dataset.val_split) + '.txt'), 'a')
            else:
                self.Log_file = open('log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits, self.weak_labels = self.inference(self.inputs, self.is_training) # (n, num_classes), (n,)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes]) # (n, num_classes)
            self.weak_labels = tf.reshape(self.weak_labels, [-1]) # (n,)
            # TODO: which to use, WCE, CE or smooth label
            # self.loss = self.get_loss_Sqn(self.logits, self.weak_labels)
            self.loss = self.get_loss(self.logits, self.weak_labels, self.class_weights)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            # self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.correct_prediction = tf.nn.in_top_k(self.logits, self.weak_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits) # (n,C)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        # self.train_writer = tf.summary.FileWriter(config.train_sum_dir, self.sess.graph)
        if hasattr(self, 'saving_path'):
            self.train_writer = tf.summary.FileWriter(self.saving_path, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):
        """similar to pytorch's forward() function where the SQN model architecture is implemented by an encoder-query structure
        Args:
            inputs ([type]): a dict containing all kinds of required inputs
            is_training (bool): training or not
        Returns:
            tensor: logits for segmentation scores
        """

        d_out = self.config.d_out # [16, 64, 128, 256], note the channels of LFA will be doubled.
        feature = inputs['features'] # (B,N,6)
        # feature = tf.layers.dense(feature, 8, activation=None, name='fc0') # (B,N,8)
        # feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2) # expand 1 more dim to use Conv2D ops, (B,N,1,8)

        # ###########################Encoder############################
        f_encoder_list = [] # in the end, collect num_layers + 1 items for a group of hierarchical point feature embeddings
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i],
                                                 'Encoder_layer_' + str(i), is_training) # similar to LAO for local feature learning
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i]) # down-sampled the input using the idx
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i) # (B,N,1,32), (B,N/4,1,32), (B,N/16,1,128), (B,N/64,1,256), (B,N/256,1,512)
        # ###########################Encoder############################


        # ###########################Query Network############################
        # obtain weakly points and labels for a batch using weak_label_masks
        # method2 using the gather_nd
        selected_idx = tf.where(tf.equal(self.weak_label_masks,1)) # (n,2)
        weak_points = tf.gather_nd(self.points, selected_idx)
        weak_points_labels=tf.gather_nd(self.labels, selected_idx)# (n,)
        # or use method1 using boolean_mask
        # weak_points = tf.boolean_mask(self.points,tf.cast(self.weak_label_masks,tf.bool)) # (n,3), e.g., one batch has 26 weak pts
        # weakly_points_labels = tf.boolean_mask(self.labels,tf.cast(self.weak_label_masks,tf.bool)) # (n,)

        # obtain batch indices to denote which batch is for every weakly point
        batch_inds = selected_idx[:,0]

        # query features for weak points
        f_query_feature_list = []
        for i in range(self.config.num_layers):
            xyz_current = inputs['xyz'][i+1] # (B,N/4,3), index i plus 1 because the first element is the point_original
            features_current = f_encoder_list[i+1] # (B,N/4,1,32), index plus 1 because the first one is the input of encoder

            # if training, shape (n,1,3), otherwise (B,N,3) (main reason here is to avoid GPU OOM issue)
            xyz_query = tf.cond(is_training,
                        lambda: tf.reshape(weak_points, (tf.shape(weak_points)[0],1,3)), # (n,1,3)
                        lambda: self.points)
            xyz_support = tf.cond(is_training,
                        lambda: tf.gather(xyz_current, batch_inds, axis=0), # (B,m,3)->(n,m,3) as each weak pt might be from diff. batch
                        lambda: xyz_current)
            features_support = tf.cond(is_training,
                        lambda: tf.gather(tf.squeeze(features_current,axis=2), batch_inds, axis=0), # (B,m,C)->(n,m,C)
                        lambda: tf.squeeze(features_current,axis=2))

            # if training (n,1,C) else (B, N, C) where n is based on (B,N) and the weak_label_mask
            f_query_feature_i = self.three_nearest_interpolation(xyz_query, xyz_support, features_support) # (B,N,C)
            f_query_feature_list.append(f_query_feature_i)

        # concat all features, (n, 1116, 1); the tricky here is n is as batch dim, 1116 as channel dim, 1 as num_pt dim
        FC_LIST =[256, 128, 64]
        if self.config.concat_type == '1234':
            features_combined = tf.concat(f_query_feature_list[:], axis=-1) # (n,1,928)
        elif self.config.concat_type == '123': 
            features_combined = tf.concat(f_query_feature_list[:3], axis=-1) # (n,1,x)
        elif self.config.concat_type == '234': 
            features_combined = tf.concat(f_query_feature_list[1:], axis=-1) # (n,1,x)
        elif self.config.concat_type == '12': 
            features_combined = tf.concat(f_query_feature_list[:2], axis=-1) # (n,1,x)
            FC_LIST =[128, 64]
        elif self.config.concat_type == '1': 
            features_combined = f_query_feature_list[0] # (n,1,x)
            FC_LIST =[16]
        else:
            raise NotImplementedError("error")

        # obtain classification scores using FCs, (n, 1, 928)-> ...-->(n, 1, num_classes) for training
        # or obtain classification scores using FCs, (B, N, 928)-> ...-->(B, N, num_classes) for validation
        f_current =features_combined
        for i in range(len(FC_LIST)):
            f_layer_fc_i = helper_tf_util.conv1d(f_current, FC_LIST[i], 1, f'fc{i+1}', 1, 'VALID', True, is_training)
            # add a dropout to its last layer
            if i == len(FC_LIST)-1:
                f_layer_drop = helper_tf_util.dropout(f_layer_fc_i, keep_prob=0.5, is_training=is_training, scope='dp1')
            f_current = f_layer_fc_i
        logits = helper_tf_util.conv1d(f_current, self.config.num_classes, 1, f'fc{len(FC_LIST)+1}', 1, 'VALID', False, is_training, activation_fn=None) 
        # ###########################Query Network############################

        # if training, logits's shape is like (n,1,C), if validation, shape like (B, N, C)
        logits=tf.cond(is_training,
                lambda: tf.squeeze(logits, [1]), # (n, num_classes)
                lambda: tf.reshape(logits,[-1, tf.shape(logits)[-1]])) # (B*N, num_classes)

        return logits, weak_points_labels # (n,num_classes), (n,)

    def train(self, dataset):

        # write current config to the log
        dict_config = json.dumps(dict((name, getattr(self.config, name)) for name in dir(self.config) if not name.startswith('__') and not name.startswith('lr')))
        log_out('****START TRAINING with {}****\n'.format(json.dumps(dict_config)), self.Log_file)
        log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)
        # use session to start the dataset iterator
        self.sess.run(dataset.train_init_op) # similar to sess.run(dataset.flat_inputs)??
        while self.training_epoch < self.config.max_epoch:
            t_start = time.time()
            try:
                ops = [self.train_op, # training optmizer
                       self.extra_update_ops,
                       self.merged, # tensorboard summary
                       self.loss,
                       self.logits,
                       self.labels,
                       self.weak_label_masks,
                       self.weak_labels,
                       self.accuracy]

                # logits, weak_labels = self.sess.run([self.logits, self.weak_labels], {self.is_training: True})
                # BUG: OOM issue reporting error OOM when allocating tensor with shape[40960,10240,32] for queired features concatenation step for the semantic query network(~line 450)  , 1 batch contains about 400 weakly points  
                _, _, summary, l_out, probs, labels, _, _, acc = self.sess.run(ops, {self.is_training: True})
                self.train_writer.add_summary(summary, self.training_step)
                t_end = time.time()
                if self.training_step % 50 == 0:
                    message = 'Step {:08d} L_out={:5.3f} Acc={:4.2f} ''---{:8.2f} ms/batch'
                    log_out(message.format(self.training_step, l_out, acc, 1000 * (t_end - t_start)), self.Log_file)
                self.training_step += 1

            except tf.errors.OutOfRangeError:
                # each training_step is 500, so if above this number, will trigger this exception(the below code)
                m_iou = self.evaluate(dataset)
                if m_iou > np.max(self.mIou_list):
                    # Save the best model
                    snapshot_directory = join(self.saving_path, 'snapshots')
                    makedirs(snapshot_directory) if not exists(snapshot_directory) else None
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step)
                self.mIou_list.append(m_iou)
                log_out('Best m_IoU is: {:5.3f}'.format(max(self.mIou_list)), self.Log_file)

                self.training_epoch += 1
                self.sess.run(dataset.train_init_op)
                # Update learning rate
                op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                           self.config.lr_decays[self.training_epoch]))
                self.sess.run(op)
                log_out('****EPOCH {}****'.format(self.training_epoch), self.Log_file)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1 / 0

        print('finished')
        self.sess.close()

    def evaluate(self, dataset):
        """For Sqn model, all test sub-sampled points will be used for evaluations. Note: only test on sub-sampled points rather raw pts.
        For each validation step:
            obtain current input's preds w. shape (B,N,13) and gt_labels (B,N) by running session (i.e., calculate logits using trained model)
            convert preds to hard labels w. shape (B,N)
            compute confusion_matrix, then compute {gt_class,positive_classes,true_positive_classes} for current input, add to their list
        use {gt_class,positive_classes,true_positive_classes} for the whole validation set to compute mIoU and save logs 
        """
        # Initialise iterator with validation data
        self.sess.run(dataset.val_init_op)

        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        for step_id in range(self.config.val_steps):
            if step_id % 50 == 0:
                print(str(step_id) + ' / ' + str(self.config.val_steps))
            try:
                ops = (self.prob_logits, self.weak_labels, self.accuracy)
                stacked_prob, weak_lbls, acc = self.sess.run(ops, {self.is_training: False})
                pred = np.argmax(stacked_prob, 1)
                # if not self.config.ignored_label_inds:
                #     pred_valid = pred
                #     labels_valid = labels
                # else:
                #     invalid_idx = np.where(labels == self.config.ignored_label_inds)[0]
                #     labels_valid = np.delete(labels, invalid_idx)
                #     labels_valid = labels_valid - 1
                #     pred_valid = np.delete(pred, invalid_idx)
                pred_valid = pred
                labels_valid = weak_lbls

                correct = np.sum(pred_valid == labels_valid)
                val_total_correct += correct
                val_total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.config.num_classes, 1))
                gt_classes += np.sum(conf_matrix, axis=1)
                positive_classes += np.sum(conf_matrix, axis=0)
                true_positive_classes += np.diagonal(conf_matrix)

            except tf.errors.OutOfRangeError:
                break

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_out('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.Log_file)
        log_out('mean IOU:{}'.format(mean_iou), self.Log_file)

        mean_iou = 100 * mean_iou
        log_out('Mean IoU = {:.1f}%'.format(mean_iou), self.Log_file)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_out('-' * len(s), self.Log_file)
        log_out(s, self.Log_file)
        log_out('-' * len(s) + '\n', self.Log_file)
        return mean_iou

    def get_loss_Sqn(self, logits, labels):
        """weighted CE loss (same as the get_loss(), but with my shape comments)
        Args:
            logits ([type]): logits, shape like: (B,N,K)
            labels ([type]): labels, shape like: (B,N) where each value is in [0,1,...,K-1]
        Returns:
            [type]: the loss
        """
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes) # (n,13)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        loss_value = tf.reduce_mean(losses)
        return loss_value

    def get_loss(self, logits, labels, pre_cal_weights):
        """weighted CE loss

        Args:
            logits ([type]): logits, shape like: (B,N,K)
            labels ([type]): labels, shape like: (B,N) where each value is in [0,1,...,K-1]
            pre_cal_weights ([type]): class weight, a list

        Returns:
            [type]: the loss
        """
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, name, is_training):
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_pc = self.building_block(xyz, f_pc, neigh_idx, d_out, name + 'LFA', is_training)
        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut)

    def building_block(self, xyz, feature, neigh_idx, d_out, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out // 2, name + 'att_pooling_1', is_training)

        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.att_pooling(f_concat, d_out, name + 'att_pooling_2', is_training)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def three_nearest_interpolation(xyz_query, xyz_support, features_support, k_interpolation=3):
        """need custom CUDA ops to support (three_nn(), three_interpolate())
        ----------
        xyz_query : Tensor
            (B, N, 3) tensor of the xyz positions of the unknown points
        xyz_support : Tensor
            (B, M, 3) tensor of the xyz positions of the known points (i.e. B PC examples, each is mx3 shape)
        features_support : Tensor
            (B, M, C) tensor of features to be propagated (i.e. B PC examples, each is mx3 shape)
        k_interpolation:
            the number of neighbors used for interpolation
        Returns
        -------
        new_features : torch.Tensor
            (B,N,C) tensor of the features of the weakly points' features(i.e., n weakly points' new features)
        """
        if xyz_support is not None:
            dist, idx = three_nn(xyz_query, xyz_support) # (B,N,3), (B,N,3)
            dist_recip = 1.0 / (dist + 1e-8) # (B,N,3)
            norm = tf.reduce_sum(dist_recip, axis=2, keepdims=True) # (B,N,1)
            weight = dist_recip / norm # (B,N,3)
            features_interpolated = three_interpolate(features_support, idx, weight) # (B,N,C)
        else:
            raise ValueError('make sure the known parameters are valid')
        return features_interpolated # (B,N,C)

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, 1, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index, assume up_num_points >= N
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2) # (B,N,d)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points]) # (B, up_num_points)
        interpolated_features = tf.batch_gather(feature, interp_idx) # (B,up_num_points,d)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2) # (B,up_num_points,1,d)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features

    @staticmethod
    def att_pooling(feature_set, d_out, name, is_training):
        batch_size = tf.shape(feature_set)[0]
        num_points = tf.shape(feature_set)[1]
        num_neigh = tf.shape(feature_set)[2]
        d = feature_set.get_shape()[3].value
        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])
        att_activation = tf.layers.dense(f_reshaped, d, activation=None, use_bias=False, name=name + 'fc')
        att_scores = tf.nn.softmax(att_activation, axis=1)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, d])
        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)
        return f_agg


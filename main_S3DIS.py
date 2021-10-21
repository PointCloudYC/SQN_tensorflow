from os.path import join
from RandLANet import Network
from tester_S3DIS import ModelTester
from helper_ply import read_ply
# S3DIS's configs
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os


class S3DIS:
    """S3DIS dataset class w/o inheriting any TensorFlow built-in classes
    Despite not inheriting any TensorFlow built-in classes, this class follow the tensorflow's dataset pattern: input pipeline w. iterator and initilizer
    - __int__(): initialize the dataset basic settings, e.g., dataset path, test_area_idx=5, classes, categories, and physical file paths, etc.Then it will call load_sub_sampled_clouds().
    - load_sub_sampled_clouds(): load S3DIS dataset physical sub-sampled files as training and test clouds; (note: these sub-sub-sampled files are prepared by the data_prepare_s3dis.py)
    - init_input_pipeline(): create tensorflow built-in dataset object using the `from_generator` method, then create its iterator and train,val_init_op operator for session running.
    - get_batch_gen(): use for the above init_input_pipeline() to obtain batch data
    - get_tf_mapping2(): use for the above init_input_pipeline() to organize each stage's points,neighbors,pools and up_samples into a list.
    """
    def __init__(self, test_area_idx):
        self.name = 'S3DIS'
        # self.path = '/data/S3DIS'
        self.path = 'data/S3DIS'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply')) # scan the folder for all ply files

        # Initiate containers
        # validation projection indice list, each item represents projection nnest id over a sub_pc for each corresponding raw pc pt-yc
        self.val_proj = []
        # validation labels list, each item represent a validation sub_pc's label-yc
        self.val_labels = []
        # possibility for control to randomly choose a point in the sub_pc evenly-yc
        self.possibility = {}
        self.min_possibility = {}
        # {training,validation} sub_pc's kd_trees, colors, labels and names-yc
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        # fill the above containers by reading physical sub_pc files-yc
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        """load sub_sampled physical files and fill all the containers, input_{trees, colors, labels, names} and val_{proj,labels}-yc

        Args:
            sub_grid_size ([type]): sub-sampling grid size, e.g., 0.040
        """
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name)) # e.g., Area_1_conferenceRoom_1_KDTree.pkl
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name)) # e.g., Area_1_conferenceRoom_1.ply

            data = read_ply(sub_ply_file) # ply format: x,y,z,red,gree,blue,class
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T # (N',3), note the transpose symbol
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # input_xx is a dict contain training or validation info for all sub_pc, each of them contain a list
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices and labels (this is useful for validating on all raw points)
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    # Generate the input data flow
    # Intuitively, it prepare data training examples, each pc will generate numerous point cloud training/validation examples by selecting a center point in the pc evenly, then select center point's neighboring points within a radius but not more than a threshold (e.g.,10000)-yc
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size
        # assign a possibility for all sub_pc and their containing points-yc
        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        """mapping for tranlating dataset's tensor to another form-yc
        The params of tf_map() just corresponds to {xyz,features,labels,idx,cloud_idx};
        Considering there are cfg.num_layers(e.g., 4) stages in the encoder, each stage will have a sub-sampling process, use a list (named flat_inputs) for managing all of them (i.e., sub_sampled point cloud info at these stages). For example, if we have 4 sub-sampling processes, the flat_inputs list will have 20 items, like: [input_points, input_neighbors, input_pools, input_up_samples, batch_features, batch_labels, batch_pc_idx(i.e. points idx in the cloud),batch_cloud_idx], so 4*(cfg.num_layers+1) items in total
        Returns:
            [type]: [description]
        """
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32) # (B,N,k)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :] # retrieve first N/sub_sampling_ratio pts
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :] # sub_sampled points' id
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32) # (B,N,K) over the sub_points
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list # contains: [input_points, input_neighbors, input_pools, input_up_samples, batch_features, batch_labels, batch_pc_idx(i.e. points idx in the cloud),batch_cloud_idx], so 4*(cfg.num_layers+1) items in total

        return tf_map

    def init_input_pipeline(self):
        """
        obtain X,Y pair and {train,val}_init_op operator following tensorflow pipline pattern.
        """
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes) # create the dataset from a generator
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size) # batch the dataset object
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func) # map to another form, each batch is a list containing 4*(num_layers+1) items corresponding to points, features, labels, cloud_idx, point_idx at different sub-sampled stages for each batch
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next() # iterator, each returns a flat_inputs list containing 20 items
        # prepare operator for session to run
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    test_area = FLAGS.test_area

    # create S3DIS dataset object using test_area as validation/test set, the rest as training set-yc
    dataset = S3DIS(test_area)
    dataset.init_input_pipeline()

    """provide 3 functionality: training, testing and visualization-yc
    - training; pass the dataset object and dataset config to create the Network object, then start training
    - testing; pass in the model checkpoint and create ModelTest object, then start testing
    - visualization; plot the raw pc and sub_pc
    """
    if Mode == 'train':
        # NOTE: cfg is S3DIS object w. common configs, a global variable here.
        model = Network(dataset, cfg) 
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
    else:
        ##################
        # Visualize data #
        ##################

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(dataset.train_init_op)
            while True:
                flat_inputs = sess.run(dataset.flat_inputs)
                pc_xyz = flat_inputs[0]
                sub_pc_xyz = flat_inputs[1]
                labels = flat_inputs[21]
                Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
                Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])

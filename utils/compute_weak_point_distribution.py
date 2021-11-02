"""
compute weak point distribution under different weak settings
Author: Chao YIN
Email: cyinac@connect.ust.hk

history: 
- Nov. 1, 2021, init the file
"""

import os, sys, glob, pickle, argparse, random
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply, read_ply
from helper_tool import DataProcessing as DP

def compute_num_points(anno_path, save_path, sub_pc_folder, 
                   weak_label_folder, weak_label_ratio, sub_grid_size, 
                   gt_class, gt_class2label):
                   
    num_raw_points = 0 # number of raw points for current room
    num_sub_points = 0 # number of sub-sampled points for current room

    # save raw_cloud
    if not os.path.exists(save_path):
        raise NotImplementedError("run the dataset_prepare_s3dis_sqn.py to generate raw and sub PC")
    else:
        # if existed then read this ply file to fill the data/xyz/colors/labels 
        data = read_ply(save_path) # ply format: x,y,z,red,gree,blue,class
        xyz = np.vstack((data['x'], data['y'], data['z'])).T # (N',3), note the transpose symbol
        num_raw_points = xyz.shape[0]
        colors = np.vstack((data['red'], data['green'], data['blue'])).T # (N',3), note the transpose symbol
        labels = data['class']
        pc_label =  np.concatenate((xyz, colors, np.expand_dims(labels, axis=1)),axis=1) # (N,7)


    # save sub_cloud
    sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
    if not os.path.exists(sub_ply_file):
        raise NotImplementedError("run the dataset_prepare_s3dis_sqn.py to generate raw and sub PC")
    else:
        data = read_ply(sub_ply_file) # ply format: x,y,z,red,gree,blue,class
        sub_xyz = np.vstack((data['x'], data['y'], data['z'])).T # (N',3), note the transpose symbol
        num_sub_points = sub_xyz.shape[0]
        sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T # (N',3), note the transpose symbol
        sub_labels = data['class']


    # weak labels
    weak_label_sub_file = join(weak_label_folder, save_path.split('/')[-1][:-4] + '_sub_weak_label.ply')
    if not os.path.exists(weak_label_sub_file):
        raise NotImplementedError("run the dataset_prepare_s3dis_sqn.py to generate weak labels")
    else:
        data = read_ply(weak_label_sub_file) 
        weak_label_mask = data['weak_mask']

    # compute number of points for each class in this room
    num_points_raw = np.zeros(len(gt_class), dtype=np.int32)
    num_points_sub = np.zeros(len(gt_class), dtype=np.int32)
    num_points_weak = np.zeros(len(gt_class), dtype=np.int32)
    # raw
    for i,item in enumerate(labels):
        num_points_raw[int(item)]+=1
    # sub-sampled points
    for i,item in enumerate(sub_labels):
        num_points_sub[int(item)]+=1
    # weak points
    for i,item in enumerate(sub_labels[weak_label_mask.astype(bool)]):
        num_points_weak[int(item)]+=1
    
    return num_points_raw, num_points_sub, num_points_weak


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--rng_seed", type=int, default=123, help='manual seed')
    parser.add_argument('--dataset_path', type=str, default='./data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version', help='dataset path')
    parser.add_argument('--sub_grid_size', type=float, default=0.04, help='grid-sampling size')
    parser.add_argument('--weak_label_ratio', type=float, default=0.001, help='the weakly semantic segmentation ratio')
    FLAGS = parser.parse_args()

    # set fixed seeds for reproducible results
    random.seed(FLAGS.rng_seed)
    np.random.seed(FLAGS.rng_seed)

    dataset_path = FLAGS.dataset_path
    anno_paths = [line.rstrip() for line in open(join(BASE_DIR, 'meta/anno_paths.txt'))]
    anno_paths = [join(dataset_path, p) for p in anno_paths] # each room's path
    # object categories for the S3DIS dataset
    gt_class = [x.rstrip() for x in open(join(BASE_DIR, 'meta/class_names.txt'))]
    gt_class2label = {cls: i for i, cls in enumerate(gt_class)}
    sub_grid_size = FLAGS.sub_grid_size # grid_subsampling size

    """ 
    create 3 folder
    - input_0.040, for sub_pc.py, kdtree for sub_pc and the projection indices
    - original_ply, raw_pc.ply
    - weak_label_0.01, weak labels for raw and sub_pc
    """
    original_pc_folder = join(dirname(dataset_path), 'original_ply')
    sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(sub_grid_size))
    weak_label_ratio = FLAGS.weak_label_ratio
    weak_label_folder = join(dirname(dataset_path), 'weak_label_{}'.format(weak_label_ratio))
    os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
    os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
    os.mkdir(weak_label_folder) if not exists(weak_label_folder) else None
    out_format = '.ply'

    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    num_points_each_class_raw = np.zeros(len(gt_class), dtype=np.int32)
    num_points_each_class_sub = np.zeros(len(gt_class), dtype=np.int32)
    num_points_each_class_weak = np.zeros(len(gt_class), dtype=np.int32)

    for annotation_path in anno_paths:

        # e.g.: data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/Annotations
        print(annotation_path) 
        elements = str(annotation_path).split('/')
        # e.g.: Area_1_conferenceRoom_1.ply
        out_file_name = elements[-3] + '_' + elements[-2] + out_format

        # convert each room's pc to ply and more(kdtree and projection indices for raw points over its corresponding sub_pc)
        num_points_room_raw, num_points_room_sub, num_points_room_weak = compute_num_points(annotation_path, 
                                        join(original_pc_folder, out_file_name), 
                                        sub_pc_folder, weak_label_folder, 
                                        weak_label_ratio, sub_grid_size, 
                                        gt_class, gt_class2label)

        num_points_each_class_raw += num_points_room_raw
        num_points_each_class_sub += num_points_room_sub
        num_points_each_class_weak += num_points_room_weak

    print("finish computing!")
    print(f"num_points for raw:{num_points_each_class_raw}\n")
    print(f"num_points for sub:{num_points_each_class_sub}\n")
    print(f"num_points for weak:{num_points_each_class_weak}")

    import xlsxwriter
    rows = [
        ['type'] + list(gt_class2label.values()),
        ['The number of raw points'] + num_points_each_class_raw.tolist(),
        ['The number of sub points'] + num_points_each_class_sub.tolist(),
        ['The number of weak points'] + num_points_each_class_weak.tolist(),
    ]
    filename_xls = join(dirname(dataset_path), f'S3DIS_point_distribution_sub_{sub_grid_size:.3f}_weak_{weak_label_ratio}.xlsx')
    print('Save file to {}'.format(filename_xls))
    with xlsxwriter.Workbook(filename_xls) as workbook:
        worksheet = workbook.add_worksheet()
        for i, data in enumerate(rows):
            worksheet.write_row(i, 0, data)
    print('Finished.')

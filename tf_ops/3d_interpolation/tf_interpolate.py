import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

# load custom tf interpolate lib
interpolate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so.so'))

def three_nn(xyz_query, xyz_support):
    '''find xyz_query's nearest 3 neighbors of xyz_support
    Input:
        xyz_query: (b,n,3) float32 array, unknown/query points
        xyz_support: (b,m,3) float32 array, known/support points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.three_nn(xyz_query, xyz_support)

ops.NoGradient('ThreeNN')

def three_interpolate(features_support, query_idx_over_support, weight):
    '''interpolate features for the xyz_query(determined by idx)
    Input:
        features_support: (b,m,c) float32 array, known/support features of the corresponding xyz_support
        query_idx_over_support: (b,n,3) int32 array, indices of nearest 3 neighbors in the known/support points for each query point
        weight: (b,n,3) float32 array, weights for query_idx_over_support

    Output:
        out: (b,n,c) float32 array, interpolated point features
    '''
    return interpolate_module.three_interpolate(features_support, query_idx_over_support, weight)

@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]

if __name__=='__main__':

    import numpy as np
    import time
    np.random.seed(100)
    features = np.random.random((32,128,64)).astype('float32')
    xyz1 = np.random.random((32,512,3)).astype('float32')
    xyz2 = np.random.random((32,128,3)).astype('float32')

    with tf.device('/cpu:0'):
        points = tf.constant(features)
        xyz1 = tf.constant(xyz1)
        xyz2 = tf.constant(xyz2)
        dist, idx = three_nn(xyz1, xyz2)
        weight = tf.ones_like(dist)/3.0
        interpolated_points = three_interpolate(points, idx, weight)

    with tf.Session('') as sess:
        now = time.time() 
        for _ in range(100):
            ret = sess.run(interpolated_points)
        print(time.time() - now)
        print(ret.shape, ret.dtype)
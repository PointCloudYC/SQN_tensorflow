import os
from os import makedirs
from os.path import exists, join
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from helper_tool import DataProcessing as DP
import helper_tf_util
import time

# custom tf ops based on PointNet++(https://github.com/charlesq34/pointnet2)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
from tf_sampling import farthest_point_sample, gather_point
from tf_interpolate import three_nn, three_interpolate

print(gather_point)

print(three_nn)
print(three_interpolate)
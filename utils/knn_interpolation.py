"""
Author: Chao YIN
Email: cyinac@connect.ust.hk
Date: Oct. 23, 2021
code base: pytorch geometric,https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/unpool/knn_interpolate.html#knn_interpolate
"""

import numpy as np
import torch
# need install torch geometric
from torch_geometric.nn import knn
from torch_scatter import scatter_add


# interpolate over x (N,C with shape (N,3)) located by pos_x to obtain y (located by pos_y with shape (M,3)) leading a tensor (N,C)
def knn_interpolate(support_features, support_points, query_points, support_x=None, support_y=None, k=3, num_workers=1):
    """KNN interpolation, copied from , https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/unpool/knn_interpolate.html#knn_interpolate
    Args:
        support_features ([type]): [description]
        support_points ([type]): [description]
        query_points ([type]): [description]
        support_x ([type], optional): [description]. Defaults to None.
        support_y ([type], optional): [description]. Defaults to None.
        k (int, optional): [description]. Defaults to 3.
        num_workers (int, optional): [description]. Defaults to 1.
    Returns:
        [type]: [description]
    """

    with torch.no_grad():
        assign_index = knn(support_points, query_points, k, batch_x=support_x, batch_y=support_y,
                           num_workers=num_workers)
        y_idx, x_idx = assign_index
        diff = support_points[x_idx] - query_points[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y = scatter_add(support_features[x_idx] * weights, y_idx, dim=0, dim_size=query_points.size(0))
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=query_points.size(0))

    return y

def batch_knn_interpolate(batch_support_features, batch_support_points, batch_query_points,k=3,num_workers=1):
    """[summary]
    Args:
        batch_support_features ([type]): (B,N,C)
        batch_support_points ([type]): (B,N,3)
        batch_query_points ([type]): (B,M,3)
        k (int, optional): [description]. Defaults to 3.
        num_workers (int, optional): [description]. Defaults to 1.
        return query features, (B,M,C)
    """
    query_feature_list=[]
    for i in range(batch_support_features.shape[0]):
        y=knn_interpolate(batch_support_features[i], batch_support_points[i], batch_query_points[i], support_x=None, support_y=None, k=3, num_workers=1)
        query_feature_list.append(y)

    query_features=torch.tensor(query_feature_list)

    return query_features # (B,M,C)


class BatchKnnInterpolate(object):

    def __init__(self, k=3, num_workers=1):
        self.k = k
        self.num_workers = num_workers

    def __call__(self, query_points, support_points, support_features):
        query_feature_list=[]
        batch_size = query_points.size()[0]
        for i in range(batch_size):
            query_features_cur=knn_interpolate(support_features[i], support_points[i], query_points[i], 
                                               k=self.k, num_workers=self.num_workers)
            query_feature_list.append(query_features_cur)

        query_features=torch.tensor(query_feature_list)
        return query_features # (B,M,C)

if __name__ == "__main__":

    batch_knn_op = BatchKnnInterpolate(k=3, num_workers=1)
    query_points = np.random.rand(4,2,3)
    support_points = np.random.rand(4,10,3)
    support_features = np.random.rand(4,10,5)

    query_features = batch_knn_op(query_points, support_points, support_features)

    print(query_features.shape, '\n', query_features)
import torch
import torch.nn as nn
# import torch_scatter as ts
# import pytorch_lightning as pl

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ocpmodels.preprocessing.vn_layers.vn_layers import (
    VNBatchNorm,
    VNLinearLeakyReLU,
    VNMaxPool,
    mean_pool,
    VNStdFeature,
)
    

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Performs k-nearest neighbors search on a given set of points.

    Args:
        x (torch.Tensor): The input tensor representing a set of points.
            Shape: (batch_size, num_points, num_dimensions).
        k (int): The number of nearest neighbors to find.

    Returns:
        torch.Tensor: The indices of the k nearest neighbors for each point in x.
            Shape: (batch_size, num_points, k).
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)

    idx_base = torch.arange(0, batch_size).type_as(idx).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


def get_graph_feature_cross(
    x: torch.Tensor, 
    k: int = 5, 
    idx: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computes the graph feature cross for a given input tensor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, num_dims, num_points).
        k (int, optional): The number of nearest neighbors to consider. Defaults to 20.
        idx (torch.Tensor, optional): The indices of the nearest neighbors. Defaults to None.

    Returns:
        torch.Tensor: The computed graph feature cross tensor of shape (batch_size, num_dims*3, num_points, k).

    """
    # breakpoint()
    # nb_atoms = x.size(0)
    # if idx is None:
    #     idx = knn(x.squeeze(1), k=k)

    # feature = x.squeeze(1)[idx, :]
    # x_rep = x.repeat(1, k, 1)
    # cross = torch.cross(feature, x_rep, dim=2)
    # feature = torch.cat((feature - x_rep, x_rep, cross), dim=2)
    # return feature.view(nb_atoms, -1, k)

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size).type_as(idx).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)

    feature = (
        torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    )

    return feature

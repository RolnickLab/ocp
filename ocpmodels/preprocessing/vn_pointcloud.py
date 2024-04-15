import torch
import torch.nn as nn
# import torch_scatter as ts
# import pytorch_lightning as pl

from typing import Optional

import torch
import torch.nn as nn

from ocpmodels.preprocessing.vn_layers.vn_layers import (
    VNBatchNorm,
    VNLinearLeakyReLU,
    VNMaxPool,
    mean_pool,
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


class VNSmall(torch.nn.Module):
    """
    VNSmall is a small variant of the vector neuron equivariant network used for canonicalization of point clouds.

    Attributes:
        n_knn (int): Number of nearest neighbors to consider.
        pooling (str): Pooling type to use, either "max" or "mean".
        conv_pos (VNLinearLeakyReLU): Convolutional layer for positional encoding.
        conv1 (VNLinearLeakyReLU): First convolutional layer.
        bn1 (VNBatchNorm): Batch normalization layer.
        conv2 (VNLinearLeakyReLU): Second convolutional layer.
        dropout (nn.Dropout): Dropout layer.
        pool (Union[VNMaxPool, mean_pool]): Pooling layer.

    Methods:
        __init__: Initializes the VNSmall network.
        forward: Forward pass of the VNSmall network.

    """

    def __init__(self, n_knn=5, pooling="mean"):
        """
        Initialize the VN Small network.

        Args:
            k_nn (int, optional): Number of nearest neighbors to consider.
            pooling (str, optional): Pooling type to use, either "max" or "mean".

        Raises:
            ValueError: If the specified pooling type is not supported.
        """
        super().__init__()
        self.n_knn = n_knn
        self.pooling = pooling
        self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
        self.bn1 = VNBatchNorm(64 // 3, dim=4)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 12 // 3, dim=4, negative_slope=0.0)
        self.dropout = nn.Dropout(p=0.5)

        if self.pooling == "max":
            self.pool = VNMaxPool(64 // 3)
        elif self.pooling == "mean":
            self.pool = mean_pool
        else:
            raise ValueError(f"Pooling type {self.pooling} not supported")

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VNSmall network.

        For every pointcloud in the batch, the network outputs three vectors that transform equivariantly with respect to SO3 group.

        Args:
            point_cloud (torch.Tensor): Input point cloud tensor made to shape (batch_size, num_points, 3).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3, 3).

        """
        if len(point_cloud.size()) == 2:
            point_cloud = point_cloud.unsqueeze(0).transpose(2,1)
        else:
            point_cloud = point_cloud.transpose(2, 1)

        batch_size, num_points, _ = point_cloud.size()
        n_knn = min(self.n_knn, num_points - 1)

        feat = get_graph_feature_cross(point_cloud, k=n_knn)
        out = self.conv_pos(feat)
        out = self.pool(out)

        out = self.bn1(self.conv1(out))
        out = self.conv2(out)
        out = self.dropout(out)
        
        return out.mean(dim=-1)[:, :3]


# class VNDeeper(torch.nn.Module):
#     """
#     VNDeeper is a deeper variant of the vector neuron equivariant network used for canonicalization of point clouds.

#     Attributes:
#         n_knn (int): Number of nearest neighbors to consider.
#         pooling (str): Pooling type to use, either "max" or "mean".
#         conv_pos (VNLinearLeakyReLU): Convolutional layer for positional encoding.
#         conv1 (VNLinearLeakyReLU): First convolutional layer.
#         bn1 (VNBatchNorm): Batch normalization layer.
#         conv2 (VNLinearLeakyReLU): Second convolutional layer.
#         conv3 (VNLinearLeakyReLU): Third convolutional layer.
#         dropout (nn.Dropout): Dropout layer.
#         pool (Union[VNMaxPool, mean_pool]): Pooling layer.

#     Methods:
#         __init__: Initializes the VN Deeper network.
#         forward: Forward pass of the VN Deeper network.

#     """

#     def __init__(self, n_knn=5, pooling="mean"):
#         """
#         Initialize the VN Deeper network.

#         Args:
#             k_nn (int, optional): Number of nearest neighbors to consider.
#             pooling (str, optional): Pooling type to use, either "max" or "mean".

#         Raises:
#             ValueError: If the specified pooling type is not supported.
#         """
#         super().__init__()
#         self.n_knn = n_knn
#         self.pooling = pooling
#         self.conv_pos = VNLinearLeakyReLU(3, 64 // 3, dim=5, negative_slope=0.0)
#         self.conv1 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
#         self.bn1 = VNBatchNorm(64 // 3, dim=4)
#         self.conv2 = VNLinearLeakyReLU(64 // 3, 64 // 3, dim=4, negative_slope=0.0)
#         self.conv3 = VNLinearLeakyReLU(64 // 3, 12 // 3, dim=4, negative_slope=0.0)
#         self.dropout = nn.Dropout(p=0.5)

#         if self.pooling == "max":
#             self.pool = VNMaxPool(64 // 3)
#         elif self.pooling == "mean":
#             self.pool = mean_pool  # type: ignore
#         else:
#             raise ValueError(f"Pooling type {self.pooling} not supported")
    
#     def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the VNDeeper network.

#         For every pointcloud in the batch, the network outputs three vectors that transform equivariantly with respect to SO3 group.

#         Args:
#             point_cloud (torch.Tensor): Input point cloud tensor made to shape (batch_size, num_points, 3).

#         Returns:
#             torch.Tensor: Output tensor of shape (batch_size, 3, 3).

#         """
#         if len(point_cloud.size()) == 2:
#             point_cloud = point_cloud.unsqueeze(0).transpose(2,1)
#         else:
#             point_cloud = point_cloud.transpose(2, 1)

#         batch_size, num_points, _ = point_cloud.size()
#         n_knn = min(self.n_knn, num_points - 1)

#         feat = get_graph_feature_cross(point_cloud, k=n_knn)
#         out = self.conv_pos(feat)
#         out = self.pool(out)

#         out = self.bn1(self.conv1(out))
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.dropout(out)

#         return out.mean(dim=-1)[:, :3]
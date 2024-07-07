import torch
import torch.nn as nn
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
from ocpmodels.preprocessing.vn_layers.vn_utils import get_graph_feature, get_graph_feature_cross



class VNSmall(torch.nn.Module):
    """
    A very simple VN model
    """
    def __init__(self, n_knn=50, pooling="mean"):
        super().__init__()
        self.n_knn = n_knn
        self.pooling = pooling
        self.pos_enc = VNLinearLeakyReLU(3, 3, dim=5, negative_slope=0.0)
        self.bn = VNBatchNorm(3, dim=5)

        if self.pooling == "max":
            self.pool = VNMaxPool(3)
        elif self.pooling == "mean":
            self.pool = mean_pool
        else:
            raise ValueError(f"Pooling type {self.pooling} not supported")
        
    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0).transpose(2,1)
        else:
            x = x.transpose(2, 1)

        batch_size, _, num_points = x.size()
        n_knn = min(self.n_knn, num_points - 1)

        x = get_graph_feature_cross(x, k=n_knn)
        x = self.pos_enc(x)
        # x = self.bn(x)
        return self.pool(self.pool(x))


class VNPointnet(torch.nn.Module):
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

        out = self.bn1(self.conv1(out)) # rm for 2 layers
        out = self.conv2(out)
        out = self.dropout(out)
        
        return out.mean(dim=-1)[:, :3]


# Adapted from Deng et al. (2021)
class VN_dgcnn(torch.nn.Module):
    def __init__(self, pooling="mean", n_knn=5, num_class=3, normal_channel=False):
        super().__init__()
        self.n_knn = n_knn       
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True)
        self.conv6 = VNLinearLeakyReLU(1024//3*2, 3, dim=4, share_nonlinearity=True)

        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        
        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0).transpose(2,1)
        else:
            x = x.transpose(2, 1)

        batch_size, _, num_points = x.size()
        x = x.unsqueeze(1)

        n_knn = min(self.n_knn, num_points - 1)

        x = get_graph_feature(x, k=n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        
        x = torch.cat((x, x_mean), 1)
        x, _ = self.std_feature(x)
        x = self.conv6(x)
        return mean_pool(x)
    
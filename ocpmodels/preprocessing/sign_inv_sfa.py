import random
from copy import deepcopy
from itertools import product
from ocpmodels.common.graph_transforms import RandomRotate

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing

from ocpmodels.preprocessing.vn_layers.vn_utils import knn

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))


    def forward(self, x, edge_index):
        return self.encode(x, edge_index)

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

class GCNSignNet(nn.Module):
    def __init__(self, num_eigs=3, hidden_channels=64, out_channels=3):
        super().__init__()
        phi_out_dim = hidden_channels // num_eigs
        self.phi = GCN(1, hidden_channels, phi_out_dim)
        self.rho = GCN(num_eigs * phi_out_dim, hidden_channels, out_channels)

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        breakpoint()
        n_knn = min(10, x.shape[0] - 1)
        mat_knn = knn(x.unsqueeze(0).transpose(1,2), n_knn)
        
        row, col = torch.meshgrid(torch.arange(mat_knn.size(1)), torch.arange(n_knn), indexing='ij')
        edge_index = torch.stack([row.reshape(-1), mat_knn[0].reshape(-1)], dim=0)

        x = x.unsqueeze(-1) # n x k x 1
        x = x.transpose(0,1) # k x n x 1
        x = self.phi(x, edge_index) + self.phi(-x, edge_index)
        x = x.transpose(0,1) # n x k x d
        x = x.reshape(x.shape[0], -1) # n x kd
        x = self.rho(x, edge_index) # n x d
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    

class SignNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(SignNet, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.kappa = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.mu(self.kappa(x) + self.kappa(-x))


class SignEquivariantNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(SignEquivariantNet, self).__init__()
        self.sign_net = SignNet(input_dim, output_dim)
        self.W1 = nn.Parameter(torch.randn(input_dim, input_dim))
        self.W2 = nn.Parameter(torch.randn(input_dim, input_dim))
        self.W3 = nn.Parameter(torch.randn(input_dim, input_dim))

    def forward(self, u):
        sign_net_output = self.sign_net(u)
        return torch.cat([self.W1 @ u[0, None].T, 
                          self.W2 @ u[1, None].T, 
                          self.W3 @ u[2, None].T], dim=1) * sign_net_output

def compute_frames(training, network, eigenvec, pos, cell, fa_method="random", pos_3D=None, det_index=0):
    """Compute all frames for a given graph.

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): centered position vector
        cell (tensor): cell direction (dxd)
        fa_method (str): the Frame Averaging (FA) inspired technique
            chosen to select frames: stochastic-FA (random), deterministic-FA (det),
            Full-FA (all) or SE(3)-FA (se3).
        pos_3D: for 2D FA, pass atoms' 3rd position coordinate.

    Returns:
        list: 3D position tensors of projected representation
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    all_fa_pos = []
    all_cell = []
    all_rots = []
    assert fa_method in {
        "all",
        "random",
        "det",
        "se3-all",
        "se3-random",
        "se3-det",
    }
    se3 = fa_method in {
        "se3-all",
        "se3-random",
        "se3-det",
    }
    fa_cell = deepcopy(cell)

    # if fa_method == "det" or fa_method == "se3-det":
    #     sum_eigenvec = torch.sum(eigenvec, axis=0)
    #     plus_minus_list = [torch.where(sum_eigenvec >= 0, 1.0, -1.0)]
    
    plus_minus_list = list(product([1, -1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]

    index = random.randint(0, len(plus_minus_list) - 1)
    random_pm = plus_minus_list[index]
    random_pm = random_pm.to(eigenvec.device)

    sign_inv_net = network
    if not training:
        for param in sign_inv_net.parameters():
            param.requires_grad = False

    eigenvec = random_pm * eigenvec
    eigenvec = sign_inv_net.to(eigenvec.device)(eigenvec)

    fa_pos = pos @ eigenvec

    if pos_3D is not None:
        full_eigenvec = torch.eye(3)
        fa_pos = torch.cat((fa_pos, pos_3D.unsqueeze(1)), dim=1)
        full_eigenvec[:2, :2] = eigenvec
        eigenvec = full_eigenvec

    if cell is not None:
        fa_cell = cell @ eigenvec
    
    return [fa_pos], [fa_cell], [eigenvec.unsqueeze(0)]

def check_constraints(eigenval, eigenvec, dim=3):
    """Check requirements for frame averaging are satisfied

    Args:
        eigenval (tensor): eigenvalues
        eigenvec (tensor): eigenvectors
        dim (int): 2D or 3D frame averaging
    """
    # Check eigenvalues are different
    if dim == 3:
        if (eigenval[1] / eigenval[0] > 0.90) or (eigenval[2] / eigenval[1] > 0.90):
            print("Eigenvalues are quite similar")
    else:
        if eigenval[1] / eigenval[0] > 0.90:
            print("Eigenvalues are quite similar")

    # Check eigenvectors are orthonormal
    if not torch.allclose(eigenvec @ eigenvec.T, torch.eye(dim), atol=1e-03):
        print("Matrix not orthogonal")

    # Check determinant of eigenvectors is 1
    if not torch.allclose(torch.linalg.det(eigenvec), torch.tensor(1.0), atol=1e-03):
        print("Determinant is not 1")



def frame_averaging_3D(network, training, pos, cell=None, fa_method="random", check=False):
    pos = pos - pos.mean(dim=0, keepdim=True)
    C = torch.matmul(pos.t(), pos)
    eigenval, eigenvec = torch.linalg.eigh(C)
    idx = eigenval.argsort(descending=True)
    eigenvec = eigenvec[:, idx]
    eigenval = eigenval[idx]

    if check:
        check_constraints(eigenval, eigenvec, 3)

    fa_pos, fa_cell, fa_rot = compute_frames(
        training, network, eigenvec, pos, cell, fa_method
    )
    return fa_pos, fa_cell, fa_rot


def frame_averaging_2D(network, training, pos, cell=None, fa_method="random", check=False):
    # Compute centroid and covariance
    pos_2D = pos[:, :2] - pos[:, :2].mean(dim=0, keepdim=True)
    C = torch.matmul(pos_2D.t(), pos_2D)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)
    # Sort eigenvalues
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Check if constraints are satisfied
    if check:
        check_constraints(eigenval, eigenvec, 3)

    # Compute all frames
    fa_pos, fa_cell, fa_rot = compute_frames(
        training, network, eigenvec, pos_2D, cell, fa_method, pos[:, 2]
    )
    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


def data_augmentation(g, d=3, *args):
    """Data augmentation where we randomly rotate each graph
    in the dataloader transform

    Args:
        g (data.Data): single graph
        d (int): dimension of the DA rotation (2D around z-axis or 3D)
        rotation (str, optional): around which axis do we rotate it.
            Defaults to 'z'.

    Returns:
        (data.Data): rotated graph
    """

    # Sampling a random rotation within [-180, 180] for all axes.
    if d == 3:
        transform = RandomRotate([-180, 180], [0, 1, 2])  # 3D
    else:
        transform = RandomRotate([-180, 180], [2])  # 2D around z-axis

    # Rotate graph
    graph_rotated, _, _ = transform(g)

    return graph_rotated
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.inits import ones, zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter

import pickle
import matplotlib.pyplot as plt
import numpy as np


class GraphNormInference(torch.nn.Module):
    r"""Applies graph normalization over individual graphs as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
        \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
        + \epsilon}} \odot \gamma + \beta

    where :math:`\alpha` denotes parameters that learn how much information
    to keep in the mean.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.empty(in_channels))
        self.bias = torch.nn.Parameter(torch.empty(in_channels))
        self.mean_scale = torch.nn.Parameter(torch.empty(in_channels))

        self.reset_parameters()

        self.training_means = {}
        self.training_vars = {}
        for interaction_block_idx in range(5):
            with open(f'ocpmodels/datasets/embeddings/60_train_graphnorm_mean_{interaction_block_idx}.pickle', 'rb') as handle:
                self.training_means[interaction_block_idx] = pickle.load(handle)
            with open(f'ocpmodels/datasets/embeddings/60_train_graphnorm_var_{interaction_block_idx}.pickle', 'rb') as handle:
                self.training_vars[interaction_block_idx] = pickle.load(handle)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        ones(self.weight)
        zeros(self.bias)
        ones(self.mean_scale)


    def forward(self, x: Tensor, idx: Optional[int]=0, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        # print("[graphnorm_inference] in forward")
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
            batch_size = 1

        if batch_size is None:
            batch_size = int(batch.max()) + 1

        # f = open("ocpmodels/datasets/embeddings/graphnorm_info.txt", "r")
        # s = int(f.read())
        # print(f"{s=}")
        mean = self.training_means[idx]
        var = self.training_vars[idx]
        
        # mean_from_batch = scatter(x, batch, 0, batch_size, reduce='mean')
        # with open(f'ocpmodels/datasets/embeddings/train_batch_0{s+1}_graphnorm_mean_{idx}.pickle', 'wb') as handle:
        #     pickle.dump(mean_from_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"mean_per_batch: {torch.linalg.norm(mean_per_batch.flatten(),ord=torch.inf)=}")
        # print(f"mean_from_pickle: {torch.linalg.norm(mean.flatten(),ord=torch.inf)=}")
        out = x - mean.index_select(0, batch) * self.mean_scale
        
        # var_from_batch = scatter(out.pow(2), batch, 0, batch_size, reduce='mean')
        # with open(f'ocpmodels/datasets/embeddings/train_batch_0{s+1}_graphnorm_var_{idx}.pickle', 'wb') as handle:
        #     pickle.dump(var_from_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"var_per_batch: {torch.linalg.norm(var_per_batch.flatten(),ord=torch.inf)=}")
        # print(f"var_from_pickle: {torch.linalg.norm(var.flatten(),ord=torch.inf)=}")
        std = (var + self.eps).sqrt().index_select(0, batch)

        # if idx == 4:
        #     s += 1
        #     with open("ocpmodels/datasets/embeddings/graphnorm_info.txt", 'w') as f:
        #         f.write('%d' % s)
        return self.weight * out / std + self.bias


    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'
"""
Code of the Scalable Frame Averaging (Rotation Invariant) GNN
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding, Linear
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout, BatchNorm1d
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import (
    MessagePassing,
    radius_graph,
    fps,
    radius,
    global_max_pool,
    MLP,
    PointNetConv,
    knn,
)
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv import PointNetConv
from torch_scatter import scatter

from ocpmodels.common.registry import registry
from ocpmodels.models.base_model import BaseModel
from ocpmodels.models.force_decoder import ForceDecoder
from ocpmodels.models.utils.activations import swish
from ocpmodels.modules.phys_embeddings import PhysEmbedding
from ocpmodels.common.utils import get_pbc_distances, conditional_grad
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, scatter_max
from torch_geometric.data.data import Data


class GaussianSmearing(nn.Module):
    r"""Smears a distance distribution by a Gaussian function."""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class PointNet2SAModule(torch.nn.Module):
    def __init__(self, sample_radio, radius, max_num_neighbors, mlp):
        super(PointNet2SAModule, self).__init__()
        self.sample_ratio = sample_radio
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.point_conv = PointNetConv(mlp)

    def forward(self, data):
        x, pos, batch = data

        # Sample
        idx = fps(pos, batch, ratio=self.sample_ratio)

        # Group(Build graph)
        row, col = radius(
            pos,
            pos[idx],
            self.radius,
            batch,
            batch[idx],
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index = torch.stack([col, row], dim=0)

        # Apply pointnet
        x1 = self.point_conv(x, (pos, pos[idx]), edge_index)
        pos1, batch1 = pos[idx], batch[idx]

        return x1, pos1, batch1


class PointNet2GlobalSAModule(torch.nn.Module):
    """
    One group with all input points, can be viewed as a simple PointNet module.
    It also return the only one output point(set as origin point).
    """

    def __init__(self, mlp):
        super(PointNet2GlobalSAModule, self).__init__()
        self.mlp = mlp

    def forward(self, data):
        x, pos, batch = data
        if x is not None:
            x = torch.cat([x, pos], dim=1)
        x1 = self.mlp(x)

        x1 = scatter_max(x1, batch, dim=0)[0]  # (batch_size, C1)

        batch_size = x1.shape[0]
        pos1 = x1.new_zeros((batch_size, 3))  # set the output point as origin
        batch1 = torch.arange(batch_size).to(batch.device, batch.dtype)

        return x1, pos1, batch1


class PointConvFP(MessagePassing):
    """
    Core layer of Feature propagtaion module.
    """

    def __init__(self, mlp=None):
        super(PointConvFP, self).__init__()
        self.mlp = mlp
        self.aggr = "add"
        self.flow = "source_to_target"

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, pos, edge_index):
        r"""
        Args:
            x (tuple), (tensor, tensor) or (tensor, NoneType)
            pos (tuple): The node position matrix. Either given as
                tensor for use in general message passing or as tuple for use
                in message passing in bipartite graphs.
            edge_index (LongTensor): The edge indices.
        """
        # Do not pass (tensor, None) directly into propagate(), sice it will check each item's size() inside.
        x_tmp = x[0] if x[1] is None else x
        aggr_out = self.propagate(edge_index, x=x_tmp, pos=pos)

        #
        i, j = (0, 1) if self.flow == "target_to_source" else (1, 0)
        x_target, pos_target = x[i], pos[i]

        add = (
            [
                pos_target,
            ]
            if x_target is None
            else [x_target, pos_target]
        )
        aggr_out = torch.cat([aggr_out, *add], dim=1)

        if self.mlp is not None:
            aggr_out = self.mlp(aggr_out)

        return aggr_out

    def message(self, x_j, pos_j, pos_i, edge_index):
        """
        x_j: (E, in_channels)
        pos_j: (E, 3)
        pos_i: (E, 3)
        """
        dist = (pos_j - pos_i).pow(2).sum(dim=1).pow(0.5)
        dist = torch.max(dist, torch.Tensor([1e-10]).to(dist.device, dist.dtype))
        weight = 1.0 / dist  # (E,)

        row, col = edge_index
        index = col
        num_nodes = maybe_num_nodes(index, None)
        wsum = (
            scatter_add(weight, col, dim=0, dim_size=num_nodes)[index] + 1e-16
        )  # (E,)
        weight /= wsum

        return weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class PointNet2FPModule(torch.nn.Module):
    def __init__(self, knn_num, mlp):
        super(PointNet2FPModule, self).__init__()
        self.knn_num = knn_num
        self.point_conv = PointConvFP(mlp)

    def forward(self, in_layer_data, skip_layer_data):
        in_x, in_pos, in_batch = in_layer_data
        skip_x, skip_pos, skip_batch = skip_layer_data

        row, col = knn(in_pos, skip_pos, self.knn_num, in_batch, skip_batch)
        edge_index = torch.stack([col, row], dim=0)

        x1 = self.point_conv((in_x, skip_x), (in_pos, skip_pos), edge_index)
        pos1, batch1 = skip_pos, skip_batch

        return x1, pos1, batch1


def make_mlp(in_channels, mlp_channels, batch_norm=True):
    assert len(mlp_channels) >= 1
    layers = []

    for c in mlp_channels:
        layers += [Lin(in_channels, c)]
        if batch_norm:
            layers += [BatchNorm1d(c)]
        layers += [ReLU()]

        in_channels = c

    return Seq(*layers)


class PointNet2PartSegmentNet(torch.nn.Module):
    """
    ref:
        - https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py
        - https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet++.py
    """

    def __init__(self):
        super(PointNet2PartSegmentNet, self).__init__()
        # SA1
        sa1_sample_ratio = 0.5
        sa1_radius = 0.2
        sa1_max_num_neighbours = 64
        sa1_mlp = make_mlp(3, [64, 64, 128])
        self.sa1_module = PointNet2SAModule(
            sa1_sample_ratio, sa1_radius, sa1_max_num_neighbours, sa1_mlp
        )

        # SA2
        sa2_sample_ratio = 0.25
        sa2_radius = 0.4
        sa2_max_num_neighbours = 64
        sa2_mlp = make_mlp(128 + 3, [128, 128, 256])
        self.sa2_module = PointNet2SAModule(
            sa2_sample_ratio, sa2_radius, sa2_max_num_neighbours, sa2_mlp
        )

        # SA3
        sa3_mlp = make_mlp(256 + 3, [256, 512, 1024])
        self.sa3_module = PointNet2GlobalSAModule(sa3_mlp)

        ##
        knn_num = 3

        # FP3, reverse of sa3
        fp3_knn_num = (
            1  # After global sa module, there is only one point in point cloud
        )
        fp3_mlp = make_mlp(1024 + 256 + 3, [256, 256])
        self.fp3_module = PointNet2FPModule(fp3_knn_num, fp3_mlp)

        # FP2, reverse of sa2
        fp2_knn_num = knn_num
        fp2_mlp = make_mlp(256 + 128 + 3, [256, 128])
        self.fp2_module = PointNet2FPModule(fp2_knn_num, fp2_mlp)

        # FP1, reverse of sa1
        fp1_knn_num = knn_num
        fp1_mlp = make_mlp(128 + 3, [128, 128, 128])
        self.fp1_module = PointNet2FPModule(fp1_knn_num, fp1_mlp)

    def forward(self, data):
        """
        data: a batch of input, torch.Tensor or torch_geometric.data.Data type
            - torch.Tensor: (batch_size, 3, num_points), as common batch input

            - torch_geometric.data.Data, as torch_geometric batch input:
                data.x: (batch_size * ~num_points, C), batch nodes/points feature,
                    ~num_points means each sample can have different number of points/nodes

                data.pos: (batch_size * ~num_points, 3)

                data.batch: (batch_size * ~num_points,), a column vector of graph/pointcloud
                    idendifiers for all nodes of all graphs/pointclouds in the batch. See
                    pytorch_gemometric documentation for more information
        """
        dense_input = True if isinstance(data, torch.Tensor) else False

        if dense_input:
            # Convert to torch_geometric.data.Data type
            data = data.transpose(1, 2).contiguous()
            batch_size, N, _ = data.shape  # (batch_size, num_points, 3)
            pos = data.view(batch_size * N, -1)
            batch = torch.zeros((batch_size, N), device=pos.device, dtype=torch.long)
            for i in range(batch_size):
                batch[i] = i
            batch = batch.view(-1)

            data = Data()
            data.pos, data.batch = pos, batch

        if not hasattr(data, "x"):
            data.x = None
        data_in = data.x, data.pos, data.batch

        sa1_out = self.sa1_module(data_in)
        sa2_out = self.sa2_module(sa1_out)
        sa3_out = self.sa3_module(sa2_out)

        fp3_out = self.fp3_module(sa3_out, sa2_out)
        fp2_out = self.fp2_module(fp3_out, sa1_out)
        fp1_out = self.fp1_module(fp2_out, data_in)

        fp1_out_x, fp1_out_pos, fp1_out_batch = fp1_out

        return fp1_out_x


class EmbeddingBlock(nn.Module):
    """Embedding block for the GNN
    Initialize nodes and edges' embeddings"""

    def __init__(
        self,
        num_gaussians,
        num_filters,
        hidden_channels,
        tag_hidden_channels,
        pg_hidden_channels,
        phys_hidden_channels,
        phys_embeds,
        act,
        second_layer_MLP,
    ):
        super().__init__()
        self.act = act
        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0
        self.use_mlp_phys = phys_hidden_channels > 0 and phys_embeds
        self.second_layer_MLP = second_layer_MLP

        # --- Node embedding ---

        # Phys embeddings
        self.phys_emb = PhysEmbedding(
            props=phys_embeds, props_grad=phys_hidden_channels > 0, pg=self.use_pg
        )
        # With MLP
        if self.use_mlp_phys:
            self.phys_lin = Linear(self.phys_emb.n_properties, phys_hidden_channels)
        else:
            phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = Embedding(
                self.phys_emb.period_size, pg_hidden_channels
            )
            self.group_embedding = Embedding(
                self.phys_emb.group_size, pg_hidden_channels
            )

        # Tag embedding
        if tag_hidden_channels:
            self.tag_embedding = Embedding(3, tag_hidden_channels)

        # Main embedding
        self.emb = Embedding(
            85,
            hidden_channels
            - tag_hidden_channels
            - phys_hidden_channels
            - 2 * pg_hidden_channels
            - 128,
        )

        # MLP
        self.lin = Linear(hidden_channels, hidden_channels)
        if self.second_layer_MLP:
            self.lin_2 = Linear(hidden_channels, hidden_channels)

        # --- Edge embedding ---
        self.lin_e1 = Linear(3, num_filters // 2)  # r_ij
        self.lin_e12 = Linear(num_gaussians, num_filters - (num_filters // 2))  # d_ij

        if self.second_layer_MLP:
            self.lin_e2 = Linear(num_filters, num_filters)
        self.pointnet = PointNet2PartSegmentNet()

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        if self.use_mlp_phys:
            nn.init.xavier_uniform_(self.phys_lin.weight)
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)
        if self.second_layer_MLP:
            nn.init.xavier_uniform_(self.lin_2.weight)
            self.lin_2.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_e2.weight)
            self.lin_e2.bias.data.fill_(0)

    def forward(self, data, z, rel_pos, edge_attr, tag=None, subnodes=None):
        # --- Edge embedding --
        rel_pos = self.lin_e1(rel_pos)  # r_ij
        edge_attr = self.lin_e12(edge_attr)  # d_ij
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = self.act(e)  # can comment out

        if self.second_layer_MLP:
            # e = self.lin_e2(e)
            e = self.act(self.lin_e2(e))

        # --- Node embedding --

        # Create atom embeddings based on its characteristic number
        h = self.emb(z)

        if self.phys_emb.device != h.device:
            self.phys_emb = self.phys_emb.to(h.device)

        # Concat tag embedding
        if self.use_tag:
            h_tag = self.tag_embedding(tag)
            h = torch.cat((h, h_tag), dim=1)

        # Concat physics embeddings
        if self.phys_emb.n_properties > 0:
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        # Concat period & group embedding
        if self.use_pg:
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        new_x = self.pointnet(data)
        h = torch.cat((h, new_x), dim=1)

        # MLP
        h = self.act(self.lin(h))
        if self.second_layer_MLP:
            h = self.act(self.lin_2(h))

        return h, e


class InteractionBlock(MessagePassing):
    """Interaction block for the GNN
    Updates node representations based on the message passing scheme"""

    def __init__(
        self,
        hidden_channels,
        num_filters,
        act,
        mp_type,
        complex_mp,
        graph_norm,
        dropout_lin,
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.mp_type = mp_type
        self.hidden_channels = hidden_channels
        self.complex_mp = complex_mp
        self.graph_norm = graph_norm
        self.dropout_lin = float(dropout_lin)
        if graph_norm:
            self.graph_norm = GraphNorm(
                hidden_channels if "updown" not in self.mp_type else num_filters
            )

        if self.mp_type == "simple":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "updownscale":
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updownscale_base":
            self.lin_geom = nn.Linear(num_filters + 2 * hidden_channels, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updown_local_env":
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_up = nn.Linear(2 * num_filters, hidden_channels)

        else:  # base
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels, hidden_channels
            )
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        if self.complex_mp:
            self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.mp_type != "simple":
            nn.init.xavier_uniform_(self.lin_geom.weight)
            self.lin_geom.bias.data.fill_(0)
        if self.complex_mp:
            nn.init.xavier_uniform_(self.other_mlp.weight)
            self.other_mlp.bias.data.fill_(0)
        if self.mp_type in {"updownscale", "updownscale_base", "updown_local_env"}:
            nn.init.xavier_uniform_(self.lin_up.weight)
            self.lin_up.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_down.weight)
            self.lin_down.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.lin_h.weight)
            self.lin_h.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        # Define edge embedding

        if self.dropout_lin > 0:
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )

        if self.mp_type in {"base", "updownscale_base"}:
            e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        if self.mp_type in {
            "updownscale",
            "base",
            "updownscale_base",
        }:
            e = self.act(self.lin_geom(e))

        # --- Message Passing block --

        if self.mp_type == "updownscale" or self.mp_type == "updownscale_base":
            h = self.act(self.lin_down(h))  # downscale node rep.
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.act(self.lin_up(h))  # upscale node rep.

        elif self.mp_type == "updown_local_env":
            h = self.act(self.lin_down(h))
            chi = self.propagate(edge_index, x=h, W=e, local_env=True)
            e = F.dropout(
                e, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            e = self.lin_geom(e)
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = torch.cat((h, chi), dim=1)
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.lin_up(h)

        elif self.mp_type in {"base", "simple"}:
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.act(self.lin_h(h))

        else:
            raise ValueError("mp_type provided does not exist")

        if self.complex_mp:
            h = F.dropout(
                h, p=self.dropout_lin, training=self.training or self.deup_inference
            )
            h = self.act(self.other_mlp(h))

        return h

    def message(self, x_j, W, local_env=None):
        if local_env is not None:
            return W
        else:
            return x_j * W


class OutputBlock(nn.Module):
    def __init__(self, energy_head, hidden_channels, act, dropout_lin):
        super().__init__()
        self.energy_head = energy_head
        self.act = act
        self.dropout_lin = float(dropout_lin)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)

        if self.energy_head == "weighted-av-final-embeds":
            self.w_lin = Linear(hidden_channels, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.energy_head == "weighted-av-final-embeds":
            nn.init.xavier_uniform_(self.w_lin.weight)
            self.w_lin.bias.data.fill_(0)
        if hasattr(self, "deup_lin"):
            nn.init.xavier_uniform_(self.deup_lin.weight)
            self.deup_lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, batch, alpha, data=None):
        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        # MLP
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training or self.deup_inference
        )
        h = self.lin1(h)
        h = self.act(h)
        h = F.dropout(
            h, p=self.dropout_lin, training=self.training or self.deup_inference
        )
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out


@registry.register_model("faenet")
class FAENet(BaseModel):
    r"""Frame Averaging GNN model FAENet.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        use_pbc (bool): Use of periodic boundary conditions.
            (default: `True`)
        act (str): Activation function
            (default: `swish`)
        max_num_neighbors (int): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: `40`)
        hidden_channels (int): Hidden embedding size.
            (default: `128`)
        tag_hidden_channels (int): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int): Hidden period and group embedding size.
            (default: :obj:`32`)
        phys_embeds (bool): Do we include fixed physics-aware embeddings.
            (default: :obj: `True`)
        phys_hidden_channels (int): Hidden size of learnable physics-aware embeddings.
            (default: :obj:`0`)
        num_interactions (int): The number of interaction (i.e. message passing) blocks.
            (default: :obj:`4`)
        num_gaussians (int): The number of gaussians :math:`\mu` to encode distance info.
            (default: :obj:`50`)
        num_filters (int): The size of convolutional filters.
            (default: :obj:`128`)
        second_layer_MLP (bool): Use 2-layers MLP at the end of the Embedding block.
            (default: :obj:`False`)
        skip_co (str): Add a skip connection between each interaction block and
            energy-head. (`False`, `"add"`, `"concat"`, `"concat_atom"`)
        mp_type (str): Specificies the Message Passing type of the interaction block.
            (`"base"`, `"updownscale_base"`, `"updownscale"`, `"updown_local_env"`, `"simple"`):
        graph_norm (bool): Whether to apply batch norm after every linear layer.
            (default: :obj:`True`)
        complex_mp (bool); Whether to add a second layer MLP at the end of each Interaction
            (default: :obj:`True`)
        energy_head (str): Method to compute energy prediction
            from atom representations.
            (`None`, `"weighted-av-initial-embeds"`, `"weighted-av-final-embeds"`)
        regress_forces (str): Specifies if we predict forces or not, and how
            do we predict them. (`None` or `""`, `"direct"`, `"direct_with_gradient_target"`)
        force_decoder_type (str): Specifies the type of force decoder
            (`"simple"`, `"mlp"`, `"res"`, `"res_updown"`)
        force_decoder_model_config (dict): contains information about the
            for decoder architecture (e.g. number of layers, hidden size).
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        act: str = "swish",
        use_pbc: bool = True,
        complex_mp: bool = False,
        max_num_neighbors: int = 40,
        num_gaussians: int = 50,
        num_filters: int = 128,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 32,
        pg_hidden_channels: int = 32,
        phys_hidden_channels: int = 0,
        phys_embeds: bool = True,
        num_interactions: int = 4,
        mp_type: str = "updownscale_base",
        graph_norm: bool = True,
        second_layer_MLP: bool = True,
        skip_co: str = "concat",
        energy_head: Optional[str] = None,
        regress_forces: Optional[str] = None,
        force_decoder_type: Optional[str] = "mlp",
        force_decoder_model_config: Optional[dict] = {"hidden_channels": 128},
        **kwargs,
    ):
        super().__init__()

        self.act = act
        self.complex_mp = complex_mp
        self.cutoff = cutoff
        self.energy_head = energy_head
        self.force_decoder_type = force_decoder_type
        self.force_decoder_model_config = force_decoder_model_config
        self.graph_norm = graph_norm
        self.hidden_channels = hidden_channels
        self.max_num_neighbors = max_num_neighbors
        self.mp_type = mp_type
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.phys_hidden_channels = phys_hidden_channels
        self.regress_forces = regress_forces
        self.second_layer_MLP = second_layer_MLP
        self.skip_co = skip_co
        self.tag_hidden_channels = tag_hidden_channels
        self.use_pbc = use_pbc

        self.dropout_edge = float(kwargs.get("dropout_edge") or 0)
        self.dropout_lin = float(kwargs.get("dropout_lin") or 0)
        self.dropout_lowest_layer = kwargs.get("dropout_lowest_layer", "output") or ""
        self.first_trainable_layer = kwargs.get("first_trainable_layer", "") or ""

        if not isinstance(self.regress_forces, str):
            assert self.regress_forces is False or self.regress_forces is None, (
                "regress_forces must be a string "
                + "('', 'direct', 'direct_with_gradient_target') or False or None"
            )
            self.regress_forces = ""

        if self.mp_type == "simple":
            self.num_filters = self.hidden_channels

        self.act = (
            (getattr(nn.functional, self.act) if self.act != "swish" else swish)
            if isinstance(self.act, str)
            else self.act
        )
        assert callable(self.act), (
            "act must be a callable function or a string "
            + "describing that function in torch.nn.functional"
        )

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_hidden_channels,
            self.phys_embeds,
            self.act,
            self.second_layer_MLP,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    self.act,
                    self.mp_type,
                    self.complex_mp,
                    self.graph_norm,
                    (
                        (
                            print(
                                f"🗑️ Setting dropout_lin for interaction block to {self.dropout_lin} ",
                                f"{i} / {self.num_interactions}",
                            )
                            or self.dropout_lin
                        )
                        if "inter" in self.dropout_lowest_layer
                        and (i >= int(self.dropout_lowest_layer.split("-")[-1]))
                        else 0
                    ),
                )
                for i in range(self.num_interactions)
            ]
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head,
            self.hidden_channels,
            self.act,
            (
                (
                    print(
                        f"🗑️ Setting dropout_lin for output block to {self.dropout_lin}"
                    )
                    or self.dropout_lin
                )
                if (
                    "inter" in self.dropout_lowest_layer
                    or "output" in self.dropout_lowest_layer
                )
                else 0
            ),
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = Linear(self.hidden_channels, 1)

        # Force head
        self.decoder = (
            ForceDecoder(
                self.force_decoder_type,
                self.hidden_channels,
                self.force_decoder_model_config,
                self.act,
            )
            if "direct" in self.regress_forces
            else None
        )

        # Skip co
        if self.skip_co == "concat":
            self.mlp_skip_co = Linear((self.num_interactions + 1), 1)
        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = Linear(
                ((self.num_interactions + 1) * self.hidden_channels),
                self.hidden_channels,
            )

        self.freeze(self.first_trainable_layer)
        print()

    def freeze(self, lowest_layer="dropout"):
        assert (
            not lowest_layer
            or "inter_" in lowest_layer
            or lowest_layer == "output"
            or lowest_layer == "dropout"
            or lowest_layer == "none"
        ), (
            "first_trainable_layer must be None, '', 'inter_{i}', "
            + f"'output', 'dropout', or 'none'. Received: {lowest_layer}"
        )
        if lowest_layer == "dropout":
            lowest_layer = self.dropout_lowest_layer

        if not lowest_layer:
            print("⛄️ No layer to freeze")
            return

        if lowest_layer == "embed":
            print("⛄️ No layer to freeze")
            return

        print("⛄️ Freezing embedding layer")
        self.freeze_layer(self.embed_block)
        if lowest_layer == "inter_0":
            return

        interaction_block_idx = (
            int(lowest_layer.replace("inter_", ""))
            if "inter_" in lowest_layer
            else len(self.interaction_blocks)
        )
        if interaction_block_idx < 0:
            interaction_block_idx = len(self.interaction_blocks) + interaction_block_idx
            self.first_trainable_layer = f"inter_{interaction_block_idx}"

        for ib in range(interaction_block_idx):
            if ib >= len(self.interaction_blocks):
                print("⛄️ Trying to freeze too many interaction blocks")
                break
            self.freeze_layer(self.interaction_blocks[ib])

        if ib > 0:
            print(
                "⛄️ Freezing interaction blocks 0 ->",
                f"{ib} / {len(self.interaction_blocks)}",
            )
        else:
            print(f"⛄️ Freezing interaction block 0 / {len(self.interaction_blocks)}")

        if self.skip_co == "concat_atom":
            self.freeze_layer(self.mlp_skip_co)
            print("⛄️ Freezing skip co atom layer")

        if "output" in lowest_layer:
            return

        print("⛄️ Freezing output block")
        self.freeze_layer(self.output_block)

    @conditional_grad(torch.enable_grad())
    def forces_forward(self, preds):
        """Predicts forces for 3D atomic systems.
        Can be utilised to predict any atom-level property.

        Args:
            preds (dict): dictionnary with predicted properties for each graph.

        Returns:
            dict: additional predicted properties, at an atom-level (e.g. forces)
        """
        if self.decoder:
            return self.decoder(preds["hidden_state"])

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data, q=None):
        """Predicts any graph-level properties (e.g. energy) for 3D atomic systems.

        Args:
            data (data.Batch): Batch of graphs datapoints.
        Returns:
            dict: predicted properties for each graph (e.g. energy)
        """
        # Rewire the graph
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        energy_skip_co = []

        # Use periodic boundary conditions
        if self.use_pbc and hasattr(data, "cell"):
            assert z.dim() == 1 and z.dtype == torch.long

            if self.dropout_edge > 0:
                edge_index, edge_mask = dropout_edge(
                    data.edge_index,
                    p=self.dropout_edge,
                    force_undirected=True,
                    training=self.training or self.deup_inference,
                )

            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            rel_pos = out["distance_vec"]
            edge_attr = self.distance_expansion(edge_weight)
        else:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )
            # edge_index = data.edge_index
            rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
            edge_weight = rel_pos.norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)
            if self.dropout_edge > 0:
                edge_index, edge_mask = dropout_edge(
                    edge_index,
                    p=self.dropout_edge,
                    force_undirected=True,
                    training=self.training or self.deup_inference,
                )
                edge_weight = edge_weight[edge_mask]
                edge_attr = edge_attr[edge_mask]
                rel_pos = rel_pos[edge_mask]

        if q is None:
            # Embedding block
            h, e = self.embed_block(data, z, rel_pos, edge_attr, data.tags)

            if "inter" and "0" in self.first_trainable_layer:
                q = h.clone().detach()

            # Compute atom weights for late energy head
            if self.energy_head == "weighted-av-initial-embeds":
                alpha = self.w_lin(h)
            else:
                alpha = None

            # Interaction blocks
            energy_skip_co = []
            for ib, interaction in enumerate(self.interaction_blocks):
                if self.skip_co == "concat_atom":
                    energy_skip_co.append(h)
                elif self.skip_co:
                    energy_skip_co.append(
                        self.output_block(
                            h, edge_index, edge_weight, batch, alpha, data
                        )
                    )
                if "inter" in self.first_trainable_layer and ib == int(
                    self.first_trainable_layer.split("_")[1]
                ):
                    q = h.clone().detach()
                h = h + interaction(h, edge_index, e)

            # Atom skip-co
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
                h = self.act(self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)))

            # Compute a graph density estimate for deup
            if "output" in self.first_trainable_layer:
                q = h.clone().detach()

        else:
            h = q
            alpha = None

        energy = self.output_block(h, edge_index, edge_weight, batch, alpha, data=data)

        # Skip-connection
        energy_skip_co.append(energy)
        if self.skip_co == "concat":
            energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))
        elif self.skip_co == "add":
            energy = sum(energy_skip_co)

        preds = {
            "energy": energy,
            "hidden_state": h,
            "q": q,
        }

        return preds

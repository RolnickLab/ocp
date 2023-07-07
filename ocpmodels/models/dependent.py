import torch
from torch.nn import Linear
from torch_scatter import scatter

from ocpmodels.models.faenet import FAENet
from ocpmodels.models.faenet import OutputBlock as conOutputBlock
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad

from torch_geometric.data import Batch

# Graph splitter should become a transform once model 3 is done.
def graph_splitter(graph):
    tags = graph.tags
    edge_index = graph.edge_index
    pos = graph.pos
    atomic_numbers = graph.atomic_numbers
    batch = graph.batch
    cell = graph.cell
    cell_offsets = graph.cell_offsets

    # Make masks to filter most data we need
    adsorbate_v_mask = (tags == 2)
    catalyst_v_mask = (tags == 1) + (tags == 0)

    adsorbate_e_mask = (tags[edge_index][0] == 2) * (tags[edge_index][1] == 2)
    catalyst_e_mask = (
        ((tags[edge_index][0] == 1) + (tags[edge_index][0] == 0)) 
        * ((tags[edge_index][1] == 1) + (tags[edge_index][1] == 0))
    )

    # Recalculate neighbors
    ads_neighbors = scatter(adsorbate_e_mask.long(), batch[edge_index[0]], dim = 0, reduce = "add")
    cat_neighbors = scatter(catalyst_e_mask.long(), batch[edge_index[0]], dim = 0, reduce = "add")

    # Reindex the edge indices.
    device = graph.edge_index.device
    natoms = graph.natoms.sum().item()

    ads_assoc = torch.full((natoms,), -1, dtype = torch.long, device = device)
    cat_assoc = torch.full((natoms,), -1, dtype = torch.long, device = device)

    ads_assoc[adsorbate_v_mask] = torch.arange(adsorbate_v_mask.sum(), device = device)
    cat_assoc[catalyst_v_mask] = torch.arange(catalyst_v_mask.sum(), device = device)

    ads_edge_index = ads_assoc[edge_index[:, adsorbate_e_mask]]
    cat_edge_index = cat_assoc[edge_index[:, catalyst_e_mask]]
    
    # Create the batches
    adsorbate = Batch(
        edge_index = ads_edge_index,
        pos = pos[adsorbate_v_mask, :],
        atomic_numbers = atomic_numbers[adsorbate_v_mask],
        batch = batch[adsorbate_v_mask],
        cell = cell,
        cell_offsets = cell_offsets[adsorbate_e_mask, :],
        tags = tags[adsorbate_v_mask],
        neighbors = ads_neighbors,
        mode="adsorbate"
    )
    catalyst = Batch(
        edge_index = cat_edge_index,
        pos = pos[catalyst_v_mask, :],
        atomic_numbers = atomic_numbers[catalyst_v_mask],
        batch = batch[catalyst_v_mask],
        cell = cell,
        cell_offsets = cell_offsets[catalyst_e_mask, :],
        tags = tags[catalyst_v_mask],
        neighbors = cat_neighbors,
        mode="catalyst"
    )

    return adsorbate, catalyst

class discOutputBlock(conOutputBlock):
    def __init__(self, energy_head, hidden_channels, act):
        super(discOutputBlock, self).__init__(
            energy_head, hidden_channels, act
        )

        del self.lin2
        self.lin2 = Linear(hidden_channels // 2, hidden_channels // 2)

        self.sys_lin1 = Linear(hidden_channels // 2 * 2, hidden_channels // 2)
        self.sys_lin2 = Linear(hidden_channels // 2, 1)

    def tags_saver(self, tags):
        self.current_tags = tags

    def forward(self, h, edge_index, edge_weight, batch, alpha):
        if self.energy_head == "weighted-av-final-embeds": # Right now, this is the only available option.
            alpha = self.w_lin(h)

        elif self.energy_head == "graclus":
            h, batch = self.graclus(h, edge_index, edge_weight, batch)

        elif self.energy_head in {"pooling", "random"}:
            h, batch, pooling_loss = self.hierarchical_pooling(
                h, edge_index, edge_weight, batch
            )

        # MLP
        h = self.lin1(h)
        h = self.lin2(self.act(h))

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        ads = self.current_tags == 2
        cat = ~ads

        ads_out = scatter(h, batch * ads, dim = 0, reduce = "add")
        cat_out = scatter(h, batch * cat, dim = 0, reduce = "add")
        system = torch.cat([ads_out, cat_out], dim = 1)
        
        system = self.sys_lin1(system)
        energy = self.sys_lin2(system)
        
        return energy

@registry.register_model("dependent")
class depFAENet(FAENet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.output_block
        self.output_block = discOutputBlock(
            self.energy_head, kwargs["hidden_channels"], self.act
        )

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        self.output_block.tags_saver(data.tags)
        pred = super().energy_forward(data)

        return pred

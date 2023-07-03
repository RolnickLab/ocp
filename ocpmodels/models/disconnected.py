import torch
from torch.nn import Linear
from torch_scatter import scatter

from ocpmodels.models.faenet import FAENet as conFAENet
from ocpmodels.models.faenet import OutputBlock as conOutputBlock
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad

from torch_geometric.data import Batch

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

        assert self.energy_head == "weighted-av-final-embeds"
        del self.lin2

        self.lin2 = Linear(hidden_channels // 2, hidden_channels // 2)

@registry.register_model("disconnected")
class discFAENet(conFAENet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.energy_head == "weighted-av-final-embeds"
        del self.output_block

        hidden_channels = kwargs["hidden_channels"]

        self.output_block = discOutputBlock(
            self.energy_head, hidden_channels, self.act
        )

        self.lin1 = Linear(hidden_channels // 2 * 2, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)

    @conditional_grad(torch.enable_grad())
    def energy_forward(self, data):
        adsorbate, catalyst = graph_splitter(data)
        
        ads_pred = super().energy_forward(adsorbate)
        cat_pred = super().energy_forward(catalyst)

        ads_energy = ads_pred["energy"]
        cat_energy = cat_pred["energy"]

        system_energy = torch.cat((ads_energy, cat_energy), dim = 1)
        system_energy = self.lin1(system_energy)
        system_energy = self.lin2(system_energy)

        return system_energy

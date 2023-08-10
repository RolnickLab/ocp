import torch

from ocpmodels.datasets.separate_dataset import SeparateLmdbDataset
from ocpmodels.common.registry import registry

from torch_geometric.data import HeteroData

@registry.register_dataset("heterogeneous")
class HeterogeneousDataset(SeparateLmdbDataset):
    def __getitem__(self, idx):
        adsorbate, catalyst = super().__getitem__(idx)

        reaction = HeteroData()
        for graph in [adsorbate, catalyst]:
            mode = graph.mode
            for key in graph.keys:
                if key == "edge_index":
                    continue
                reaction[mode][key] = graph[key]

            reaction[mode, "is_close", mode].edge_index = graph.edge_index

        sender = torch.repeat_interleave(torch.arange(adsorbate.natoms.item()), catalyst.natoms.item())
        receiver = torch.arange(0, catalyst.natoms.item()).repeat(adsorbate.natoms.item())
        reaction["adsorbate", "is_disc", "catalyst"].edge_index = torch.stack([sender, receiver])

        return reaction

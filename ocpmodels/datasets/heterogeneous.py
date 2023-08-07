import torch

from ocpmodels.datasets.lmdb_dataset import LmdbDataset
from ocpmodels.common.registry import registry

from torch_geometric.data import HeteroData

@registry.register_dataset("heterogeneous")
class HeterogeneousDataset(LmdbDataset):
    def __getitem__(self, idx):
        adsorbate, catalyst = super(HeterogeneousDataset, self).__getitem__(self, idx)

        reaction = HeteroData()
        for graph in [adsorbate, catalyst]:
            mode = graph.mode
            for key in graph.keys():
                if key == "edge_index":
                    continue
                reaction[mode][key] = graph[key]

            reaction[mode, "is_close", mode].edge_index = graph.edge_index

        sender = torch.arange(0, adsorbate.natoms, 1/catalyst.natoms)
        receiver = torch.arange(0.0, catalyst.natoms).repeat(adsorbate.natoms)

        reaction["adsorbate", "is_disc", "catalyst"].edge_index = torch.stack([sender. reciver])

        return reaction

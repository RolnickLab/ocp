import bisect
import logging
import pickle
import time
from pathlib import Path

import lmdb
import numpy as np
import torch

from ocpmodels.datasets.lmdb_dataset import LmdbDataset
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import pyg2_data_transform

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

@registry.register_dataset("separate")
class SeparateLmdbDataset(LmdbDataset):
    def __getitem__(self, idx):
        t0 = time.time_ns()
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        import ipdb
        ipdb.set_trace()

        t1 = time.time_ns()
        if self.transform is not None:
            data_object = self.transform(data_object)
        t2 = time.time_ns()

        load_time = (t1 - t0) * 1e-9  # time in s
        transform_time = (t2 - t1) * 1e-9  # time in s
        total_get_time = (t2 - t0) * 1e-9  # time in s

        data_object.load_time = load_time
        data_object.transform_time = transform_time
        data_object.total_get_time = total_get_time

        return data_object

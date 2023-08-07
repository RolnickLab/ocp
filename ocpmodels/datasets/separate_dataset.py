import bisect
import logging
import pickle
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data 

from ocpmodels.datasets.lmdb_dataset import LmdbDataset
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import pyg2_data_transform

def graph_splitter(graph):
    edge_index = graph.edge_index
    pos = graph.pos
    cell = graph.cell
    atomic_numbers = graph.atomic_numbers
    natoms = graph.natoms
    cell_offsets = graph.cell_offsets
    force = graph.force
    distances = graph.distances
    fixed = graph.fixed
    tags = graph.tags
    y_init = graph.y_init
    y_relaxed = graph.y_relaxed
    pos_relaxed = graph.pos_relaxed
    id = graph.id

    # Make masks to filter most data we need
    adsorbate_v_mask = (tags == 2)
    catalyst_v_mask = ~adsorbate_v_mask

    adsorbate_e_mask = (tags[edge_index][0] == 2) * (tags[edge_index][1] == 2)
    catalyst_e_mask = (tags[edge_index][0] != 2) * (tags[edge_index][1] != 2)

    # Reindex the edge indices.
    device = graph.edge_index.device

    ads_assoc = torch.full((natoms,), -1, dtype = torch.long, device = device)
    cat_assoc = torch.full((natoms,), -1, dtype = torch.long, device = device)

    ads_natoms = adsorbate_v_mask.sum()
    cat_natoms = catalyst_v_mask.sum()

    ads_assoc[adsorbate_v_mask] = torch.arange(ads_natoms, device = device)
    cat_assoc[catalyst_v_mask] = torch.arange(cat_natoms, device = device)

    ads_edge_index = ads_assoc[edge_index[:, adsorbate_e_mask]]
    cat_edge_index = cat_assoc[edge_index[:, catalyst_e_mask]]

    # Create the graphs
    adsorbate = Data(
        edge_index = ads_edge_index,
        pos = pos[adsorbate_v_mask, :],
        cell = cell,
        atomic_numbers = atomic_numbers[adsorbate_v_mask],
        natoms = ads_natoms,
        cell_offsets = cell_offsets[adsorbate_e_mask, :],
        force = force[adsorbate_v_mask, :],
        tags = tags[adsorbate_v_mask],
        y_init = y_init,
        y_relaxed = y_relaxed,
        pos_relaxed = pos_relaxed[adsorbate_v_mask, :],
        id = id,
        mode="adsorbate"
    )

    catalyst = Data(
        edge_index = cat_edge_index,
        pos = pos[catalyst_v_mask, :],
        cell = cell,
        atomic_numbers = atomic_numbers[catalyst_v_mask],
        natoms = cat_natoms,
        cell_offsets = cell_offsets[catalyst_e_mask, :],
        force = force[catalyst_v_mask, :],
        tags = tags[catalyst_v_mask],
        y_init = y_init,
        y_relaxed = y_relaxed,
        pos_relaxed = pos_relaxed[catalyst_v_mask, :],
        id = id,
        mode="catalyst"
    )

    return adsorbate, catalyst

@registry.register_dataset("separate")
class SeparateLmdbDataset(LmdbDataset): # Check that the dataset works as intended, with an specific example.
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

        adsorbate, catalyst = graph_splitter(data_object)

        t1 = time.time_ns()
        if self.transform is not None:
            adsorbate = self.transform(adsorbate)
            catalyst = self.transform(catalyst)
        t2 = time.time_ns()

        load_time = (t1 - t0) * 1e-9  # time in s
        transform_time = (t2 - t1) * 1e-9  # time in s
        total_get_time = (t2 - t0) * 1e-9  # time in s

        adsorbate.load_time = load_time
        adsorbate.transform_time = transform_time
        adsorbate.total_get_time = total_get_time

        catalyst.load_time = load_time
        catalyst.transform_time = transform_time
        catalyst.total_get_time = total_get_time

        return (adsorbate, catalyst)

"""lmdb_dataset.py
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import json
import logging
import pickle
import time
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import pyg2_data_transform


@registry.register_dataset("lmdb")
@registry.register_dataset("single_point_lmdb")
@registry.register_dataset("trajectory_lmdb")
class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
            fa_frames (str, optional): type of frame averaging method applied, if any.
            adsorbates (str, optional): comma-separated list of adsorbates to filter.
                    If None or "all", no filtering is applied.
                    (default: None)
            adsorbates_ref_dir: where metadata files for adsorbates are stored.
                    (default: "/network/scratch/s/schmidtv/ocp/datasets/ocp/per_ads")
    """

    def __init__(
        self,
        config,
        transform=None,
        fa_frames=None,
        lmdb_glob=None,
        adsorbates=None,
        adsorbates_ref_dir=None,
        silent=False,
    ):
        super().__init__()
        self.config = config
        self.adsorbates = adsorbates
        self.adsorbates_ref_dir = adsorbates_ref_dir
        self.silent = silent

        self.path = Path(self.config["src"])
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            if lmdb_glob:
                db_paths = [
                    p for p in db_paths if any(lg in p.stem for lg in lmdb_glob)
                ]
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = self.envs[-1].begin().get("length".encode("ascii"))
                if length is not None:
                    length = pickle.loads(length)
                else:
                    length = self.envs[-1].stat()["entries"]
                assert length is not None, f"Could not find length of LMDB {db_path}"
                self._keys.append([str(i).encode("ascii") for i in range(length)])

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii") for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        self.filter_per_adsorbates()
        self.transform = transform
        self.fa_method = fa_frames

    def filter_per_adsorbates(self):
        """Filter the dataset to only include structures with a specific
        adsorbate.
        """
        # no adsorbates specified, or asked for all: return
        if not self.adsorbates or self.adsorbates == "all":
            return

        # val_ood_ads and val_ood_both don't have targeted adsorbates
        if Path(self.config["src"]).parts[-1] in {"val_ood_ads", "val_ood_both"}:
            return

        # make set of adsorbates from a list or a string. If a string, split on comma.
        ads = []
        if isinstance(self.adsorbates, str):
            if "," in self.adsorbates:
                ads = [a.strip() for a in self.adsorbates.split(",")]
            else:
                ads = [self.adsorbates]
        else:
            ads = self.adsorbates
        ads = set(ads)

        # find reference file for this dataset
        ref_path = self.adsorbates_ref_dir
        if not ref_path:
            print("No adsorbate reference directory provided as `adsorbate_ref_dir`.")
            return
        ref_path = Path(ref_path)
        if not ref_path.is_dir():
            print(f"Adsorbate reference directory {ref_path} does not exist.")
            return
        pattern = f"{self.config['split']}-{self.path.parts[-1]}"
        candidates = list(ref_path.glob(f"*{pattern}*.json"))
        if not candidates:
            print(
                f"No adsorbate reference files found for {self.path.name}.:"
                + "\n".join(
                    [
                        str(p)
                        for p in [
                            ref_path,
                            pattern,
                            list(ref_path.glob(f"*{pattern}*.json")),
                            list(ref_path.glob("*")),
                        ]
                    ]
                )
            )
            return
        if len(candidates) > 1:
            print(
                f"Multiple adsorbate reference files found for {self.path.name}."
                "Using the first one."
            )
        ref = json.loads(candidates[0].read_text())

        # find dataset indices with the appropriate adsorbates
        allowed_idxs = set(
            str(i).encode("ascii")
            for i, a in zip(ref["ds_idx"], ref["ads_symbols"])
            if a in ads
        )

        previous_samples = self.num_samples

        # filter the dataset indices
        if isinstance(self._keys[0], bytes):
            self._keys = [i for i in self._keys if i in allowed_idxs]
            self.num_samples = len(self._keys)
        else:
            assert isinstance(self._keys[0], list)
            self._keys = [[i for i in k if i in allowed_idxs] for k in self._keys]
            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)

        if not self.silent:
            print(
                f"Filtered dataset {pattern} from {previous_samples} to",
                f"{self.num_samples} samples. (adsorbates: {ads})",
            )

        assert self.num_samples > 0, f"No samples found for adsorbates {ads}."

    def __len__(self):
        return self.num_samples

    def get_pickled_from_db(self, idx):
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            return (
                f"{db_idx}_{el_idx}",
                self.envs[db_idx].begin().get(self._keys[db_idx][el_idx]),
            )

        return None, self.env.begin().get(self._keys[idx])

    def __getitem__(self, idx):
        t0 = time.time_ns()
        # je peux noise les positions ici et rajouter des attributs
        # si on veut débugguer, il faut un seul worker, donc il faut utiliser
        # --no_cpus_to_workers et optim.num_workers =0.

        el_id, datapoint_pickled = self.get_pickled_from_db(idx)
        data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
        if el_id:
            data_object.id = el_id

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
        data_object.idx_in_dataset = idx
        # data_object.noised_version=
        # si je veux stocker n'importe quoi associé à ma data, le stocker ici
        # par ex une target, une magnitude de noise.
        # Quand on débuggue, mettre num_workers à 0 (pas de subprocess, tout est dans le main process). 
        # Sinon l'erreur sur les différents CPU peut ê dans un subworker, asynchrone par rapp au main process.
        # Raison de set les seeds dans les workers si on fait appel à des fonctions aléatoires dedans, de 
        # façon à ce que l'erreur ait toujours lieu au même endroit.

        return data_object

    def connect_db(self, lmdb_path=None):
        # https://lmdb.readthedocs.io/en/release/#environment-class
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        print("Closing", str(self.path))
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()

@registry.register_dataset("lmdb_noisy")
class NoisyLmdbDataset(Dataset):
    def noise_graph(self, graph, idx):
        nn_config = self.config.get("noisy_nodes")
        # Except for train set, nn_config is None, so return graph, i.e. no noising
        if not isinstance(nn_config, dict): return graph

        # TODO: mettre fonction noising de equiformer en adaptant code ci-dessous
        graph.original_pos = graph.pos.clone()

        assert (
            nn_config.get("type") in ["constant", "rand", "rand_deter"], 
            f"Unknown noisy node type in {nn_config}",
        )
        assert isinstance(nn_config.get("value"), float), f"Unknown noisy node value in {nn_config}"

        if nn_config["type"] == "constant":
            # graph.pos = graph.pos + torch.ones_like(graph.pos) * nn_config["value"]
            
        elif nn_config["type"] == "rand":
            graph.pos = graph.pos + torch.rand_like(graph.pos) * nn_config["value"]
        elif nn_config["type"] == "rand_deter":
            graph.pos = graph.pos + (1 / torch.log(idx + 2))

        return graph
    
    def __getitem__(self, idx):
        graph_data = super().__getitem__(idx)
        return self.noise_graph(graph_data, idx)
    
    def interpolate_init_relaxed_pos(self, batch):#Mettre dans le dataloader
        # rien de cette fonction n'a besoin d'autre info que le batch de data
        # We keep the name batch for the argument of the method, although it is very 
        
        # The method acts on a batch, i.e. an instance of torch_geometric.data.Batch
        # see explore.ipynb
        _interpolate_threshold = 0.5
        _min_interpolate_factor = 0.0 #0.1
        _gaussian_noise_std = 0.3
        
        batch_index = batch.batch
        batch_size = batch_index.max() + 1
        
        threshold_tensor = torch.rand((batch_size, 1), dtype=batch.pos.dtype, device=batch.pos.device)
        threshold_tensor = threshold_tensor + (1 - _interpolate_threshold)
        threshold_tensor = threshold_tensor.floor_() # 1: has interpolation, 0: no interpolation
        threshold_tensor = threshold_tensor[batch_index]
        
        interpolate_factor = torch.zeros((batch_index.shape[0], 1), 
            dtype=batch.pos.dtype, device=batch.pos.device)
        interpolate_factor = interpolate_factor.uniform_(_min_interpolate_factor, 1)
        
        noise_vec = torch.zeros((batch_index.shape[0], 3), 
            dtype=batch.pos.dtype, device=batch.pos.device)
        noise_vec = noise_vec.uniform_(-1, 1)
        noise_vec_norm = noise_vec.norm(dim=1, keepdim=True)
        noise_vec = noise_vec / (noise_vec_norm + 1e-6)
        noise_scale = torch.zeros((batch_index.shape[0], 1), 
            dtype=batch.pos.dtype, device=batch.pos.device)
        noise_scale = noise_scale.normal_(mean=0, std=_gaussian_noise_std)
        noise_vec = noise_vec * noise_scale
        
        noise_vec = noise_vec.normal_(mean=0, std=_gaussian_noise_std)
        
        #interpolate_factor = interpolate_factor * threshold_tensor
        #interpolate_factor = 1 - interpolate_factor
        #assert torch.all(interpolate_factor >= 0.0)
        #assert torch.all(interpolate_factor <= 1.0)
        #interpolate_factor = interpolate_factor[batch_index]
        #batch.pos = batch.pos * interpolate_factor + (1 - interpolate_factor) * batch.pos_relaxed
        
        tags = batch.tags
        tags = (tags > 0)
        pos = batch.pos
        pos_relaxed = batch.pos_relaxed
        pos_interpolated = pos * interpolate_factor + (1 - interpolate_factor) * pos_relaxed
        pos_noise = pos_interpolated + noise_vec
        new_pos = pos_noise * threshold_tensor + pos * (1 - threshold_tensor) 
        batch.pos[tags] = new_pos[tags]
        
        return batch


@registry.register_dataset("deup_lmdb")
class DeupDataset(LmdbDataset):
    def __init__(self, all_datasets_configs, deup_split, transform=None, silent=False):
        # ! WARNING: this does not (yet?) handle adsorbate filtering
        super().__init__(
            all_datasets_configs[deup_split],
            lmdb_glob=deup_split.replace("deup-", "").split("-"),
            silent=silent,
        )
        ocp_splits = deup_split.split("-")[1:]
        self.ocp_datasets = {
            d: LmdbDataset(all_datasets_configs[d], transform, silent=silent)
            for d in ocp_splits
        }

    def __getitem__(self, idx):
        _, datapoint_pickled = self.get_pickled_from_db(idx)
        deup_sample = pickle.loads(datapoint_pickled)
        ocp_sample = self.ocp_datasets[deup_sample["ds"]][deup_sample["idx_in_dataset"]]
        for k, v in deup_sample.items():
            setattr(ocp_sample, f"deup_{k}", v)
        return ocp_sample


@registry.register_dataset("stats_lmdb")
class StatsDataset(LmdbDataset):
    def to_reduced_formula(self, list_of_z):
        from collections import Counter

        from pymatgen.core.composition import Composition
        from pymatgen.core.periodic_table import Element

        return Composition.from_dict(
            Counter([Element.from_Z(i).symbol for i in list_of_z])
        ).reduced_formula

    def __getitem__(self, idx):
        data_object = super().__getitem__(idx)
        data_object.stats = {
            "atomic_numbers_bulk": data_object.atomic_numbers[data_object["tags"] < 2]
            .int()
            .tolist(),
            "atomic_numbers_ads": data_object.atomic_numbers[data_object["tags"] == 2]
            .int()
            .tolist(),
            "composition_bulk": self.to_reduced_formula(
                data_object.atomic_numbers[data_object["tags"] < 2].int()
            ),
            "composition_ads": self.to_reduced_formula(
                data_object.atomic_numbers[data_object["tags"] == 2].int()
            ),
            "idx_in_dataset": [data_object.idx_in_dataset],
            "sid": [data_object.sid],
            "y_relaxed": [data_object.y_relaxed],
            "y_init": [data_object.y_init],
        }
        return data_object


class SinglePointLmdbDataset(LmdbDataset):
    def __init__(self, config, transform=None):
        super(SinglePointLmdbDataset, self).__init__(config, transform)
        warnings.warn(
            "SinglePointLmdbDataset is deprecated and will be removed in the future."
            "Please use 'LmdbDataset' instead.",
            stacklevel=3,
        )


class TrajectoryLmdbDataset(LmdbDataset):
    def __init__(self, config, transform=None):
        super(TrajectoryLmdbDataset, self).__init__(config, transform)
        warnings.warn(
            "TrajectoryLmdbDataset is deprecated and will be removed in the future."
            "Please use 'LmdbDataset' instead.",
            stacklevel=3,
        )


def data_list_collater(data_list, otf_graph=False):
    batch = Batch.from_data_list(data_list)

    if (
        not otf_graph
        and hasattr(data_list[0], "edge_index")
        and data_list[0].edge_index is not None
    ):
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except NotImplementedError:
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )

    return batch

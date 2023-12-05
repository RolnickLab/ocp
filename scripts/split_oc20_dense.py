from pathlib import Path

import lmdb
import pickle
import shutil
import json
import joblib
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

DEFAULT_DENSE_DATASET_PATH = Path(f"/network/scratch/s/schmidtv/ocp/datasets/ocp/")
DEFAULT_DENSE_METADATA_PATH = Path(
    f"/network/scratch/s/schmidtv/ocp/datasets/ocp/dense/metadata/"
)
DEFAULT_BASE_METADATA_PATH = Path(
    f"/network/scratch/s/schmidtv/ocp/datasets/ocp/per_ads"
)
splits = ["id", "ood_ads", "ood_both", "ood_cat"]

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    targets = joblib.load(DEFAULT_DENSE_METADATA_PATH.joinpath("oc20dense_targets.pkl"))
    ref_energies = joblib.load(
        DEFAULT_DENSE_METADATA_PATH.joinpath("oc20dense_ref_energies.pkl")
    )
    mapping = joblib.load(DEFAULT_DENSE_METADATA_PATH.joinpath("oc20dense_mapping.pkl"))

    columns = [
        "system_id",
        "adsorbate",
        "ref_energy",
        "config_id",
        "target_energy",
        "mpid",
        "global_targeet",
    ]
    df = pd.DataFrame(columns=columns)
    rows = list()
    for sid, config_mapping in tqdm(mapping.items()):
        system_id = config_mapping["system_id"]
        ref_energy = ref_energies[system_id]

        target_energy = [
            energy
            for energy in targets[system_id]
            if energy[0] == config_mapping["config_id"]
        ]

        assert (
            len(target_energy) <= 1
        ), "More than one target energy found, this is strange"

        if len(target_energy) == 0:
            target_energy = None
        else:
            target_energy = target_energy[0][1]
        global_target_energy = np.min([energy[1] for energy in targets[system_id]])

        new_row = pd.DataFrame(
            {
                "system_id": system_id,
                "adsorbate": config_mapping["adsorbate"],
                "ref_energy": ref_energy,
                "config_id": config_mapping["config_id"],
                "target_energy": target_energy,
                "mpid": config_mapping["mpid"],
                "global_target": global_target_energy,
            },
            index=[0],
        )
        rows.append(new_row)
    df = pd.concat(rows, ignore_index=True)
    df_description = df.copy().dropna()

    oc20_metadata = {}
    for split in splits:
        oc20_metadata[split] = json.loads(
            open(
                DEFAULT_BASE_METADATA_PATH.joinpath(f"is2re-all-val_{split}.json"), "r"
            ).read()
        )

    df_description["ood_ads"] = df_description["adsorbate"].isin(
        oc20_metadata["ood_ads"]["ads_symbols"]
    )
    df_description["ood_cat"] = df_description["mpid"].isin(
        oc20_metadata["ood_cat"]["bulk_mpid"]
    )
    df_description["ood_both"] = df_description["adsorbate"].isin(
        oc20_metadata["ood_both"]["ads_symbols"]
    ) & df_description["mpid"].isin(oc20_metadata["ood_both"]["bulk_mpid"])
    df_description["id"] = df_description["adsorbate"].isin(
        oc20_metadata["id"]["ads_symbols"]
    ) & df_description["mpid"].isin(oc20_metadata["id"]["bulk_mpid"])

    for split in splits:
        Path(f"{DEFAULT_DENSE_DATASET_PATH}/dense/{split}/").mkdir(
            parents=True, exist_ok=True
        )
        shutil.rmtree(
            f"{DEFAULT_DENSE_DATASET_PATH}/dense/{split}/oc20dense_{split}.lmdb",
            ignore_errors=True,
        )

        lmdb_split_path = (
            f"{DEFAULT_DENSE_DATASET_PATH}/dense/{split}/oc20dense_{split}.lmdb"
        )

        print(f"Creating lmdb file for split: {split}")

        # Open the new file with the same config as the original env
        env_split = lmdb.open(
            str(lmdb_split_path),
            subdir=False,
            map_size=1e12,
            readonly=False,
        )

        env = lmdb.open(
            str(DEFAULT_DENSE_DATASET_PATH.joinpath("dense/oc20dense.lmdb")),
            subdir=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            readonly=True,
        )

        # Iterate over all examples using lmdb
        with env.begin(write=False) as txn:
            with env_split.begin(write=True) as txn_split:
                for i, (j, element) in tqdm(
                    enumerate(
                        zip(
                            df_description[df_description[split]].index,
                            df_description[df_description[split]].to_dict("records"),
                        )
                    )
                ):
                    value = txn.get(f"{j}".encode())
                    structure = pickle.loads(value)
                    structure["y_init"] = element["ref_energy"]
                    structure["y_relaxed"] = element["target_energy"]
                    structure["system_id"] = element["system_id"]
                    structure["global_target"] = element["global_target"]
                    value = pickle.dumps(structure)
                    txn_split.put(f"{i}".encode(), value)

        env_split.sync()
        env_split.close()

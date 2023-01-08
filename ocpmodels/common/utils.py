"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import ast
import collections
import copy
import glob
import importlib
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import time
from bisect import bisect
from copy import deepcopy
from functools import wraps
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_scatter import segment_coo, segment_csr

import ocpmodels
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry

OCP_TASKS = {"s2ef", "is2re", "is2es"}
ROOT = Path(__file__).resolve().parent.parent.parent
JOB_ID = os.environ.get("SLURM_JOB_ID")


def move_lmdb_data_to_slurm_tmpdir(trainer_config):
    if (
        not trainer_config.get("cp_data_to_tmpdir")
        or "-qm9-" in trainer_config["config"]
    ):
        return trainer_config

    tmp_dir = Path(f"/Tmp/slurm.{JOB_ID}.0")
    for s, split in trainer_config["dataset"].items():
        if not isinstance(split, dict):
            continue
        new_dir = tmp_dir / Path(split["src"]).name
        if new_dir.exists():
            print(
                f"Data already copied to {str(new_dir)} for split",
                f"{s} with source path {split['src']}",
            )
            trainer_config["dataset"][s]["src"] = str(new_dir)
            continue
        new_dir.mkdir()
        command = ["rsync", "-av", f'{split["src"]}/', str(new_dir)]
        print("Copying data: ", " ".join(command))
        subprocess.run(command)
        for f in new_dir.glob("*.lmdb-lock"):
            f.unlink()
        trainer_config["dataset"][s]["src"] = str(new_dir)
    print("Done moving data to", str(new_dir))
    return trainer_config


def override_narval_paths(trainer_config):
    is_narval = (
        "narval.calcul.quebec" in os.environ.get("HOSTNAME", "")
        or os.environ.get("HOME") == "/home/vsch"
        or trainer_config["narval"]
    )
    if not is_narval:
        return trainer_config
    path_overrides = yaml.safe_load(
        (ROOT / "configs" / "models" / "tasks" / "_narval.yaml").read_text()
    )
    task = trainer_config["task"]["name"]
    split = trainer_config["task"]["split"]
    assert task in path_overrides, f"Task {task} not found in Narval paths overrides"

    assert (
        split in path_overrides[task]
    ), f"Split {split} not found in Narval paths overrides for task {task}"

    print(
        "Is on Narval. Overriding",
        trainer_config["dataset"],
        "with",
        path_overrides[task][split],
    )
    trainer_config["dataset"] = merge_dicts(
        trainer_config["dataset"], path_overrides[task][split]
    )

    return trainer_config


def set_qm9_target_stats(trainer_config):
    """
    Set target stats for QM9 dataset if the trainer config specifies the
    qm9 task as `model-task-split`.

    For the qm9 task, for each dataset, if "normalize_labels" is set to True,
    then new keys are added to the dataset config: "target_mean" and "target_std"
    according to the dataset's "target" key which is an index in the list of QM9
    properties to predict.


    Stats can be recomputed with:
        from torch_geometric.datasets import QM9
        qm = QM9(root=path)
        qm.mean(7)
        qm.std(7)

    Args:
        trainer_config (dict): The trainer config.

    Returns:
        dict: The trainer config with stats for each dataset, if relevant.
    """
    target_means = [
        2.672952651977539,
        75.28118133544922,
        -6.536452770233154,
        0.32204368710517883,
        6.858491897583008,
        1189.4105224609375,
        4.056937217712402,
        -11178.966796875,
        -11178.7353515625,
        -11178.7099609375,
        -11179.875,
        31.620365142822266,
        -76.11600494384766,
        -76.58049011230469,
        -77.01825714111328,
        -70.83665466308594,
        9.966022491455078,
        1.4067283868789673,
        1.1273993253707886,
    ]

    target_stds = [
        1.5034793615341187,
        8.17383098602295,
        0.5977412462234497,
        1.274855375289917,
        1.2841686010360718,
        280.4781494140625,
        0.9017231464385986,
        1085.5787353515625,
        1085.57275390625,
        1085.57275390625,
        1085.5924072265625,
        4.067580699920654,
        10.323753356933594,
        10.415176391601562,
        10.489270210266113,
        9.498342514038086,
        1830.4630126953125,
        1.6008282899856567,
        1.107471227645874,
    ]
    if "-qm9-" not in trainer_config["config"]:
        return trainer_config

    for d, dataset in deepcopy(trainer_config["dataset"]).items():
        if d == "default_val":
            continue
        if not dataset.get("normalize_labels", False):
            continue
        assert "target" in dataset
        mean = target_means[dataset["target"]]
        std = target_stds[dataset["target"]]
        trainer_config["dataset"][d]["target_mean"] = mean
        trainer_config["dataset"][d]["target_std"] = std

    return trainer_config


def set_qm7x_target_stats(trainer_config):
    """
    Set target stats for QM7-X dataset if the trainer config specifies the
    qm7x task as `model-task-split`.

    For the qm7x task, for each dataset, if "normalize_labels" is set to True,
    then new keys are added to the dataset config: "target_mean" and "target_std"
    according to the dataset's "target" key which is an index in the list of QM9
    properties to predict.


    Stats can be recomputed with:
        python ocpmodels/datasets/qm7x.py

    Args:
        trainer_config (dict): The trainer config.

    Returns:
        dict: The trainer config with stats for each dataset, if relevant.
    """
    if "-qm7x-" not in trainer_config["config"]:
        return trainer_config

    target_stats = json.loads(
        (ROOT / "configs" / "models" / "qm7x-metadata" / "stats.json").read_text()
    )

    for d, dataset in deepcopy(trainer_config["dataset"]).items():
        if d == "default_val":
            continue
        if not dataset.get("normalize_labels", False):
            continue
        assert "target" in dataset, "target must be specified."
        mean = target_stats[dataset["target"]]["mean"]
        std = target_stats[dataset["target"]]["std"]
        std_divider = trainer_config["dataset"][d].get("std_divider", 1.0)
        trainer_config["dataset"][d]["target_mean"] = mean
        trainer_config["dataset"][d]["target_std"] = std / std_divider

        if trainer_config["model"].get("regress_forces"):
            assert "forces_target" in dataset, "forces_target must be specified."
            mean = target_stats[dataset["forces_target"]]["mean"]
            std = target_stats[dataset["forces_target"]]["std"]
            trainer_config["dataset"][d]["grad_target_mean"] = mean
            trainer_config["dataset"][d]["grad_target_std"] = std / std_divider

    return trainer_config


def auto_note(trainer_config):
    """
    Turns a trainer's config note dictionary into a string.
    Eg: note = {"model": "hidden_channels, num_gaussians", "optim": "lr, decay_steps"}
       -> note = "hc128 ng20 - lr0.001 ds10000"

    Does nothing if the note is not a dictionary.

    Args:
        trainer_config (dict): Trainer's full configuration

    Returns:
        dict: updated (or not) trainer config (identical but for the "note" key
            if it was a dict)
    """
    if not isinstance(trainer_config.get("note"), dict):
        return trainer_config

    note = ""
    for k, (key, subkeys) in enumerate(trainer_config["note"].items()):
        if k > 0:
            note += " - "
        for i, subkey in enumerate(subkeys.split(",")):
            subkey = subkey.strip()
            if i > 0:
                note += " "
            new_subkey = (
                "".join(s[0] for s in subkey.split("_")) if subkey != "name" else ""
            )
            dic = trainer_config[key] if key != "_root_" else trainer_config
            note += f"{new_subkey}{dic[subkey]}"
    trainer_config["note"] = note

    if not trainer_config.get("wandb_name"):
        trainer_config["wandb_name"] = JOB_ID + " - " + note

    return trainer_config


class Units:
    """
    Energy converter:
    https://www.unitsconverters.com/fr/Kcal/Mol-A-Ev/Particle/Utu-7727-6180
    """

    @staticmethod
    def ev_to_kcalmol(energy):
        return energy * 23.0621

    @staticmethod
    def kcalmol_to_ev(energy):
        return energy / 23.0621


def run_command(command):
    """
    Run a shell command and return the output.
    """
    return subprocess.check_output(command.split(" ")).decode("utf-8").strip()


def count_gpus():
    gpus = 0
    if JOB_ID:
        try:
            slurm_gpus = run_command(f"squeue --job {JOB_ID} -o %b").split("\n")[1]
            gpus = re.findall(r".*(\d+)", slurm_gpus) or 0
            gpus = int(gpus[0]) if gpus != 0 else gpus
        except subprocess.CalledProcessError:
            gpus = torch.cuda.device_count()
    else:
        gpus = torch.cuda.device_count()

    return gpus


def count_cpus():
    cpus = None
    if JOB_ID:
        try:
            slurm_cpus = run_command(f"squeue --job {JOB_ID} -o %c").split("\n")[1]
            cpus = int(slurm_cpus)
        except subprocess.CalledProcessError:
            cpus = os.cpu_count()
    else:
        cpus = os.cpu_count()

    return cpus


def pyg2_data_transform(data: Data):
    # if we're on the new pyg (2.0 or later), we need to convert the data to the
    # new format
    if torch_geometric.__version__ >= "2.0":
        source = data.__dict__
        if "_store" in source:
            source = source["_store"]
        return Data(**{k: v for k, v in source.items() if v is not None})

    return data


def save_checkpoint(
    state, checkpoint_dir="checkpoints/", checkpoint_file="checkpoint.pt"
):
    filename = os.path.join(checkpoint_dir, checkpoint_file)
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def warmup_lr_lambda(current_step, optim_config):
    """Returns a learning rate multiplier.
    Till `warmup_steps`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """

    # keep this block for older configs that have warmup_epochs instead of warmup_steps
    # and lr_milestones are defined in epochs
    lr_milestones = optim_config.get("lr_milestones")
    if lr_milestones is None:
        assert optim_config.get("lr_gamma_freq") is not None
        lr_milestones = (
            np.arange(1, optim_config["max_epochs"]) * optim_config["lr_gamma_freq"]
        )

    if any(x < 100 for x in lr_milestones) or "warmup_epochs" in optim_config:
        raise Exception(
            "ConfigError: please define lr_milestones in steps not"
            + " epochs and define warmup_steps instead of warmup_epochs"
        )

    # warmup
    if current_step <= optim_config["warmup_steps"]:
        alpha = current_step / float(optim_config["warmup_steps"])
        return optim_config["warmup_factor"] * (1.0 - alpha) + alpha

    # post warm up
    if "decay_steps" in optim_config:
        # exponential decay per step
        assert "decay_rate" in optim_config, "decay_rate must be defined in optim"
        ds = optim_config["decay_steps"]
        if ds == "max_steps":
            assert "max_steps" in optim_config, "max_steps must be defined in optim"
            ds = optim_config["max_steps"]

        return optim_config["decay_rate"] ** (
            (current_step - optim_config["warmup_steps"]) / ds
        )
    # per-milestones decay
    idx = bisect(lr_milestones, current_step)
    return pow(optim_config["lr_gamma"], idx)


def print_cuda_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print(
        "Max Memory Allocated:",
        torch.cuda.max_memory_allocated() / (1024 * 1024),
    )
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))


def conditional_grad(dec):
    """
    Decorator to enable/disable grad depending on whether force/energy
    predictions are being made
    """
    # Adapted from
    # https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces in {"from_energy", "direct_with_gradient_target"}:
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


def plot_histogram(data, xlabel="", ylabel="", title=""):
    assert isinstance(data, list)

    # Preset
    fig = Figure(figsize=(5, 4), dpi=150)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    # Plot
    ax.hist(data, bins=20, rwidth=0.9, zorder=3)

    # Axes
    ax.grid(color="0.95", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout(pad=2)

    # Return numpy array
    canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )

    return image_from_plot


# Override the collation method in `pytorch_geometric.data.InMemoryDataset`
def collate(data_list):
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
        elif isinstance(item[key], int) or isinstance(item[key], float):
            s = slices[key][-1] + 1
        else:
            raise ValueError("Unsupported attribute type")
        slices[key].append(s)

    if hasattr(data_list[0], "__num_nodes__"):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        if torch.is_tensor(data_list[0][key]):
            data[key] = torch.cat(
                data[key], dim=data.__cat_dim__(key, data_list[0][key])
            )
        else:
            data[key] = torch.tensor(data[key])
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices


def add_edge_distance_to_graph(
    batch,
    device="cpu",
    dmin=0.0,
    dmax=6.0,
    num_gaussians=50,
):
    # Make sure x has positions.
    if not all(batch.pos[0][:] == batch.x[0][-3:]):
        batch.x = torch.cat([batch.x, batch.pos.float()], dim=1)
    # First set computations to be tracked for positions.
    batch.x = batch.x.requires_grad_(True)
    # Then compute Euclidean distance between edge endpoints.
    pdist = torch.nn.PairwiseDistance(p=2.0)
    distances = pdist(
        batch.x[batch.edge_index[0]][:, -3:],
        batch.x[batch.edge_index[1]][:, -3:],
    )
    # Expand it using a gaussian basis filter.
    gdf_filter = torch.linspace(dmin, dmax, num_gaussians)
    var = gdf_filter[1] - gdf_filter[0]
    gdf_filter, var = gdf_filter.to(device), var.to(device)
    gdf_distances = torch.exp(-((distances.view(-1, 1) - gdf_filter) ** 2) / var**2)
    # Reassign edge attributes.
    batch.edge_weight = distances
    batch.edge_attr = gdf_distances.float()
    return batch


# Copied from https://github.com/facebookresearch/mmf/blob/master/mmf/utils/env.py#L89.
def setup_imports():
    from ocpmodels.common.registry import registry

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("ocpmodels_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "..")

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(root_folder, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "*.py")
    task_folder = os.path.join(root_folder, "tasks")
    task_pattern = os.path.join(task_folder, "*.py")

    importlib.import_module("ocpmodels.common.logger")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
        + glob.glob(task_pattern, recursive=True)
    )

    for f in files:
        for key in ["/trainers", "/datasets", "/models", "/tasks"]:
            if f.find(key) != -1:
                splits = f.split(os.sep)
                file_name = splits[-1]
                module_name = file_name[: file_name.find(".py")]
                importlib.import_module("ocpmodels.%s.%s" % (key[1:], module_name))

    experimental_folder = os.path.join(root_folder, "../experimental/")
    if os.path.exists(experimental_folder):
        experimental_files = glob.glob(
            experimental_folder + "**/*py",
            recursive=True,
        )
        # Ignore certain directories within experimental
        ignore_file = os.path.join(experimental_folder, ".ignore")
        if os.path.exists(ignore_file):
            ignored = []
            with open(ignore_file) as f:
                for line in f.read().splitlines():
                    ignored += glob.glob(
                        experimental_folder + line + "/**/*py", recursive=True
                    )
            for f in ignored:
                experimental_files.remove(f)
        for f in experimental_files:
            splits = f.split(os.sep)
            file_name = ".".join(splits[-splits[::-1].index("..") :])
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module(module_name)

    registry.register("imports_setup", True)


def dict_set_recursively(dictionary, key_sequence, val):
    top_key = key_sequence.pop(0)
    if len(key_sequence) == 0:
        dictionary[top_key] = val
    else:
        if top_key not in dictionary:
            dictionary[top_key] = {}
        dict_set_recursively(dictionary[top_key], key_sequence, val)


def parse_value(value):
    """
    Parse string as Python literal if possible and fallback to string.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Use as string if nothing else worked
        return value


def create_dict_from_args(args: list, sep: str = "."):
    """
    Create a (nested) dictionary from console arguments.
    Keys in different dictionary levels are separated by sep.
    """
    return_dict = {}
    for arg in args:
        arg = arg.strip("--")
        keys_concat, val = arg.split("=")
        val = parse_value(val)
        key_sequence = keys_concat.split(sep)
        dict_set_recursively(return_dict, key_sequence, val)
    return return_dict


def load_config_legacy(path: str, previous_includes: list = []):
    path = Path(path)
    if path in previous_includes:
        raise ValueError(
            "Cyclic config include detected. "
            + f"{path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]
    direct_config = yaml.safe_load(open(path, "r"))

    # Load config from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )

    config = {}
    duplicates_warning = []
    duplicates_error = []

    for include in includes:
        include_config, inc_dup_warning, inc_dup_error = load_config_legacy(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def load_config(config_str):
    model, task, split = config_str.split("-")
    conf_path = ROOT / "configs" / "models"

    model_conf_path = list(conf_path.glob(f"{model}.y*ml"))[0]
    task_conf_path = list(conf_path.glob(f"tasks/{task}.y*ml"))[0]

    model_conf = yaml.safe_load(model_conf_path.read_text())
    task_conf = yaml.safe_load(task_conf_path.read_text())

    assert "default" in model_conf
    assert task in model_conf
    assert split in model_conf[task]

    assert "default" in task_conf
    assert split in task_conf

    config = merge_dicts({}, model_conf["default"])
    config = merge_dicts(config, model_conf[task].get("default", {}))
    config = merge_dicts(config, model_conf[task][split])
    config = merge_dicts(config, task_conf["default"])
    config = merge_dicts(config, task_conf[split])
    config["task"]["name"] = task
    config["task"]["split"] = split

    return config


def build_config(args, args_override):

    if args.config_yml:
        raise ValueError(
            "Using LEGACY config format. Please update your config to the new format."
        )

    config = load_config(args.config)

    # Check for overridden parameters.
    if args_override != []:
        overrides = create_dict_from_args(args_override)
        config = merge_dicts(config, overrides)

    config = merge_dicts(config, {k: v for k, v in vars(args).items() if v is not None})
    config["data_split"] = args.config.split("-")[-1]
    config["run_dir"] = resolve(config["run_dir"])
    config["slurm"] = {}
    config["job_id"] = JOB_ID or "no-job-id"

    if "regress_forces" in config["model"]:
        if not isinstance(config["model"]["regress_forces"], str):
            if config["model"]["regress_forces"] is False:
                config["model"]["regress_forces"] = ""
            else:
                raise ValueError(
                    "regress_forces must be False or a string: "
                    + "'from_energy' or 'direct' or 'direct_with_gradient_target'"
                    + f". Received: `{str(config['model']['regress_forces'])}`"
                )
        elif config["model"]["regress_forces"] not in {
            "from_energy",
            "direct",
            "direct_with_gradient_target",
        }:
            raise ValueError(
                "regress_forces must be False or a string: "
                + "'from_energy' or 'direct' or 'direct_with_gradient_target'"
                + f". Received: `{str(config['model']['regress_forces'])}`"
            )

    config = set_qm9_target_stats(config)
    config = set_qm7x_target_stats(config)
    config = override_narval_paths(config)
    config = auto_note(config)
    config = move_lmdb_data_to_slurm_tmpdir(config)

    if not config["no_cpus_to_workers"]:
        cpus = count_cpus()
        gpus = count_gpus()
        if cpus is not None:
            if gpus == 0:
                workers = cpus - 1
            else:
                workers = cpus // gpus
            if not config["silent"]:
                print(
                    f"Overriding num_workers from {config['optim']['num_workers']}",
                    f"to {workers} to match the machine's CPUs.",
                    "Use --no_cpus_to_workers=true to disable this behavior.",
                )
            config["optim"]["num_workers"] = workers
    config["world_size"] = args.num_nodes * args.num_gpus

    return config


def create_grid(base_config, sweep_file):
    def _flatten_sweeps(sweeps, root_key="", sep="."):
        flat_sweeps = []
        for key, value in sweeps.items():
            new_key = root_key + sep + key if root_key else key
            if isinstance(value, collections.MutableMapping):
                flat_sweeps.extend(_flatten_sweeps(value, new_key).items())
            else:
                flat_sweeps.append((new_key, value))
        return collections.OrderedDict(flat_sweeps)

    def _update_config(config, keys, override_vals, sep="."):
        for key, value in zip(keys, override_vals):
            key_path = key.split(sep)
            child_config = config
            for name in key_path[:-1]:
                child_config = child_config[name]
            child_config[key_path[-1]] = value
        return config

    sweeps = yaml.safe_load(open(sweep_file, "r"))
    flat_sweeps = _flatten_sweeps(sweeps)
    keys = list(flat_sweeps.keys())
    values = list(itertools.product(*flat_sweeps.values()))

    configs = []
    for i, override_vals in enumerate(values):
        config = copy.deepcopy(base_config)
        config = _update_config(config, keys, override_vals)
        # WARNING identifier has been deprecated in favour of wandb_name
        config["identifier"] = config["identifier"] + f"_run{i}"
        configs.append(config)
    return configs


def save_experiment_log(args, jobs, configs):
    log_file = args.logdir / "exp" / time.strftime("%Y-%m-%d-%I-%M-%S%p.log")
    log_file.parent.mkdir(exist_ok=True, parents=True)
    with open(log_file, "w") as f:
        for job, config in zip(jobs, configs):
            print(
                json.dumps(
                    {
                        "config": config,
                        "slurm_id": job.job_id,
                        "timestamp": time.strftime("%I:%M:%S%p %Z %b %d, %Y"),
                    }
                ),
                file=f,
            )
    return log_file


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets=False,
    return_distance_vec=False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


def radius_graph_pbc(data, radius, max_num_neighbors_threshold):
    device = data.pos.device
    batch_size = len(data.natoms)

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list
    # of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )
    # Compute a tensor containing sequences of numbers that range from 0 to
    # num_atoms_per_image_sqr for each image that is used to compute indices for
    # the pairs of atoms. This is a very convoluted way to implement the following
    # (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([
    #        batch_count,
    #        torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)
    #    ], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this approach could run into numerical
    # precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="trunc")
        + index_offset_expand
    )
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    rep_a1 = torch.ceil(radius * inv_min_dist_a1)

    cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    rep_a2 = torch.ceil(radius * inv_min_dist_a2)

    if radius >= 20:
        # Cutoff larger than the vacuum layer of 20A
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)
    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep
    ]
    unit_cell = torch.cat(torch.meshgrid(cells_per_dim), dim=-1).reshape(-1, 3)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most
        # max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(natoms, index, atom_distance, max_num_neighbors_threshold):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances
    # of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def get_pruned_edge_idx(edge_index, num_atoms=None, max_neigh=1e9):
    assert num_atoms is not None

    # removes neighbors > max_neigh
    # assumes neighbors are sorted in increasing distance
    _nonmax_idx = []
    for i in range(num_atoms):
        idx_i = torch.arange(len(edge_index[1]))[(edge_index[1] == i)][:max_neigh]
        _nonmax_idx.append(idx_i)
    _nonmax_idx = torch.cat(_nonmax_idx)

    return _nonmax_idx


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary
    as a value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py

    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share
        the same key.

    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k] = merge_dicts(dict1[k], dict2[k])
            elif isinstance(v, list) and isinstance(dict1[k], list):
                if len(dict1[k]) != len(dict2[k]):
                    raise ValueError(
                        f"List for key {k} has different length in dict1 and dict2."
                        + " Use an empty dict {} to pad for items in the shorter list."
                    )
                return_dict[k] = [merge_dicts(d1, d2) for d1, d2 in zip(dict1[k], v)]
            else:
                return_dict[k] = dict2[k]

    return return_dict


class SeverityLevelBetween(logging.Filter):
    def __init__(self, min_level, max_level):
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record):
        return self.min_level <= record.levelno < self.max_level


def setup_logging():
    root = logging.getLogger()

    # Perform setup only if logging has not been configured
    if not root.hasHandlers():
        root.setLevel(logging.INFO)

        log_formatter = logging.Formatter(
            "%(asctime)s (%(levelname)s): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Send INFO to stdout
        handler_out = logging.StreamHandler(sys.stdout)
        handler_out.addFilter(SeverityLevelBetween(logging.INFO, logging.WARNING))
        handler_out.setFormatter(log_formatter)
        root.addHandler(handler_out)

        # Send WARNING (and higher) to stderr
        handler_err = logging.StreamHandler(sys.stderr)
        handler_err.setLevel(logging.WARNING)
        handler_err.setFormatter(log_formatter)
        root.addHandler(handler_err)


def compute_neighbors(data, edge_index):
    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    num_neighbors = segment_coo(ones, edge_index[1], dim_size=data.natoms.sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    return neighbors


def check_traj_files(batch, traj_dir):
    if traj_dir is None:
        return False
    traj_dir = Path(traj_dir)
    traj_files = [traj_dir / f"{id}.traj" for id in batch[0].sid.tolist()]
    return all(fl.exists() for fl in traj_files)


def resolve(path):
    """
    Resolves a path: expand user (~) and env vars ($SCRATCH) and resolves to
    an absolute path.

    Args:
        path (Union[str, pathlib.Path]): the path to resolve

    Returns:
        pathlib.Path: the resolved Path
    """
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def update_from_sbatch_py_vars(args):
    sbatch_py_vars = {
        k.replace("SBATCH_PY_", "").lower(): v if v != "true" else True
        for k, v in os.environ.items()
        if k.startswith("SBATCH_PY_")
    }
    for k, v in sbatch_py_vars.items():
        setattr(args, k, v)
    return args


def make_script_trainer(str_args=[], overrides={}, silent=False, mode="train"):
    argv = [a for a in sys.argv]
    assert isinstance(str_args, list)

    if silent and all("--silent" not in s for s in str_args):
        str_args.append("--silent")
    if all("--mode" not in s for s in str_args):
        str_args.append(f"--mode={mode}")

    sys.argv[1:] = str_args
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    trainer_config = build_config(args, override_args)

    for k, v in overrides.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                trainer_config[k][kk] = vv
        else:
            trainer_config[k] = v

    trainer_config["silent"] = silent

    setup_imports()
    trainer = registry.get_trainer_class(trainer_config["trainer"])(**trainer_config)

    task = registry.get_task_class(trainer_config["mode"])(trainer_config)
    task.setup(trainer)

    sys.argv = argv

    return trainer


def get_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    ocpmodels.__path__[0],
                    "describe",
                    "--always",
                ]
            )
            .strip()
            .decode("ascii")
        )
    # catch instances where code is not being run from a git repo
    except Exception:
        commit_hash = None
    return commit_hash


def base_config(config, overrides={}):
    from argparse import Namespace

    n = Namespace()
    n.num_gpus = 1
    n.num_nodes = 1
    n.config_yml = None
    n.config = config

    conf = build_config(
        n,
        [
            "run_dir=.",
            "narval=",
            "no_qm7x_cp=true",
            "no_cpus_to_workers=true",
            "silent=",
        ],
    )

    return merge_dicts(conf, overrides)

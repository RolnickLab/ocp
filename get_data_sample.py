"""
Exploration file to get a `batch` in-memory and play around with it.
Use it in notebooks or ipython console

$ ipython
...
In [1]: run get_data_sample.py
Out[1]: ...

In [2]: print(batch)

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: F401
from tqdm import tqdm

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports, setup_logging

if __name__ == "__main__":

    sys.argv.append("--mode=train")
    sys.argv.append("--config=configs/is2re/10k/schnet/schnet.yml")
    setup_logging()

    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    config["optim"]["num_workers"] = 4
    config["optim"]["batch_size"] = 2
    config["logger"] = "dummy"

    setup_imports()
    trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
        task=config["task"],
        model_attributes=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        timestamp_id=config.get("timestamp_id", None),
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", "tensorboard"),
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
        slurm=config.get("slurm", {}),
    )

    task = registry.get_task_class(config["mode"])(config)
    task.setup(trainer)

    DELETE_TAG_0 = True
    if DELETE_TAG_0:
        for batch in trainer.train_loader:
            b = batch[0]
            batch_size = len(b.natoms)
            non_sub = torch.where(b.tags != 0)[0]
            src_is_not_sub = torch.isin(b.edge_index[0], non_sub)
            target_is_not_sub = torch.isin(b.edge_index[1], non_sub)
            neither_is_sub = src_is_not_sub * target_is_not_sub
            new_ei = b.edge_index[:, neither_is_sub]
            new_pos = b.pos[non_sub, :]
            new_an = b.atomic_numbers[non_sub]
            new_batch = b.batch[non_sub]
            new_natoms = torch.tensor(
                [(new_batch == i).sum() for i in range(batch_size)],
                dtype=b.natoms.dtype,
                device=b.natoms.device,
            )
            new_cell_offsets = b.cell_offsets[neither_is_sub, :]
            new_force = b.force[non_sub, :]
            new_fixed = b.fixed[non_sub]
            new_tags = b.tags[non_sub]
            new_distances = b.distances[neither_is_sub]
            new_pos_relaxed = b.pos_relaxed[non_sub, :]
            new_ptr = torch.tensor(
                [0] + [b.natoms[:i].sum() for i in range(1, batch_size + 1)],
                dtype=b.ptr.dtype,
                device=b.ptr.device,
            )
            TODO = "new_neighbors and adjust atom ids in edge_index"
            break

    PLOT_TAGS = False
    if PLOT_TAGS:
        tags = {
            0: [],
            1: [],
            2: [],
        }
        for batch in tqdm(trainer.train_loader):
            for b in batch:
                for t in tags:
                    tags[t].append((b.tags == t).sum().item())

        x = np.arange(len(tags[0]))
        ys = [np.array(tags[t]) for t in range(3)]
        z = np.zeros(len(x))
        fig = plt.figure(num=1)
        ax = fig.add_subplot(111)
        colors = {
            0: "b",
            1: "y",
            2: "g",
        }
        for t in tags:
            ax.plot(x, ys[t], color=colors[t], lw=1, label=f"tag {t}")
        for t in tags:
            ax.fill_between(
                x, ys[t], where=ys[t] > z, color=colors[t], interpolate=True
            )
        plt.legend()
        plt.savefig("tags_dist.png", dpi=150)

"""
Exploration file to get a `batch` in-memory and play around with it.
Use it in notebooks or ipython console

$ ipython
...
In [1]: run get_data_sample.py
Out[1]: ...

In [2]: print(batch)

"""
import torch # noqa: F401
import sys

from ocpmodels.common.utils import (
    build_config,
    setup_imports,
    setup_logging,
)
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry

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
        model=config["model"],
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

    for batch in trainer.train_loader:
        break

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from pathlib import Path

import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import set_deup_samples_path
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.common.utils import build_config
from ocpmodels.common.flags import flags
from ocpmodels.datasets.deup_dataset_creator import DeupDatasetCreator


class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        if self.config.get("checkpoint") is not None:
            print("\n🔵 Resuming:\n  • ", end="", flush=True)
            self.trainer.load_checkpoint(self.config["checkpoint"])
            print()

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. "
                        + "Consider removing it from the model."
                    )

    @torch.no_grad()
    def create_deup_dataset(self):
        dds = self.config["deup_dataset"]
        ddc = DeupDatasetCreator(
            trainers_conf={
                "checkpoints": (
                    Path(self.config["checkpoint_dir"]) / "best_checkpoint.pt"
                ),
                "dropout": self.config["model"].get("dropout_lin") or 0.7,
            },
            overrides={"logger": "dummy"},
        )

        output_path = ddc.create_deup_dataset(
            output_path=(
                dds.get("output_path") or Path(self.config["run_dir"]) / "deup_dataset"
            ),
            dataset_strs=dds["dataset_strs"],
            n_samples=dds["n_samples"],
            max_samples=-1,
            batch_size=128,
        )
        print("\n🤠 DEUP Dataset created in:", str(output_path))
        return output_path

    def run(self):
        self.config = self.trainer.config
        try:
            if self.config.get("deup_dataset", {}).get("create") == "before":
                output_path = self.create_deup_dataset()
                # self.trainer must be an EnsembleTrainer at this point
                self.trainer.config["deup_samples_path"] = str(output_path)
                self.trainer.config = set_deup_samples_path(self.trainer.config)
                self.trainer.load()

            loops = self.config.get("inference_time_loops", 5)
            if loops > 0:
                print("----------------------------------------")
                print("⏱️  Measuring inference time.")
                self.trainer.measure_inference_time(loops=loops)
                print("----------------------------------------\n")
            torch.set_grad_enabled(True)
            training_signal = self.trainer.train(
                disable_eval_tqdm=self.config.get("show_eval_progressbar", True),
                debug_batches=self.config.get("debug_batches", -1),
            )
            if training_signal == "SIGTERM":
                return

            if self.config.get("deup_dataset", {}).get("create") == "after":
                self.create_deup_dataset()

        except RuntimeError as e:
            self._process_error(e)
            raise e


@registry.register_task("run-relaxations")
class RelaxationTask(BaseTask):
    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. "
                        + "Consider removing it from the model."
                    )

    def run(self):
        self.config = self.trainer.config
        try:
            if "relax_dataset" in self.config["task"]:
                results = self.trainer.run_relaxations()
                print(results)
            else:
                raise ValueError(
                    "Relaxation task requires 'relax_dataset' in the config file"
                )

        except RuntimeError as e:
            self._process_error(e)
            raise e


# In order to improve this task, create specific configs for this
@registry.register_task("s2ef-to-is2re")
class S2EFtoIS2RE(BaseTask):
    def _process_error(self, e: RuntimeError):
        e_str = str(e)
        if (
            "find_unused_parameters" in e_str
            and "torch.nn.parallel.DistributedDataParallel" in e_str
        ):
            for name, parameter in self.trainer.model.named_parameters():
                if parameter.requires_grad and parameter.grad is None:
                    logging.warning(
                        f"Parameter {name} has no gradient. "
                        + "Consider removing it from the model."
                    )

    def setup(self, trainer, config_name):
        self.trainer = trainer
        if self.config.get("checkpoint") is not None:
            print("\n🔵 Resuming:\n  • ", end="", flush=True)
            cp_tmp = self.config["cp_data_to_tmp_dir"]
            self.trainer.load_checkpoint(self.config["checkpoint"])
            self.config["cp_data_to_tmp_dir"] = cp_tmp
            print()

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["checkpoint_dir"], "checkpoint.pt"
        )

        trainer_args = flags.parser.parse_args([f"--config={config_name}"])
        config_is2re = build_config(trainer_args)

        self.trainer.config["optim"] = config_is2re["optim"]
        self.trainer.config["dataset"] = config_is2re["dataset"]

        self.trainer.task_name = "is2re"

        # Make that cleaner:
        if not (self.trainer.config.get("is_debug", False)):
            self.trainer.config["logger"] = "wandb"
            self.trainer.config["wandb_name"] = (
                self.trainer.config["job_id"]
                + "-"
                + self.trainer.config["config"]
                + "-ft-is2re"
            )
            self.trainer.config["wandb_id"] = "-ft-is2re"
            self.trainer.load_logger()

        self.trainer.step = 0
        self.trainer.load_seed_from_config()
        self.trainer.load_datasets()
        self.trainer.load_optimizer()
        self.trainer.load_loss()
        self.trainer.load_extras()

        self.trainer.config["model"]["regress_forces"] = ""

        self.trainer.evaluator = Evaluator(
            task="is2re",
            model_regresses_forces=self.trainer.config["model"].get(
                "regress_forces", ""
            ),
        )

    def run(self):
        self.config = self.trainer.config
        try:
            torch.cuda.empty_cache()

            training_signal = self.trainer.train(
                disable_eval_tqdm=self.config.get("show_eval_progressbar", True),
                debug_batches=self.config.get("debug_batches", -1),
            )

            if training_signal == "SIGTERM":
                return

        except RuntimeError as e:
            self._process_error(e)
            raise e

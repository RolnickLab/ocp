"""scheduler.py
"""
import inspect
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

from ocpmodels.common.utils import warmup_lr_lambda
import pytorch_warmup as warmup


class LRScheduler:
    """
    Learning rate scheduler class for torch.optim learning rate schedulers

    Notes:
        If no learning rate scheduler is specified in the config the default
        scheduler is warmup_lr_lambda (ocpmodels.common.utils) not no scheduler,
        this is for backward-compatibility reasons. To run without a lr scheduler
        specify scheduler: "Null" in the optim section of the config.

    Args:
        config (dict): Optim dict from the input config
        optimizer (obj): torch optim object
    """

    def __init__(self, optimizer, optim_config, silent=False):
        self.optimizer = optimizer
        self.optim_config = optim_config.copy()
        self.warmup_scheduler = None
        self.silent = silent
        if self.optim_config.get("scheduler"):
            self.scheduler_type = self.optim_config["scheduler"]
        else:
            self.scheduler_type = "LambdaLR"

            def scheduler_lambda_fn(x):
                return warmup_lr_lambda(x, self.optim_config)

            self.optim_config["lr_lambda"] = scheduler_lambda_fn

        if (
            self.scheduler_type != "Null"
            and self.scheduler_type != "LinearWarmupCosineAnnealingLR"
        ):
            self.scheduler = getattr(lr_scheduler, self.scheduler_type)
            scheduler_args = self.filter_kwargs(self.optim_config)
            self.scheduler = self.scheduler(optimizer, **scheduler_args)
        elif self.scheduler_type == "LinearWarmupCosineAnnealingLR":
            T_max = self.optim_config.get("fidelity_max_steps")
            if T_max is None:
                T_max = self.optim_config["max_steps"]
                if not self.silent:
                    print(f"Using max_steps for scheduler -> {T_max}")
            else:
                if not self.silent:
                    print(f"Using fidelity_max_steps for scheduler -> {T_max}")
            if self.optim_config["warmup_steps"] > 0:
                self.warmup_scheduler = warmup.ExponentialWarmup(
                    self.optimizer, warmup_period=self.optim_config["warmup_steps"]
                )
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=1e-7
            )

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            if self.warmup_scheduler:
                with self.warmup_scheduler.dampening():
                    self.scheduler.step(epoch)
            else:
                self.scheduler.step()

    def filter_kwargs(self, optim_config):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(self.scheduler)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        filter_keys.remove("optimizer")
        scheduler_args = {
            arg: optim_config[arg] for arg in optim_config if arg in filter_keys
        }
        return scheduler_args

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]

class LossWeightScheduler:
    def __init__(self, optim_config, loss_type='auxiliary', max_steps=10000):
        """
        Initializes the scheduler with a specified weight decay strategy.

        Args:
            optim_config (dict): Configuration dictionary containing the settings for the scheduler.
            loss_type (str): Specifies the type of weight ('auxiliary' or 'energy').
            max_steps (int): Total steps over which the weight should be adjusted.
        """
        self.loss_type = loss_type
        self.max_steps = max_steps
        self.current_step = 0

        if loss_type == 'auxiliary':
            self.initial_weight = optim_config.get('auxiliary_task_weight', 1.0)
            self.min_weight = optim_config.get('auxiliary_min_weight', 0.1)
            self.decay = optim_config.get('auxiliary_decay', True)
            self.warmup_steps = optim_config.get('auxiliary_warmup_steps', 0)
            self.warmup_start_weight = optim_config.get('auxiliary_warmup_start_weight', 0)
            self.eta_min = optim_config.get('auxiliary_eta_min', 0.1)
            self.scheduler_type = optim_config.get('auxiliary_scheduler_type', 'linear')
        elif loss_type == 'energy':
            self.initial_weight = optim_config.get('energy_coefficient', 1.0)
            self.min_weight = optim_config.get('min_energy_weight', 1.0)
            self.decay = optim_config.get('energy_decay', False)
            self.warmup_steps = optim_config.get('energy_warmup_steps', 0)
            self.warmup_start_weight = optim_config.get('energy_warmup_start_weight', 0)
            self.eta_min = optim_config.get('energy_eta_min', 0.1)
            self.scheduler_type = optim_config.get('energy_scheduler_type', '')

        self.current_weight = self.warmup_start_weight if self.warmup_steps > 0 else self.initial_weight

    def step(self):
        """
        Update the weight based on the specified scheduler type.
        Adjusts the weight according to the current step and scheduling strategy,
        including handling of warmup phase if configured.
        """
        if not self.decay:
            return

        if self.current_step < self.warmup_steps:
            self.current_weight += (self.initial_weight - self.warmup_start_weight) / self.warmup_steps
        else:
            if self.scheduler_type == 'linear':
                weight_range = self.initial_weight - self.min_weight
                decay_step = weight_range / (self.max_steps - self.warmup_steps)
                self.current_weight = max(self.min_weight, self.current_weight - decay_step)
            elif self.scheduler_type == 'cosine':
                adjusted_step = self.current_step - self.warmup_steps
                adjusted_max_steps = self.max_steps - self.warmup_steps
                self.current_weight = self.eta_min + 0.5 * (self.initial_weight - self.eta_min) * (1 + np.cos(np.pi * adjusted_step / adjusted_max_steps))

        self.current_step += 1

    def get_weight(self):
        """
        Returns the current weight of the loss component being scheduled.
        """
        return self.current_weight

class EarlyStopper:
    """
    Class that stores the current best metric score and monitors whether
    it's improving or not. If it does not decrease for a certain number
    of validation calls (with some minimal improvement) then it tells the trainer
    to stop.
    """

    def __init__(
        self,
        patience=7,
        mode="min",
        min_abs_change=1e-5,
        store_all_steps=True,
        min_lr=-1,
        warmup_epochs=-1,
    ):
        """
        Whether train should stop or not.

        Args:
            patience (int, optional): How many calls to `should_stop` with no
                improvement before stopping training. Defaults to 7.
            mode (str, optional): "min" or "max". Defaults to "min".
            min_abs_change (float, optional): Minimum metric change to be considered an
                improvement. Defaults to 1e-5.
            store_all_steps (bool, optional): Whether to store all metrics passed to
                `should_stop` or only the last `patience` ones. Defaults to True.
            min_lr (bool, optional): Whether to stop when the current learning rate
                reaches the . Defaults to -1.

        Raises:
            ValueError: Unknown mode (neither min nor max)
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.min_abs_change = min_abs_change
        self.store_all_steps = store_all_steps
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.metrics = []

        if self.mode == "min":
            self.best_score = float("inf")
        elif self.mode == "max":
            self.best_score = float("-inf")
        else:
            raise ValueError("mode must be either min or max")

        self.early_stop = ""

    def should_stop(self, metric, lr=None, epoch=None):
        """
        Returns why the training should stop:
        • Empty string if the training shouldn't stop
        • "metric" if the metric has not improved for a certain number of
          steps.
        • "lr" if the learning rate has reached the minimum value.

        Stores the metric in `self.metrics`: all the steps if
        `self.store_all_steps` is `True`, otherwise only the last `n=self.patience`.

        Args:
            metric (Number): Metric to track.

        Returns:
            bool: Wether to stop training or not
        """
        metric = float(metric)
        self.metrics.append(metric)
        if not self.store_all_steps:
            self.metrics = self.metrics[-self.patience :]

        if self.mode == "min":
            if metric < self.best_score - self.min_abs_change:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == "max":
            if metric > self.best_score + self.min_abs_change:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = "metric"

        if lr is not None and lr <= self.min_lr:
            self.early_stop = "lr"

        if self.warmup_epochs > 0 and epoch is not None and epoch < self.warmup_epochs:
            self.early_stop = ""
            self.counter = 0

        return self.early_stop

    @property
    def reason(self):
        if self.early_stop == "metric":
            return (
                f"Early stopping after {self.counter} steps with no improvement:\n"
                + " -> ".join([f"{m:.6f}" for m in self.metrics[-self.patience :]])
            )
        elif self.early_stop == "lr":
            return f"Early stopping because learning rate reached {self.min_lr}"

        return ""

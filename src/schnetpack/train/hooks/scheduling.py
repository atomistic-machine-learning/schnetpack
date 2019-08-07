import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from schnetpack.train.hooks import Hook


class EarlyStoppingHook(Hook):
    r"""Hook to stop training if validation loss fails to improve.

    Args:
        patience (int): number of epochs which can pass without improvement
            of validation loss before training ends.
        threshold_ratio (float, optional): counter increases if
            curr_val_loss > (1-threshold_ratio) * best_loss

    """

    def __init__(self, patience, threshold_ratio=0.0001):
        self.best_loss = float("Inf")
        self.counter = 0
        self.threshold_ratio = threshold_ratio
        self.patience = patience

    @property
    def state_dict(self):
        return {"counter": self.counter}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.counter = state_dict["counter"]

    def on_validation_end(self, trainer, val_loss):
        if val_loss > (1 - self.threshold_ratio) * self.best_loss:
            self.counter += 1
        else:
            self.best_loss = val_loss
            self.counter = 0

        if self.counter > self.patience:
            trainer._stop = True


class WarmRestartHook(Hook):
    def __init__(
        self, T0=10, Tmult=2, each_step=False, lr_min=1e-6, lr_factor=1.0, patience=1
    ):
        self.scheduler = None
        self.each_step = each_step
        self.T0 = T0
        self.Tmult = Tmult
        self.Tmax = T0
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.patience = patience
        self.waiting = 0

        self.best_previous = float("Inf")
        self.best_current = float("Inf")

    def on_train_begin(self, trainer):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, self.Tmax, self.lr_min
        )
        self.init_opt_state = trainer.optimizer.state_dict()

    def on_batch_begin(self, trainer, train_batch):
        """Log at the beginning of train batch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.
            train_batch (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        """
        if self.each_step:
            self.scheduler.step()

    def on_epoch_begin(self, trainer):
        """Log at the beginning of train epoch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.

        """
        if not self.each_step:
            self.scheduler.step()

    def on_validation_end(self, trainer, val_loss):
        if self.best_current > val_loss:
            self.best_current = val_loss

        if self.scheduler.last_epoch >= self.Tmax:
            self.Tmax *= self.Tmult
            self.scheduler.last_epoch = -1
            self.scheduler.T_max = self.Tmax
            self.scheduler.base_lrs = [
                base_lr * self.lr_factor for base_lr in self.scheduler.base_lrs
            ]
            trainer.optimizer.load_state_dict(self.init_opt_state)

            if self.best_current >= self.best_previous:
                self.waiting += 1
            else:
                self.waiting = 0
                self.best_previous = self.best_current

            if self.waiting > self.patience:
                trainer._stop = True


class MaxEpochHook(Hook):
    """Hook to stop training when a maximum number of epochs is reached.

    Args:
       max_epochs (int): maximal number of epochs.

   """

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def on_epoch_begin(self, trainer):
        """Log at the beginning of train epoch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.

        """
        # stop training if max_epochs is reached
        if trainer.epoch > self.max_epochs:
            trainer._stop = True


class MaxStepHook(Hook):
    """Hook to stop training when a maximum number of steps is reached.

    Args:
        max_steps (int): maximum number of steps.

    """

    def __init__(self, max_steps):
        self.max_steps = max_steps

    def on_batch_begin(self, trainer, train_batch):
        """Log at the beginning of train batch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.
            train_batch (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        """
        # stop training if max_steps is reached
        if trainer.step > self.max_steps:
            trainer._stop = True


class LRScheduleHook(Hook):
    """Base class for learning rate scheduling hooks.

    This class provides a thin wrapper around torch.optim.lr_schedule._LRScheduler.

    Args:
        scheduler (torch.optim.lr_schedule._LRScheduler): scheduler.
        each_step (bool, optional): if set to True scheduler.step() is called every
            step, otherwise every epoch.

    """

    def __init__(self, scheduler, each_step=False):
        self.scheduler = scheduler
        self.each_step = each_step

    @property
    def state_dict(self):
        return {"scheduler": self.scheduler.state_dict()}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def on_train_begin(self, trainer):
        self.scheduler.last_epoch = trainer.epoch - 1

    def on_batch_begin(self, trainer, train_batch):
        """Log at the beginning of train batch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.
            train_batch (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        """
        if self.each_step:
            self.scheduler.step()

    def on_epoch_begin(self, trainer):
        """Log at the beginning of train epoch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.

        """
        if not self.each_step:
            self.scheduler.step()


class ReduceLROnPlateauHook(Hook):
    r"""Hook for reduce plateau learning rate scheduling.

    This class provides a thin wrapper around
    torch.optim.lr_schedule.ReduceLROnPlateau. It takes the parameters
    of ReduceLROnPlateau as arguments and creates a scheduler from it whose
    step() function will be called every epoch.

    Args:
        patience (int, optional): number of epochs with no improvement after which
            learning rate will be reduced. For example, if `patience = 2`, then we
            will ignore the first 2 epochs with no improvement, and will only
            decrease the LR after the 3rd epoch if the loss still hasn't improved then.
        factor (float, optional): factor by which the learning rate will be reduced.
            new_lr = lr * factor.
        min_lr (float or list, optional): scalar or list of scalars. A lower bound on
            the learning rate of all param groups or each group respectively.
        window_length (int, optional): window over which the accumulated loss will
            be averaged.
        stop_after_min (bool, optional): if enabled stops after minimal learning rate
            is reached.

    """

    def __init__(
        self,
        optimizer,
        patience=25,
        factor=0.5,
        min_lr=1e-6,
        window_length=1,
        stop_after_min=False,
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.scheduler = ReduceLROnPlateau(
            optimizer, patience=self.patience, factor=self.factor, min_lr=self.min_lr
        )
        self.window_length = window_length
        self.stop_after_min = stop_after_min
        self.window = []

    @property
    def state_dict(self):
        return {"scheduler": self.scheduler}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.scheduler = state_dict["scheduler"]

    def on_validation_end(self, trainer, val_loss):
        self.window.append(val_loss)
        if len(self.window) > self.window_length:
            self.window.pop(0)
        accum_loss = np.mean(self.window)

        self.scheduler.step(accum_loss)

        if self.stop_after_min:
            for i, param_group in enumerate(self.scheduler.optimizer.param_groups):
                old_lr = float(param_group["lr"])
                if old_lr <= self.scheduler.min_lrs[i]:
                    trainer._stop = True


class ExponentialDecayHook(Hook):
    """Hook for reduce plateau learning rate scheduling.

    This class provides a thin wrapper around torch.optim.lr_schedule.StepLR.
    It takes the parameters of StepLR as arguments and creates a scheduler
    from it whose step() function will be called every
    step.

    Args:
        gamma (float): Factor by which the learning rate will be
            reduced. new_lr = lr * gamma
        step_size (int): Period of learning rate decay.

    """

    def __init__(self, optimizer, gamma=0.96, step_size=100000):
        self.scheduler = StepLR(optimizer, step_size, gamma)

    def on_batch_end(self, trainer, train_batch, result, loss):
        self.scheduler.step()


class UpdatePrioritiesHook(Hook):
    r"""Hook for updating the priority sampler"""

    def __init__(self, prioritized_sampler, priority_fn):
        self.prioritized_sampler = prioritized_sampler
        self.update_fn = priority_fn

    def on_batch_end(self, trainer, train_batch, result, loss):
        idx = train_batch["_idx"]
        self.prioritized_sampler.update_weights(
            idx.data.cpu().squeeze(),
            self.update_fn(train_batch, result).data.cpu().squeeze(),
        )

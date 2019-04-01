import os
import time

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

__all__ = [
    "Hook",
    "LoggingHook",
    "TensorboardHook",
    "CSVHook",
    "EarlyStoppingHook",
    "MaxEpochHook",
    "MaxStepHook",
    "LRScheduleHook",
    "ReduceLROnPlateauHook",
    "ExponentialDecayHook",
    "WarmRestartHook",
]


class Hook:
    """Base class for hooks."""

    @property
    def state_dict(self):
        return {}

    @state_dict.setter
    def state_dict(self, state_dict):
        pass

    def on_train_begin(self, trainer):
        pass

    def on_train_ends(self, trainer):
        pass

    def on_train_failed(self, trainer):
        pass

    def on_epoch_begin(self, trainer):
        pass

    def on_batch_begin(self, trainer, train_batch):
        pass

    def on_batch_end(self, trainer, train_batch, result, loss):
        pass

    def on_validation_begin(self, trainer):
        pass

    def on_validation_batch_begin(self, trainer):
        pass

    def on_validation_batch_end(self, trainer, val_batch, val_result):
        pass

    def on_validation_end(self, trainer, val_loss):
        pass

    def on_epoch_end(self, trainer):
        pass


class LoggingHook(Hook):
    """Base class for logging hooks.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
    ):
        self.log_train_loss = log_train_loss
        self.log_validation_loss = log_validation_loss
        self.log_learning_rate = log_learning_rate
        self.log_path = log_path

        self._train_loss = 0
        self._counter = 0
        self.metrics = metrics

    def on_epoch_begin(self, trainer):
        if self.log_train_loss:
            self._train_loss = 0.0
            self._counter = 0
        else:
            self._train_loss = None

    def on_batch_end(self, trainer, train_batch, result, loss):
        if self.log_train_loss:
            n_samples = self._batch_size(result)
            self._train_loss += float(loss.data) * n_samples
            self._counter += n_samples

    def _batch_size(self, result):
        if type(result) is dict:
            n_samples = list(result.values())[0].size(0)
        elif type(result) in [list, tuple]:
            n_samples = result[0].size(0)
        else:
            n_samples = result.size(0)
        return n_samples

    def on_validation_begin(self, trainer):
        for metric in self.metrics:
            metric.reset()

    def on_validation_batch_end(self, trainer, val_batch, val_result):
        for metric in self.metrics:
            metric.add_batch(val_batch, val_result)


class CSVHook(LoggingHook):
    """Hook for logging training process to CSV files.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        every_n_epochs (int, optional): epochs after which logging takes place.

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        every_n_epochs=1,
    ):
        log_path = os.path.join(log_path, "log.csv")
        super(CSVHook, self).__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate
        )
        self._offset = 0
        self._restart = False
        self.every_n_epochs = every_n_epochs

    def on_train_begin(self, trainer):

        if os.path.exists(self.log_path):
            remove_file = False
            with open(self.log_path, "r") as f:
                # Ensure there is one entry apart from header
                lines = f.readlines()
                if len(lines) > 1:
                    self._offset = float(lines[-1].split(",")[0]) - time.time()
                    self._restart = True
                else:
                    remove_file = True

            # Empty up to header, remove to avoid adding header twice
            if remove_file:
                os.remove(self.log_path)
        else:
            self._offset = -time.time()
            # Create the log dir if it does not exists, since write cannot
            # create a full path
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        if not self._restart:
            log = ""
            log += "Time"

            if self.log_learning_rate:
                log += ",Learning rate"

            if self.log_train_loss:
                log += ",Train loss"

            if self.log_validation_loss:
                log += ",Validation loss"

            if len(self.metrics) > 0:
                log += ","

            for i, metric in enumerate(self.metrics):
                log += str(metric.name)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a+") as f:
                f.write(log + os.linesep)

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            ctime = time.time() + self._offset
            log = str(ctime)

            if self.log_learning_rate:
                log += "," + str(trainer.optimizer.param_groups[0]["lr"])

            if self.log_train_loss:
                log += "," + str(self._train_loss / self._counter)

            if self.log_validation_loss:
                log += "," + str(val_loss)

            if len(self.metrics) > 0:
                log += ","

            for i, metric in enumerate(self.metrics):
                m = metric.aggregate()
                if hasattr(m, "__iter__"):
                    log += ",".join([str(j) for j in m])
                else:
                    log += str(m)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a") as f:
                f.write(log + os.linesep)


class TensorboardHook(LoggingHook):
    """Hook for logging training process to tensorboard.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        every_n_epochs (int, optional): epochs after which logging takes place.
        img_every_n_epochs (int, optional):
        log_histogram (bool, optional):

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        every_n_epochs=1,
        img_every_n_epochs=10,
        log_histogram=False,
    ):
        from tensorboardX import SummaryWriter

        super(TensorboardHook, self).__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate
        )
        self.writer = SummaryWriter(self.log_path)
        self.every_n_epochs = every_n_epochs
        self.log_histogram = log_histogram
        self.img_every_n_epochs = img_every_n_epochs

    def on_epoch_end(self, trainer):
        if trainer.epoch % self.every_n_epochs == 0:
            if self.log_train_loss:
                self.writer.add_scalar(
                    "train/loss", self._train_loss / self._counter, trainer.epoch
                )
            if self.log_learning_rate:
                self.writer.add_scalar(
                    "train/learning_rate",
                    trainer.optimizer.param_groups[0]["lr"],
                    trainer.epoch,
                )

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            for metric in self.metrics:
                m = metric.aggregate()

                if np.isscalar(m):
                    self.writer.add_scalar(
                        "metrics/%s" % metric.name, float(m), trainer.epoch
                    )
                elif m.ndim == 2:
                    if trainer.epoch % self.img_every_n_epochs == 0:
                        import matplotlib.pyplot as plt

                        # tensorboardX only accepts images as numpy arrays.
                        # we therefore convert plots in numpy array
                        # see https://github.com/lanpa/tensorboard-pytorch/blob/master/examples/matplotlib_demo.py
                        fig = plt.figure()
                        plt.colorbar(plt.pcolor(m))
                        fig.canvas.draw()

                        np_image = np.fromstring(
                            fig.canvas.tostring_rgb(), dtype="uint8"
                        )
                        np_image = np_image.reshape(
                            fig.canvas.get_width_height()[::-1] + (3,)
                        )

                        plt.close(fig)

                        self.writer.add_image(
                            "metrics/%s" % metric.name, np_image, trainer.epoch
                        )

            if self.log_validation_loss:
                self.writer.add_scalar("train/val_loss", float(val_loss), trainer.step)

            if self.log_histogram:
                for name, param in trainer._model.named_parameters():
                    self.writer.add_histogram(
                        name, param.detach().cpu().numpy(), trainer.epoch
                    )

    def on_train_ends(self, trainer):
        self.writer.close()

    def on_train_failed(self, trainer):
        self.writer.close()


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
        if self.each_step:
            self.scheduler.step()

    def on_epoch_begin(self, trainer):
        if not self.each_step:
            self.scheduler.step()

    def on_validation_end(self, trainer, val_loss):
        if self.best_current < val_loss:
            self.best_current = val_loss

        if self.scheduler.last_epoch >= self.Tmax:
            self.Tmax *= self.Tmult
            self.scheduler.last_epoch = -1
            self.scheduler.T_max = self.Tmax
            self.scheduler.base_lrs = [
                base_lr * self.lr_factor for base_lr in self.scheduler.base_lrs
            ]
            trainer.optimizer.load_state_dict(self.init_opt_state)

            if self.best_current > self.best_previous:
                self.waiting += 1
            else:
                self.waiting = 0
                self.best_previous = self.best_current

            if self.waiting > self.patience:
                trainer._stop = True


class MaxEpochHook(Hook):
    """Hook to stop training if a maximal number of epochs is reached.

    Args:
       max_epochs (int): maximal number of epochs.

   """

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def on_epoch_begin(self, trainer):
        if trainer.epoch > self.max_epochs:
            trainer._stop = True


class MaxStepHook(Hook):
    """Hook to stop training if a maximal number of steps is reached.

    Args:
        max_steps (int): maximal number of steps.

    """

    def __init__(self, max_steps):
        self.max_steps = max_steps

    def on_batch_begin(self, trainer, train_batch):
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
        if self.each_step:
            self.scheduler.step()

    def on_epoch_begin(self, trainer):
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
        patience=25,
        factor=0.2,
        min_lr=1e-6,
        window_length=1,
        stop_after_min=False,
    ):
        self.scheduler = None
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.window_length = window_length
        self.stop_after_min = stop_after_min
        self.window = []

    @property
    def state_dict(self):
        return {
            "best": self.scheduler.best,
            "cooldown_counter": self.scheduler.cooldown_counter,
            "num_bad_epochs": self.scheduler.num_bad_epochs,
        }

    @state_dict.setter
    def state_dict(self, state_dict):
        self.scheduler.best = state_dict["best"]
        self.scheduler.cooldown_counter = state_dict["cooldown_counter"]
        self.scheduler.num_bad_epochs = state_dict["num_bad_epochs"]

    def on_train_begin(self, trainer):
        self.scheduler = ReduceLROnPlateau(
            trainer.optimizer,
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
        )

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

    def __init__(self, gamma=0.96, step_size=100000):
        self.scheduler = None
        self.gamma = gamma
        self.step_size = step_size

    def on_train_begin(self, trainer):
        self.scheduler = StepLR(trainer.optimizer, self.step_size, self.gamma)

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

import os
import time

import torch
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

__all__ = [
    'Hook', 'LoggingHook', 'TensorboardHook', 'CSVHook', 'EarlyStoppingHook', 'MaxEpochHook', 'MaxStepHook',
    'LRScheduleHook', 'ReduceLROnPlateauHook', 'ExponentialDecayHook', 'WarmRestartHook'
]


class Hook:
    """ Base class for hooks """

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
    """ Base class for hooks for logging.

        This class serves as base class for logging hooks.

        Args:
            log_path (str): path to directory to which log files will be written.
            metrics (list): list containing all metrics that will be logged. Metrics have to be subclass of spk.Metric.
            log_train_loss (bool): enable logging of training loss (default: True)
            log_validation_loss (bool): enable logging of validation loss (default: True)
            log_learning_rate (bool): enable logging of current learning rate (default: True)
    """

    def __init__(self, log_path, metrics, log_train_loss=True,
                 log_validation_loss=True, log_learning_rate=True):
        self.log_train_loss = log_train_loss
        self.log_validation_loss = log_validation_loss
        self.log_learning_rate = log_learning_rate
        self.log_path = log_path

        self._train_loss = 0
        self._counter = 0
        self.metrics = metrics

    def on_epoch_begin(self, trainer):
        if self.log_train_loss:
            self._train_loss = 0.
            self._counter = 0
        else:
            self._train_loss = None

    def on_batch_end(self, trainer, train_batch, result, loss):
        if self.log_train_loss:
            self._train_loss += float(loss.data)
            self._counter += 1

    def on_validation_begin(self, trainer):
        for metric in self.metrics:
            metric.reset()

    def on_validation_batch_end(self, trainer, val_batch, val_result):
        for metric in self.metrics:
            metric.add_batch(val_batch, val_result)


class CSVHook(LoggingHook):
    """ Hook for logging to csv files.

            This class provides an interface to write logging information about the training process to csv files.

            Args:
                log_path (str): path to directory to which log files will be written.
                metrics (list): list containing all metrics to be logged. Metrics have to be subclass of spk.Metric.
                log_train_loss (bool): enable logging of training loss (default: True)
                log_validation_loss (bool): enable logging of validation loss (default: True)
                log_learning_rate (bool): enable logging of current learning rate (default: True)
                every_n_epochs (int): interval after which logging takes place (default: 1)
        """

    def __init__(self, log_path, metrics, log_train_loss=True,
                 log_validation_loss=True, log_learning_rate=True, every_n_epochs=1):
        log_path = os.path.join(log_path, 'log.csv')
        super(CSVHook, self).__init__(log_path, metrics, log_train_loss,
                                      log_validation_loss, log_learning_rate)
        self._offset = 0
        self._restart = False
        self.every_n_epochs = every_n_epochs

    def on_train_begin(self, trainer):

        if os.path.exists(self.log_path):
            remove_file = False
            with open(self.log_path, 'r') as f:
                # Ensure there is one entry apart from header
                lines = f.readlines()
                if len(lines) > 1:
                    self._offset = float(lines[-1].split(',')[0]) - time.time()
                    self._restart = True
                else:
                    remove_file = True

            # Empty up to header, remove to avoid adding header twice
            if remove_file:
                os.remove(self.log_path)
        else:
            self._offset = -time.time()
            # Create the log dir if it does not exists, since write cannot create a full path
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        if not self._restart:
            log = ''
            log += 'Time'

            if self.log_train_loss:
                log += ',Learning rate'

            if self.log_train_loss:
                log += ',Train loss'

            if self.log_validation_loss:
                log += ',Validation loss'

            if len(self.metrics) > 0:
                log += ','

            for i, metric in enumerate(self.metrics):
                log += str(metric.name)
                if i < len(self.metrics) - 1:
                    log += ','

            with open(self.log_path, 'a+') as f:
                f.write(log + os.linesep)

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            ctime = time.time() + self._offset
            log = str(ctime)

            if self.log_learning_rate:
                log += ',' + str(trainer.optimizer.param_groups[0]['lr'])

            if self.log_train_loss:
                if hasattr(self._train_loss, "__iter__"):
                    train_string = ','.join([str(k) for k in self._train_loss])
                    log += ',' + train_string
                else:
                    log += ',' + str(self._train_loss)

            if self.log_validation_loss:
                if hasattr(val_loss, "__iter__"):
                    valid_string = ','.join([str(k) for k in val_loss])
                    log += ',' + valid_string
                else:
                    log += ',' + str(val_loss)

            if len(self.metrics) > 0:
                log += ','

            for i, metric in enumerate(self.metrics):
                m = metric.aggregate()
                if hasattr(m, "__iter__"):
                    log += ','.join([str(j) for j in m])
                else:
                    log += str(m)
                if i < len(self.metrics) - 1:
                    log += ','

            with open(self.log_path, 'a') as f:
                f.write(log + os.linesep)


class TensorboardHook(LoggingHook):
    """ Hook for logging to tensorboard.

        This class provides an interface to write logging information about the training process to tensorboard.

        Args:
            log_path (str): path to directory to which log files will be written.
            metrics (list): list containing all metrics that will be logged. Metrics have to be subclass of spk.Metric.
            log_train_loss (bool): enable logging of training loss (default: True)
            log_validation_loss (bool): enable logging of validation loss (default: True)
            log_learning_rate (bool): enable logging of current learning rate (default: True)
            every_n_epochs (int): interval after which logging takes place (default: 1)
    """

    def __init__(self, log_path, metrics, log_train_loss=True,
                 log_validation_loss=True, log_learning_rate=True, every_n_epochs=1,
                 log_histogram=False):
        from tensorboardX import SummaryWriter
        super(TensorboardHook, self).__init__(log_path, metrics, log_train_loss,
                                              log_validation_loss, log_learning_rate)
        self.writer = SummaryWriter(self.log_path)
        self.every_n_epochs = every_n_epochs
        self.log_histogram = log_histogram

    def on_epoch_end(self, trainer):
        if trainer.epoch % self.every_n_epochs == 0:
            if self.log_train_loss:
                self.writer.add_scalar("train/loss", self._train_loss / self._counter, trainer.epoch)
            if self.log_learning_rate:
                self.writer.add_scalar("train/learning_rate", trainer.optimizer.param_groups[0]['lr'], trainer.epoch)

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            for metric in self.metrics:
                m = metric.aggregate()

                if np.isscalar(m):
                    self.writer.add_scalar("metrics/%s" % metric.name, float(m), trainer.epoch)
                elif m.ndim == 2:
                    import matplotlib.pyplot as plt
                    # tensorboardX only accepts images as numpy arrays.
                    # we therefore convert plots in numpy array
                    # see https://github.com/lanpa/tensorboard-pytorch/blob/master/examples/matplotlib_demo.py
                    fig = plt.figure()
                    plt.colorbar(plt.pcolor(m))
                    fig.canvas.draw()

                    np_image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
                    np_image = np_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    plt.close(fig)

                    self.writer.add_image("metrics/%s" % metric.name,
                                          np_image, trainer.epoch)

            if self.log_validation_loss:
                self.writer.add_scalar("train/val_loss", float(val_loss), trainer.step)

            if self.log_histogram:
                for name, param in trainer._model.named_parameters():
                    self.writer.add_histogram(name, param.detach().cpu().numpy(), trainer.epoch)

    def on_train_ends(self, trainer):
        self.writer.close()

    def on_train_failed(self, trainer):
        self.writer.close()


class EarlyStoppingHook(Hook):
    """ Hook for early stopping.

        This hook can be used to stop training early if the validation loss has not improved over a certain number
        of epochs.

        Args:
            patience (int): number of epochs which can pass without improvement of validation loss before training ends.
            threshold_ratio (float): counter increases if curr_val_loss>(1-threshold_ratio)*best_loss (default: 0.0001)
    """

    def __init__(self, patience, threshold_ratio=0.0001):
        self.best_loss = float('Inf')
        self.counter = 0
        self.threshold_ratio = threshold_ratio
        self.patience = patience

    @property
    def state_dict(self):
        return {'counter': self.counter}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.counter = state_dict['counter']

    def on_validation_end(self, trainer, val_loss):
        if val_loss > (1 - self.threshold_ratio) * self.best_loss:
            self.counter += 1
        else:
            self.best_loss = val_loss
            self.counter = 0

        if self.counter > self.patience:
            trainer._stop = True


class WarmRestartHook(Hook):

    def __init__(self, T0=10, Tmult=2, each_step=False, lr_min=1e-6, patience=1):
        self.scheduler = None
        self.each_step = each_step
        self.T0 = T0
        self.Tmult = Tmult
        self.Tmax = T0
        self.lr_min = lr_min
        self.patience = patience
        self.waiting = 0

        self.best_previous = float('Inf')
        self.best_current = float('Inf')

    def on_train_begin(self, trainer):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer,
                                                                    self.Tmax, self.lr_min)
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
            trainer.optimizer.load_state_dict(self.init_opt_state)

            if self.best_current > self.best_previous:
                self.waiting += 1
            else:
                self.waiting = 0
                self.best_previous = self.best_current

            if self.waiting > self.patience:
                trainer._stop = True


class MaxEpochHook(Hook):
    """Hook for stopping after a maximal number of epochs.

       This hook can be used to stop training early if a certain number of epochs have passed.

       Args:
           max_epochs (int): maximal number of epochs.
   """

    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def on_epoch_begin(self, trainer):
        if trainer.epoch > self.max_epochs:
            trainer._stop = True


class MaxStepHook(Hook):
    """ Hook for stopping after a maximal number of steps.

        This hook can be used to stop training early if a certain number of steps have passed.

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
          each_step (bool): if set to true (false) scheduler.step() is called every step (every epoch) (default: False)
      """

    def __init__(self, scheduler, each_step=False):
        self.scheduler = scheduler
        self.each_step = each_step

    @property
    def state_dict(self):
        return {'scheduler': self.scheduler.state_dict()}

    @state_dict.setter
    def state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def on_train_begin(self, trainer):
        self.scheduler.last_epoch = trainer.epoch - 1

    def on_batch_begin(self, trainer, train_batch):
        if self.each_step:
            self.scheduler.step()

    def on_epoch_begin(self, trainer):
        if not self.each_step:
            self.scheduler.step()


class ReduceLROnPlateauHook(Hook):
    """Hook for reduce plateau learning rate scheduling.

      This class provides a thin wrapper around torch.optim.lr_schedule.ReduceLROnPlateau. It takes the parameters
      of ReduceLROnPlateau as arguments and creates a scheduler from it whose step() function will be called every
      epoch.

      Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. (default: 0.2).
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            (default: 25).
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. (default: 1e-6).
        window_length (int): window over which the accumulated loss will be averaged. (default: 1).
        stop_after_min (bool): if enabled stops after minimal learning rate is reached (default: False).
      """

    def __init__(self, optimizer, patience=25, factor=0.2, min_lr=1e-6, window_length=1,
                 stop_after_min=False):
        self.scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=factor, min_lr=min_lr)
        self.window_length = window_length
        self.stop_after_min = stop_after_min
        self.window = []

    @property
    def state_dict(self):
        return {
            'best': self.scheduler.best,
            'cooldown_counter': self.scheduler.cooldown_counter,
            'num_bad_epochs': self.scheduler.num_bad_epochs
        }

    @state_dict.setter
    def state_dict(self, state_dict):
        self.scheduler.best = state_dict['best']
        self.scheduler.cooldown_counter = state_dict['cooldown_counter']
        self.scheduler.num_bad_epochs = state_dict['num_bad_epochs']

    def on_validation_end(self, trainer, val_loss):
        self.window.append(val_loss)
        if len(self.window) > self.window_length:
            self.window.pop(0)
        accum_loss = np.mean(self.window)

        self.scheduler.step(accum_loss)

        if self.stop_after_min:
            for i, param_group in enumerate(self.scheduler.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                if old_lr <= self.scheduler.min_lrs[i]:
                    trainer._stop = True


class ExponentialDecayHook(Hook):
    """Hook for reduce plateau learning rate scheduling.

      This class provides a thin wrapper around torch.optim.lr_schedule.StepLR. It takes the parameters
      of StepLR as arguments and creates a scheduler from it whose step() function will be called every
      step.

      Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Factor by which the learning rate will be
            reduced. new_lr = lr * gamma (default: 0.96).
        step_size (int): Period of learning rate decay (default: 100 000)
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
        idx = train_batch['_idx']
        self.prioritized_sampler.update_weights(idx.data.cpu().squeeze(),
                                                self.update_fn(train_batch, result).data.cpu().squeeze())

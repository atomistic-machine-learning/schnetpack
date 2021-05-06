import torch

__all__ = ["ReduceLROnPlateau"]


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Extends PyTorch ReduceLROnPlateau by exponential smoothing of the monitored metric

    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
        smoothing_factor=0.0,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            mode (str): One of `min`, `max`. In `min` mode, lr will
                be reduced when the quantity monitored has stopped
                decreasing; in `max` mode it will be reduced when the
                quantity monitored has stopped increasing. Default: 'min'.
            factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
                Default: 10.
            threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            cooldown (int): Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
            min_lr (float or list): A scalar or a list of scalars. A
                lower bound on the learning rate of all param groups
                or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            smoothing_factor: smoothing_factor of exponential moving average
        """
        super().__init__(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
            verbose=verbose,
        )
        self.smoothing_factor = smoothing_factor
        self.ema_loss = None

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if self.ema_loss is None:
            self.ema_loss = current
        else:
            self.ema_loss = (
                self.smoothing_factor * self.ema_loss
                + (1.0 - self.smoothing_factor) * current
            )
        super().step(current, epoch)

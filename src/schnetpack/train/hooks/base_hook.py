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
        """Log at the beginning of train epoch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.

        """
        pass

    def on_batch_begin(self, trainer, train_batch):
        """Log at the beginning of train batch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.
            train_batch (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        """
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

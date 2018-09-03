import os
import torch
import numpy as np


class Trainer:
    r"""
       Class to train models.

       Runs an internal training loop, takes care of validation and can be extended with custom functionality using hooks.

       Args:
           model_path (str): path to the model directory
           model (torch.Module): model to be trained
           loss_fn (callable): loss function used for training the model
           optimizer (torch.optim.optimizer): optimizer used for training
           train_loader (torch.utils.data.DataLoader): data loader for training set
           validation_loader (torch.utils.data.DataLoader): data loader for validation set
           keep_n_checkpoints (int): number of saved checkpoints (default: 3)
           checkpoint_interval (int): interval after which checkpoints is saved (default: 10)
           hooks (list): hooks to customize training process (default: [])
           loss_is_normalized (bool): if true, the loss per datapoint will be reported. Otherwise, the accumulated loss
                                     (default: True)
       """
    def __init__(self, model_path, model, loss_fn, optimizer,
                 train_loader, validation_loader, keep_n_checkpoints=3,
                 checkpoint_interval=10, validation_interval=1, hooks=[], loss_is_normalized=True):
        self.model_path = model_path
        self.checkpoint_path = os.path.join(self.model_path, 'checkpoints')
        self.best_model = os.path.join(self.model_path, 'best_model')
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.validation_interval = validation_interval
        self.keep_n_checkpoints = keep_n_checkpoints
        self.hooks = hooks
        self.loss_is_normalized = loss_is_normalized

        self._model = model
        self._stop = False
        self.checkpoint_interval = checkpoint_interval

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if os.path.exists(self.checkpoint_path):
            self.restore_checkpoint()
        else:
            os.makedirs(self.checkpoint_path)
            self.epoch = 0
            self.step = 0
            self.best_loss = float('inf')
            self.store_checkpoint()

    def _check_is_parallel(self):
        return True if isinstance(self._model, torch.nn.DataParallel) else False

    def _load_model_state_dict(self, state_dict):
        if self._check_is_parallel():
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)

    @property
    def state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'step': self.step,
            'model': self._model.state_dict() if not self._check_is_parallel() else self._model.module.state_dict(),
            'best_loss': self.best_loss,
            'optimizer': self.optimizer.state_dict(),
            'hooks': [h.state_dict for h in self.hooks]
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.step = state_dict['step']
        self.best_loss = state_dict['best_loss']
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self._load_model_state_dict(state_dict['model'])

        for h, s in zip(self.hooks, self.state_dict['hooks']):
            h.state_dict = s

    def store_checkpoint(self):
        chkpt = os.path.join(self.checkpoint_path,
                             'checkpoint-' + str(self.epoch) + '.pth.tar')
        torch.save(self.state_dict, chkpt)

        chpts = [f for f in os.listdir(self.checkpoint_path)
                 if f.endswith('.pth.tar')]
        if len(chpts) > self.keep_n_checkpoints:
            chpt_epochs = [int(f.split('.')[0].split('-')[-1]) for f in chpts]
            sidx = np.argsort(chpt_epochs)
            for i in sidx[:-self.keep_n_checkpoints]:
                os.remove(os.path.join(self.checkpoint_path, chpts[i]))

    def restore_checkpoint(self, epoch=None):
        if epoch is None:
            epoch = max([int(f.split('.')[0].split('-')[-1]) for f in os.listdir(self.checkpoint_path)])
        epoch = str(epoch)

        chkpt = os.path.join(self.checkpoint_path,
                             'checkpoint-' + str(epoch) + '.pth.tar')
        self.state_dict = torch.load(chkpt)

    def train(self, device):
        r"""
        Starts training of model on a specified device.

        Args:
            device (torch.torch.Device): device on which training takes place

        """
        self._stop = False

        for h in self.hooks:
            h.on_train_begin(self)

        try:
            while True:
                self.epoch += 1

                for h in self.hooks:
                    h.on_epoch_begin(self)

                if self._stop:
                    break

                # perform training epoch
                for train_batch in self.train_loader:
                    self.optimizer.zero_grad()

                    for h in self.hooks:
                        h.on_batch_begin(self, train_batch)

                    # move input to gpu, if needed
                    train_batch = {
                        k: v.to(device)
                        for k, v in train_batch.items()
                    }

                    result = self._model(train_batch)
                    loss = self.loss_fn(train_batch, result)

                    loss.backward()
                    self.optimizer.step()
                    self.step += 1

                    for h in self.hooks:
                        h.on_batch_end(self, train_batch, result, loss)

                    if self._stop:
                        break

                if self.epoch % self.checkpoint_interval == 0:
                    self.store_checkpoint()

                # validation
                if self.epoch % self.validation_interval == 0 or self._stop:
                    for h in self.hooks:
                        h.on_validation_begin(self)

                    val_loss = 0.
                    n_val = 0
                    for val_batch in self.validation_loader:
                        # append batch_size
                        vsize = list(val_batch.values())[0].size(0)
                        n_val += vsize

                        for h in self.hooks:
                            h.on_validation_batch_begin(self)

                        # move input to gpu, if needed
                        val_batch = {
                            k: v.to(device)
                            for k, v in val_batch.items()
                        }

                        val_result = self._model(val_batch)
                        val_batch_loss = self.loss_fn(val_batch, val_result).data.cpu().numpy()
                        if self.loss_is_normalized:
                            val_loss += val_batch_loss * vsize
                        else:
                            val_loss += val_batch_loss

                        for h in self.hooks:
                            h.on_validation_batch_end(self, val_batch, val_result)

                    # weighted average over batches
                    if self.loss_is_normalized:
                        val_loss /= n_val

                    if self.best_loss > val_loss:
                        self.best_loss = val_loss
                        state_dict = self._model.state_dict() if not self._check_is_parallel() else self._model.module.state_dict()
                        torch.save(state_dict, self.best_model)

                    for h in self.hooks:
                        h.on_validation_end(self, val_loss)

                for h in self.hooks:
                    h.on_epoch_end(self)

                if self._stop:
                    break

            for h in self.hooks:
                h.on_train_ends(self)

        except Exception as e:
            for h in self.hooks:
                h.on_train_failed(self)

            raise e

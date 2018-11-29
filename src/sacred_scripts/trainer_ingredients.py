import os

import torch
from sacred import Ingredient
from torch.optim import Adam

import schnetpack as spk
from schnetpack.metrics import MeanAbsoluteError
from schnetpack.property_model import Properties


train_ingredient = Ingredient('train')


class SumMAE(MeanAbsoluteError):
    r"""
    Metric for mean absolute error of length.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs (Default: None)
        name (str): name used in logging for this metric. If set to `None`,
                    `LengthMAE_[target]` will be used (Default: None)
   """

    def __init__(self, target, model_output=None, axis=1, name=None):
        name = 'SumMAE_' + target if name is None else name
        self.axis= axis
        super(SumMAE, self).__init__(target, model_output, name=name)

    def _get_diff(self, y, yp):
        ypl = torch.sum(yp, dim=self.axis)
        return torch.sum(torch.abs(y - ypl))


@train_ingredient.config
def cfg():
    optimizer = 'adam'
    schedule = 'plateau'
    learning_rate = 1e-4
    max_epochs = None
    patience = 10
    lr_min = 1e-7
    lr_factor = 0.5
    t0 = None
    tmult = None


@train_ingredient.named_config
def sgdr():
    schedule = 'sgdr'
    t0 = 50
    tmult = 1
    patience = 2
    lr_factor = 1.


@train_ingredient.capture
def setup_trainer(model, modeldir, properties, loss_fn,
                  train_loader, val_loader,
                  optimizer, schedule, learning_rate, max_epochs,
                  patience, lr_min, lr_factor, t0, tmult, exclude=[],
                  custom_hooks=[]):
    hooks = [h for h in custom_hooks]
    if max_epochs is not None and max_epochs > 0:
        hooks.append(spk.train.MaxEpochHook(max_epochs))

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = filter(lambda p: p not in exclude, trainable_params)

    if optimizer == 'adam':
        optimizer = Adam(trainable_params, lr=learning_rate)
    else:
        raise NotImplementedError

    if schedule == 'plateau':
        schedule = spk.train.ReduceLROnPlateauHook(optimizer,
                                                   patience=patience,
                                                   factor=lr_factor,
                                                   min_lr=lr_min,
                                                   window_length=1,
                                                   stop_after_min=True)
    elif schedule == 'sgdr':
        schedule = spk.train.WarmRestartHook(T0=t0, Tmult=tmult,
                                             each_step=False,
                                             lr_min=lr_min,
                                             lr_factor=lr_factor,
                                             patience=patience)
    else:
        raise NotImplementedError
    hooks.append(schedule)

    metrics = [
        spk.metrics.MeanAbsoluteError(tgt, p)
        for p, tgt in properties.items()
        if tgt is not None
    ]
    metrics += [
        spk.metrics.RootMeanSquaredError(tgt, p)
        for p, tgt in properties.items()
        if tgt is not None
    ]
    if Properties.dipole_moment in properties or \
            Properties.total_dipole_moment in properties:
        metrics.append(
            SumMAE(spk.data.Structure.charge, Properties.charges)
        )

    logger = spk.train.TensorboardHook(os.path.join(modeldir, 'log'),
                                       metrics)
    hooks.append(logger)

    trainer = spk.train.Trainer(modeldir, model, loss_fn, optimizer,
                                train_loader, val_loader, hooks=hooks)
    return trainer

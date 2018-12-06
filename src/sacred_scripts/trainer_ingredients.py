import os
from sacred import Ingredient
from torch.optim import Adam

from schnetpack.train.hooks import *
from schnetpack.train.trainer import Trainer
from schnetpack.metrics import *


train_ingredient = Ingredient('trainer')


@train_ingredient.config
def cfg():
    optimizer = 'adam'
    schedule = None
    learning_rate = 1e-4
    max_epochs = None
    patience = None
    lr_min = None
    lr_factor = None
    t0 = None
    tmult = None
    custom_hooks = []
    custom_metrics = []


@train_ingredient.named_config
def sgdr():
    schedule = 'sgdr'
    t0 = 50
    tmult = 1
    patience = 2
    lr_factor = 1.


@train_ingredient.named_config
def plateau():
    schedule = 'plateau'
    patience = 10
    lr_min = 1e-7
    lr_factor = 0.5


@train_ingredient.named_config
def base_hooks():
    custom_hooks = ['csv']
    custom_metrics = ['rmse', 'mae']


@train_ingredient.capture
def build_hooks(custom_hooks, schedule, optimizer, patience, lr_factor, lr_min,
                t0, tmult, modeldir, max_epochs, properties):
    metrics = build_metrics(properties=properties)
    hooks = build_schedule(schedule, optimizer, patience, lr_factor,
                                  lr_min, t0, tmult)
    if max_epochs is not None and max_epochs > 0:
        hooks.append(MaxEpochHook(max_epochs))
    for hook in custom_hooks:
        hook = hook.lower()
        if hook == 'tensorboard':
            hooks.append(TensorboardHook(os.path.join(modeldir, 'log'),
                                         metrics))
        elif hook == 'csv':
            hooks.append(CSVHook(modeldir, metrics))
        else:
            raise NotImplementedError
    return hooks


@train_ingredient.capture
def build_metrics(custom_metrics, properties):
    metrics = []
    for metric in custom_metrics:
        metric = metric.lower()
        if metric == 'mae':
            metrics += [MeanAbsoluteError(tgt, p) for p, tgt in
                        properties.items() if tgt is not None]
        elif metric == 'rmse':
            metrics += [RootMeanSquaredError(tgt, p) for p, tgt in
                        properties.items() if tgt is not None]
        else:
            raise NotImplementedError
    return metrics


@train_ingredient.capture
def build_schedule(schedule, optimizer, patience, lr_factor, lr_min, t0, tmult):
    if schedule is None:
        return []
    elif schedule == 'plateau':
        schedule = ReduceLROnPlateauHook(optimizer,
                                         patience=patience,
                                         factor=lr_factor,
                                         min_lr=lr_min,
                                         window_length=1,
                                         stop_after_min=True)
    elif schedule == 'sgdr':
        schedule = WarmRestartHook(T0=t0, Tmult=tmult,
                                   each_step=False,
                                   lr_min=lr_min,
                                   lr_factor=lr_factor,
                                   patience=patience)
    else:
        raise NotImplementedError
    return schedule


@train_ingredient.capture
def setup_trainer(model, modeldir, loss_fn, train_loader, val_loader,
                  optimizer, learning_rate, properties, exclude=[]):
    hooks = build_hooks(modeldir=modeldir, properties=properties)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = filter(lambda p: p not in exclude, trainable_params)

    if optimizer == 'adam':
        optimizer = Adam(trainable_params, lr=learning_rate)
    else:
        raise NotImplementedError

    trainer = Trainer(modeldir, model, loss_fn, optimizer,
                      train_loader, val_loader, hooks=hooks)
    return trainer

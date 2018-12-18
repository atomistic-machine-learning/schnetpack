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
    hooks = []
    metrics = []
    threshold_ratio = None
    max_steps = None
    early_stopping = False
    lr_schedule = None
    logging_hooks = []


@train_ingredient.named_config
def sgdr():
    schedule = 'sgdr'
    t0 = 50
    tmult = 1
    patience = 2
    lr_min = 1e-7
    lr_factor = 1.


@train_ingredient.named_config
def plateau():
    schedule = 'plateau'
    patience = 10
    lr_min = 1e-7
    lr_factor = 0.5


@train_ingredient.named_config
def base_hooks():
    logging_hooks = ['csv']
    metrics = ['rmse', 'mae']


@train_ingredient.capture
def build_hooks(logging_hooks, schedule, optimizer, patience, lr_factor, lr_min,
                t0, tmult, modeldir, max_epochs, property_map,
                threshold_ratio, early_stopping, lr_schedule, max_steps):
    metrics_objects = build_metrics(property_map=property_map)
    hook_objects = build_schedule(schedule, optimizer, patience, lr_factor,
                                  lr_min, t0, tmult)
    hook_objects += build_logging_hooks(logging_hooks=logging_hooks,
                                        modeldir=modeldir,
                                        metrics_objects=metrics_objects)
    if early_stopping:
        hook_objects.append(EarlyStoppingHook(patience, threshold_ratio))
    if max_epochs is not None:
        hook_objects.append((MaxEpochHook(max_epochs)))
    if max_steps is not None:
        hook_objects.append(MaxStepHook(max_steps))
    if lr_schedule is not None:
        hook_objects.append(LRScheduleHook(lr_schedule))
    return hook_objects


@train_ingredient.capture
def build_logging_hooks(logging_hooks, modeldir, metrics_objects):
    hook_objects = []
    if not logging_hooks:
        return hook_objects
    for hook in logging_hooks:
        if hook == 'tensorboard':
            hook_objects.append(TensorboardHook(os.path.join(modeldir, 'log'),
                                                metrics_objects))
        elif hook == 'csv':
            hook_objects.append(CSVHook(modeldir, metrics_objects))
        else:
            raise NotImplementedError
    return hook_objects


@train_ingredient.capture
def build_schedule(schedule, optimizer, patience, lr_factor, lr_min, t0, tmult):
    hook_objects = []
    if schedule is None:
        return hook_objects
    elif schedule == 'plateau':
        hook_objects.append(ReduceLROnPlateauHook(optimizer,
                                                  patience=patience,
                                                  factor=lr_factor,
                                                  min_lr=lr_min,
                                                  window_length=1,
                                                  stop_after_min=True))
    elif schedule == 'sgdr':
        hook_objects.append(WarmRestartHook(T0=t0, Tmult=tmult,
                                            each_step=False,
                                            lr_min=lr_min,
                                            lr_factor=lr_factor,
                                            patience=patience))
    else:
        raise NotImplementedError
    return hook_objects


@train_ingredient.capture
def build_metrics(metrics, property_map):
    metrics_objects = []
    for metric in metrics:
        metric = metric.lower()
        if metric == 'mae':
            metrics_objects += [MeanAbsoluteError(tgt, p) for p, tgt in
                                property_map.items() if tgt is not None]
        elif metric == 'rmse':
            metrics_objects += [RootMeanSquaredError(tgt, p) for p, tgt in
                                property_map.items() if tgt is not None]
        else:
            raise NotImplementedError
    return metrics_objects


@train_ingredient.capture
def setup_trainer(model, modeldir, loss_fn, train_loader, val_loader,
                  optimizer, learning_rate, property_map, exclude=[]):
    hooks = build_hooks(modeldir=modeldir, property_map=property_map)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = filter(lambda p: p not in exclude, trainable_params)

    if optimizer == 'adam':
        optimizer = Adam(trainable_params, lr=learning_rate)
    else:
        raise NotImplementedError

    trainer = Trainer(modeldir, model, loss_fn, optimizer,
                      train_loader, val_loader, hooks=hooks)
    return trainer

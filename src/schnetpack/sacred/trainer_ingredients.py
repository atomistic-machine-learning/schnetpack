import os
from sacred import Ingredient
from torch.optim import Adam

from schnetpack.train.hooks import *
from schnetpack.train.trainer import Trainer
from schnetpack.metrics import *


train_ingredient = Ingredient('trainer')


@train_ingredient.config
def cfg():
    """
    base config for the trainer

    """
    optimizer = 'adam'
    schedule = None
    learning_rate = 1e-4
    max_epochs = None
    hooks = []
    metrics = []
    max_steps = None
    early_stopping = False
    lr_schedule = None
    logging_hooks = []


@train_ingredient.named_config
def sgdr():
    """
    default config for the SGDR schedule

    """
    schedule = 'sgdr'
    t0 = 50
    tmult = 1
    patience = 2
    lr_min = 1e-7
    lr_factor = 1.


@train_ingredient.named_config
def plateau():
    """
    default config for the ReduceOnPlateau schedule

    """
    schedule = 'plateau'
    patience = 10
    lr_min = 1e-7
    lr_factor = 0.5


@train_ingredient.named_config
def early_stopping():
    """
    default config for early stopping hook

    """
    early_stopping = True
    threshold_ratio = 0.
    patience = 0


@train_ingredient.named_config
def base_hooks():
    """
    default config for logging hooks

    """
    logging_hooks = ['csv']
    metrics = ['rmse', 'mae']


@train_ingredient.capture
def get_early_stopping_hook(patience, threshold_ratio):
    return EarlyStoppingHook(patience, threshold_ratio)


@train_ingredient.capture
def get_reduce_on_plateau_hook(optimizer, patience, lr_factor, lr_min):
    return ReduceLROnPlateauHook(optimizer, patience=patience, factor=lr_factor,
                                 min_lr=lr_min, window_length=1,
                                 stop_after_min=True)


@train_ingredient.capture
def get_warm_restart_hook(t0, tmult, lr_min, lr_factor, patience):
    return WarmRestartHook(T0=t0, Tmult=tmult, each_step=False, lr_min=lr_min,
                           lr_factor=lr_factor, patience=patience)


@train_ingredient.capture
def get_optimizer(optimizer, learning_rate, trainable_params):
    """
    build optimizer object

    Args:
        optimizer (str): name of the optimizer
        learning_rate (float): learning rate
        trainable_params (dict): trainable parameters of the model

    Returns:
        Optimizer object
    """
    if optimizer == 'adam':
        return Adam(trainable_params, lr=learning_rate)
    else:
        raise NotImplementedError

@train_ingredient.capture
def build_hooks(logging_hooks, schedule, modeldir, max_epochs, property_map,
                early_stopping, lr_schedule, max_steps):
    """
    build a list with hook objects

    Args:
        logging_hooks (list): list with names of logging hooks
        schedule (str): name of the lr_schedule
        modeldir (str): path to the model directory
        max_epochs (str): max number of training epochs
        property_map (dict): mapping between model properties and dataset
            properties
        early_stopping (bool): add the EarlyStoppingHook if set to True
        lr_schedule (torch.optim.lr_schedule._LRScheduler): scheduler
        max_steps (int): max number of training steps

    Returns:
        list of hook objects

    """
    metrics_objects = build_metrics(property_map=property_map)
    hook_objects = build_schedule(schedule)
    hook_objects += build_logging_hooks(logging_hooks=logging_hooks,
                                        modeldir=modeldir,
                                        metrics_objects=metrics_objects)
    if early_stopping:
        hook_objects.append(get_early_stopping_hook())
    if max_epochs is not None:
        hook_objects.append((MaxEpochHook(max_epochs)))
    if max_steps is not None:
        hook_objects.append(MaxStepHook(max_steps))
    if lr_schedule is not None:
        hook_objects.append(LRScheduleHook(lr_schedule))
    return hook_objects


@train_ingredient.capture
def build_logging_hooks(logging_hooks, modeldir, metrics_objects):
    """
    build a list of logging hooks

    Args:
        logging_hooks (list): names of the logging hooks
        modeldir (str): path to the model directory
        metrics_objects (list): list with schnetpack.metrics.Metric objects

    Returns:
        list of logging hooks

    """
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
def build_schedule(schedule):
    """
    builds a list with a single schedule hook

    Args:
        schedule (str): Name of the schedule

    Returns:
        list with a single schedule hook

    """
    hook_objects = []
    if schedule is None:
        return hook_objects
    elif schedule == 'plateau':
        hook_objects.append(get_reduce_on_plateau_hook())
    elif schedule == 'sgdr':
        hook_objects.append(get_warm_restart_hook())
    else:
        raise NotImplementedError
    return hook_objects


@train_ingredient.capture
def build_metrics(metrics, property_map):
    """
    builds a list with schnetpack.metrics.Metric objects

    Args:
        metrics (list): names of the metrics that should be used
        property_map (dict): mapping between model properties and dataset
            properties

    Returns:
        list of schnetpack.metrics.Metric objects

    """
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
                  property_map, exclude=[]):
    """
    build a trainer object

    Args:
        model (torch.nn.Module): model object
        modeldir (str): path to the model directory
        loss_fn (callable): loss function
        train_loader (schnetpack.data.AtomsLoader): dataloader for train data
        val_loader (schnetpack.data.Atomsloader):  dataloader fro validation
            data
        property_map (dict): maps the model properties on dataset properties
        exclude (list): model parameters that should not be optimized

    Returns:
        schnetpack.train.Trainer object

    """
    hooks = build_hooks(modeldir=modeldir, property_map=property_map)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = filter(lambda p: p not in exclude, trainable_params)

    optim = get_optimizer(trainable_params=trainable_params)

    trainer = Trainer(modeldir, model, loss_fn, optim, train_loader,
                      val_loader, hooks=hooks)
    return trainer

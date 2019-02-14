import os
from sacred import Ingredient
from schnetpack.train.hooks import *
from schnetpack.sacred.train_metrics_ingredients import metrics_ing,\
    build_metrics


logging_hook_ing = Ingredient('logging_hooks', ingredients=[metrics_ing])


@logging_hook_ing.config
def config():
    """configuration for the logging hook ingredient"""
    names = ['csv', 'tensorboard']


@logging_hook_ing.capture
def build_logging_hooks(training_dir, property_map, names):
    """
    build a list of logging hooks

    Args:
        training_dir (str): path to the training directory
        property_map (dict): property mapping between model and dataset
            properties
        names (list): names of the logging hooks

    Returns:
        list of logging hooks
    """
    metrics_objects = build_metrics(property_map=property_map)
    hook_objects = []
    if not names:
        return hook_objects
    for hook in names:
        if hook == 'tensorboard':
            hook_objects.append(TensorboardHook(os.path.join(training_dir,
                                                             'log'),
                                                metrics_objects))
        elif hook == 'csv':
            hook_objects.append(CSVHook(training_dir, metrics_objects))
        else:
            raise NotImplementedError
    return hook_objects


stopping_hook_ing = Ingredient('stopping_hooks')


@stopping_hook_ing.config
def config():
    """configuration for the stopping hook ingredient"""
    max_steps = None
    max_epochs = None
    patience = None
    threshold_ratio = None


@stopping_hook_ing.capture
def get_early_stopping_hook(patience, threshold_ratio):
    if threshold_ratio:
        return EarlyStoppingHook(patience, threshold_ratio)
    return EarlyStoppingHook(patience)


@stopping_hook_ing.capture
def build_stopping_hooks(max_steps, max_epochs, patience):
    hook_objects = []
    if patience:
        hook_objects.append(get_early_stopping_hook())
    if max_epochs is not None:
        hook_objects.append((MaxEpochHook(max_epochs)))
    if max_steps is not None:
        hook_objects.append(MaxStepHook(max_steps))
    return hook_objects


scheduling_hook_ing = Ingredient('schedule_hooks')


@scheduling_hook_ing.config
def config():
    """configuration for the scheduling hook ingredient"""
    name = None


@scheduling_hook_ing.named_config
def reduce_on_plateau():
    name = 'reduce_on_plateau'
    patience = 25
    factor = 0.2
    min_lr = 1e-6
    window_length = 1
    stop_after_min = False


@scheduling_hook_ing.named_config
def exponential_decay():
    name = 'exponential_decay'
    gamma=0.96
    step_size=100000


@scheduling_hook_ing.named_config
def warm_restart():
    name = 'warm_restart'
    T0 = 10
    Tmult = 2
    each_step = False
    lr_min = 1e-6
    lr_factor = 1.
    patience = 1


@scheduling_hook_ing.named_config
def lr_schedule():
    name = 'lr_schedule'
    schedule = None
    each_step = False


@scheduling_hook_ing.capture
def build_schedule_hook(name):
    if name is None:
        return []
    elif name == 'reduce_on_plateau':
        return [get_reduce_on_plateau_hook()]
    elif name == 'exponential_decay':
        return [get_exponential_decay_hook()]
    elif name == 'warm_restart':
        return [get_warm_restart_hook()]
    elif name == 'lr_scheduler':
        return [get_lr_scheduler_hook()]
    else:
        raise NotImplementedError


@scheduling_hook_ing.capture
def get_reduce_on_plateau_hook(patience, lr_factor, lr_min):
    return ReduceLROnPlateauHook(patience=patience, factor=lr_factor,
                                 min_lr=lr_min, window_length=1,
                                 stop_after_min=True)


@scheduling_hook_ing.capture
def get_exponential_decay_hook(gamma, step_size):
    return ExponentialDecayHook(gamma, step_size)


@scheduling_hook_ing.capture
def get_warm_restart_hook(t0, Tmult, each_step, lr_min, lr_factor, patience):
    return WarmRestartHook(T0=t0, Tmult=Tmult, each_step=each_step,
                           lr_min=lr_min, lr_factor=lr_factor,
                           patience=patience)


@scheduling_hook_ing.capture
def get_lr_scheduler_hook(schedule, each_step):
    if schedule is None:
        raise NotImplementedError
    return LRScheduleHook(schedule, each_step)


hooks_ing = Ingredient('hooks', ingredients=[logging_hook_ing,
                                             stopping_hook_ing,
                                             scheduling_hook_ing])


@hooks_ing.config
def config():
    """configuration for the hook ingredient"""
    pass


@hooks_ing.capture
def build_hooks(train_dir, property_map):
    hook_objects = []
    hook_objects += build_logging_hooks(train_dir, property_map)
    hook_objects += build_schedule_hook()
    hook_objects += build_stopping_hooks()
    return hook_objects


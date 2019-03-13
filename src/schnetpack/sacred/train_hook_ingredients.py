import os
from sacred import Ingredient
from schnetpack.train.hooks import *
from schnetpack.sacred.train_metrics_ingredients import metrics_ing,\
    build_metrics


logging_hook_ing = Ingredient('logging_hooks', ingredients=[metrics_ing])


@logging_hook_ing.config
def config():
    r"""
    configuration for logging hooks
    """
    names = ['csv', 'tensorboard']  # hooks that will be used during training


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
    r"""
    Settings for hooks that are used for early stopping during training.
    """
    max_steps = None        # maximum number of steps
    max_epochs = None       # maximum number of epochs
    patience = None         # maximum number of training epochs without improvement on val loss
    threshold_ratio = None  # threshold of validation loss


@stopping_hook_ing.capture
def get_early_stopping_hook(patience, threshold_ratio):
    """
    Build hook for early stopping.
    Args:
        patience (int): maximum number of training epochs without improvement
            on val loss
        threshold_ratio (float): threshold of validation loss

    Returns:
        EarlyStoppingHook
    """
    if threshold_ratio:
        return EarlyStoppingHook(patience, threshold_ratio)
    return EarlyStoppingHook(patience)


@stopping_hook_ing.capture
def build_stopping_hooks(max_steps, max_epochs, patience):
    """
    Build selected stopping hooks.
    Args:
        max_steps (int): maximum number of training steps
        max_epochs (int): maximum number of training epochs
        patience (int): maximum number of epochs without improvement on val loss

    Returns:
        hooks for early stopping
    """
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
    r"""
    Settings for hooks that schedule the learning rate during training.
    """
    name = None     # name of scheduling hook


@scheduling_hook_ing.named_config
def reduce_on_plateau():
    r"""
    Settings for ReduceOnPlateau hook.
    Adds:
        patience (int): number of epochs without improvement on val loss
        factor (float):
        min_lr (float): stop training when min_lr is reached
        window_length (int):
        stop_after_min (bool):
    """
    name = 'reduce_on_plateau'
    patience = 25
    factor = 0.2
    min_lr = 1e-6
    window_length = 1
    stop_after_min = False


@scheduling_hook_ing.named_config
def exponential_decay():
    r"""
    Settings for exponential lr decay.
    Adds:
        gamma (float): lr decay factor
        step_size (int): decay learning after step_size steps
    """
    name = 'exponential_decay'
    gamma=0.96
    step_size=100000


@scheduling_hook_ing.named_config
def warm_restart():
    r"""
    Settings for WarmRestartHook.
    Adds:
        T0 (int):
        Tmult (int):
        each_step (bool):
        lr_min (float):
        lr_factor (float):
        patience (int):
    """
    name = 'warm_restart'
    T0 = 10
    Tmult = 2
    each_step = False
    lr_min = 1e-6
    lr_factor = 1.
    patience = 1


@scheduling_hook_ing.named_config
def lr_schedule():
    r"""
    Settings for custom lr schedules.
    Adds:
        schedule: custom schedule
        each_step (bool): update lr at each step if True
    """
    name = 'lr_schedule'
    schedule = None
    each_step = False


@scheduling_hook_ing.capture
def build_schedule_hook(name):
    """
    Builds a hook for scheduling the lr.
    Args:
        name (str): name of selected hook

    Returns:
        hook for lr scheduling
    """
    if name is None:
        return []
    elif name == 'reduce_on_plateau':
        return [get_reduce_on_plateau_hook()]
    elif name == 'exponential_decay':
        return [get_exponential_decay_hook()]
    elif name == 'warm_restart':
        return [get_warm_restart_hook()]
    elif name == 'lr_schedule':
        return [get_lr_scheduler_hook()]
    else:
        raise NotImplementedError


@scheduling_hook_ing.capture
def get_reduce_on_plateau_hook(patience, factor, min_lr,
                               window_length, stop_after_min):
    return ReduceLROnPlateauHook(patience=patience, factor=factor,
                                 min_lr=min_lr, window_length=window_length,
                                 stop_after_min=stop_after_min)


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
    pass


@hooks_ing.capture
def build_hooks(train_dir, property_map):
    r"""
    Builds selected hooks for logging, early stopping and scheduling.
    Args:
        train_dir (str): path to the training directory
        property_map (dict): mapping between model and dataset properties

    Returns:
        list of train hooks
    """
    hook_objects = []
    hook_objects += build_logging_hooks(train_dir, property_map)
    hook_objects += build_schedule_hook()
    hook_objects += build_stopping_hooks()
    return hook_objects


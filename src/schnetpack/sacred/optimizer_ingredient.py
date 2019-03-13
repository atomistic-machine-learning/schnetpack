from sacred import Ingredient
from torch.optim import Adam


optimizer_ing = Ingredient('optimizer')


@optimizer_ing.config
def config():
    r"""
    settings for optimizer class
    """
    name = 'adam'           # optimizer type
    learning_rate = 1e-4    # initial learning rate


@optimizer_ing.capture
def build_optimizer(name, learning_rate, trainable_params):
    """
    build optimizer object

    Args:
        name (str): name of the optimizer
        learning_rate (float): learning rate
        trainable_params (dict): trainable parameters of the model

    Returns:
        Optimizer object
    """
    if name == 'adam':
        return Adam(trainable_params, lr=learning_rate)
    else:
        raise NotImplementedError

import torch
from sacred import Ingredient


loss_ing = Ingredient("loss_function")


@loss_ing.config
def config():
    r"""
    Settings for the loss function that will be used during training.
    """
    loss_tradeoff = {}  # weighting  dictionary for loss calculation


@loss_ing.capture
def build_loss(property_map, loss_tradeoff):
    """
    Build the loss function.

    Args:
        property_map (dict): mapping between the model properties and the
            dataset properties
        loss_tradeoff (dict): contains tradeoff factors for properties,
            if needed

    Returns:
        loss function

    """

    def loss_fn(batch, result):
        loss = 0.0
        for p, tgt in property_map.items():
            if tgt is not None:
                diff = batch[tgt] - result[p]
                diff = diff ** 2
                err_sq = torch.mean(diff)
                if p in loss_tradeoff.keys():
                    err_sq *= loss_tradeoff[p]
                loss += err_sq
        return loss

    return loss_fn

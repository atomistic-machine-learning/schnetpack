import torch


def mse_loss(properties, loss_tradeoff={}):
    """
    Build the loss function.

    Args:
        properties (list): mapping between the model properties and the
            dataset properties
        loss_tradeoff (dict): multiply loss value of property with tradeoff factor

    Returns:
        loss function

    """

    def loss_fn(batch, result):
        loss = 0.0
        for prop in properties:
            diff = batch[prop] - result[prop]
            diff = diff ** 2
            err_sq = torch.mean(diff)
            if prop in loss_tradeoff.keys():
                loss *= loss_tradeoff[prop]
            loss += err_sq
        return loss

    return loss_fn

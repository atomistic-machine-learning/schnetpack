from sacred import Ingredient

from schnetpack.train.trainer import Trainer
from schnetpack.sacred.optimizer_ingredient import optimizer_ing, build_optimizer
from schnetpack.sacred.train_hook_ingredients import hooks_ing, build_hooks
from schnetpack.sacred.loss_ingredient import loss_ing, build_loss

train_ingredient = Ingredient(
    "trainer", ingredients=[optimizer_ing, hooks_ing, loss_ing]
)


@train_ingredient.config
def cfg():
    pass


@train_ingredient.capture
def setup_trainer(model, train_dir, train_loader, val_loader, property_map, exclude=[]):
    """
    build a trainer object

    Args:
        model (torch.nn.Module): model object
        train_dir (str): path to the training directory
        train_loader (schnetpack.data.AtomsLoader): dataloader for train data
        val_loader (schnetpack.data.Atomsloader):  dataloader fro validation
            data
        property_map (dict): maps the model properties on dataset properties
        exclude (list): model parameters that should not be optimized

    Returns:
        schnetpack.train.Trainer object

    """
    hooks = build_hooks(train_dir=train_dir, property_map=property_map)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = filter(lambda p: p not in exclude, trainable_params)

    optim = build_optimizer(trainable_params=trainable_params)
    loss_fn = build_loss(property_map=property_map)
    trainer = Trainer(
        train_dir, model, loss_fn, optim, train_loader, val_loader, hooks=hooks
    )
    return trainer

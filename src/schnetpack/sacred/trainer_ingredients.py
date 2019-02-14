from sacred import Ingredient

from schnetpack.train.trainer import Trainer
from schnetpack.sacred.optimizer_ingredient import optimizer_ing,\
    build_optimizer
from schnetpack.sacred.train_hook_ingredients import hooks_ing,\
    build_hooks


train_ingredient = Ingredient('trainer', ingredients=[optimizer_ing,
                                                      hooks_ing])


@train_ingredient.config
def cfg():
    """configuration for the trainer ingredient"""
    optimizer = 'adam'
    schedule = None
    learning_rate = 1e-4
    max_epochs = None
    metrics = []
    max_steps = None
    early_stopping = False
    lr_schedule = None
    logging_hooks = []


@train_ingredient.capture
def setup_trainer(model, train_dir, loss_fn, train_loader, val_loader,
                  property_map, exclude=[]):
    """
    build a trainer object

    Args:
        model (torch.nn.Module): model object
        training_dir (str): path to the training directory
        loss_fn (callable): loss function
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

    trainer = Trainer(train_dir, model, loss_fn, optim, train_loader,
                      val_loader, hooks=hooks)
    return trainer

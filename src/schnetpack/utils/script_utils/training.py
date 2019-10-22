import os

import schnetpack as spk
import torch
from torch.optim import Adam


__all__ = ["get_trainer", "simple_loss_fn", "tradeoff_loss_fn", "get_metrics"]


def get_trainer(args, model, train_loader, val_loader, metrics):
    # setup optimizer
    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    # setup hook and logging
    hooks = [spk.train.MaxEpochHook(args.max_epochs)]
    if args.max_steps:
        hooks.append(spk.train.MaxStepHook(max_steps=args.max_steps))

    schedule = spk.train.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=args.lr_patience,
        factor=args.lr_decay,
        min_lr=args.lr_min,
        window_length=1,
        stop_after_min=True,
    )
    hooks.append(schedule)

    if args.logger == "csv":
        logger = spk.train.CSVHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)
    elif args.logger == "tensorboard":
        logger = spk.train.TensorboardHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)

    # setup loss function
    loss_fn = get_loss_fn(args)

    # setup trainer
    trainer = spk.train.Trainer(
        args.modelpath,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_interval=args.checkpoint_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        hooks=hooks,
    )
    return trainer


def get_loss_fn(args):
    derivative = spk.utils.get_derivative(args)
    contributions = spk.utils.get_contributions(args)
    stress = spk.utils.get_stress(args)

    # simple loss function for training on property only
    if derivative is None and contributions is None and stress is None:
        return simple_loss_fn(args)

    # loss function with tradeoff weights
    if type(args.rho) == float:
        rho = dict(property=args.rho, derivative=1 - args.rho)
    else:
        rho = dict()
        rho["property"] = (
            1.0 if "property" not in args.rho.keys() else args.rho["property"]
        )
        if derivative is not None:
            rho["derivative"] = (
                1.0 if "derivative" not in args.rho.keys() else args.rho["derivative"]
            )
        if contributions is not None:
            rho["contributions"] = (
                1.0
                if "contributions" not in args.rho.keys()
                else args.rho["contributions"]
            )
        if stress is not None:
            rho["stress"] = (
                1.0 if "stress" not in args.rho.keys() else args.rho["stress"]
            )
        # type cast of rho values
        for key in rho.keys():
            rho[key] = float(rho[key])
        # norm rho values
        norm = sum(rho.values())
        for key in rho.keys():
            rho[key] = rho[key] / norm
    property_names = dict(
        property=args.property,
        derivative=derivative,
        contributions=contributions,
        stress=stress,
    )
    return tradeoff_loss_fn(rho, property_names)


def simple_loss_fn(args):
    def loss(batch, result):
        diff = batch[args.property] - result[args.property]
        diff = diff ** 2
        err_sq = torch.mean(diff)
        return err_sq

    return loss


def tradeoff_loss_fn(rho, property_names):
    def loss(batch, result):
        err = 0.0
        for prop, tradeoff_weight in rho.items():
            diff = batch[property_names[prop]] - result[property_names[prop]]
            diff = diff ** 2
            err += tradeoff_weight * torch.mean(diff)

        return err

    return loss


def get_metrics(args):
    # setup property metrics
    metrics = [
        spk.train.metrics.MeanAbsoluteError(args.property, args.property),
        spk.train.metrics.RootMeanSquaredError(args.property, args.property),
    ]

    # add metrics for derivative
    derivative = spk.utils.get_derivative(args)
    if derivative is not None:
        metrics += [
            spk.train.metrics.MeanAbsoluteError(
                derivative, derivative, element_wise=True
            ),
            spk.train.metrics.RootMeanSquaredError(
                derivative, derivative, element_wise=True
            ),
        ]

    return metrics

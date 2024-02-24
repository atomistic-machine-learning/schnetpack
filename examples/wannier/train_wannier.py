from pathlib import Path

import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn
import torch
import torchmetrics
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.cli import instantiate_class as lit_instantiate_class
from schnetpack.atomistic.wannier import WannierCenter
from schnetpack.data import AtomsDataModule


def instantiate_class(d: dict | list[dict]):
    """Instantiate one or a list of LightningModule classes from a dictionary."""
    args = tuple()  # no positional args
    if isinstance(d, dict):
        return lit_instantiate_class(args, d)
    elif isinstance(d, list):
        return [lit_instantiate_class(args, x) for x in d]
    else:
        raise ValueError(f"Cannot instantiate class from {d}")


def get_args(path: Path):
    """Get the arguments from the config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_module(module_path: str):
    """Get the module from the module path."""
    module_path = module_path.split(".")
    module = __import__(".".join(module_path[:-1]), fromlist=[module_path[-1]])
    return getattr(module, module_path[-1])


def get_datamodule(
    datapath: str,
    split_file: str,
    batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
    cutoff: float,
    **kwargs,
):
    """
    Create a datamodule for existing db and split files.

    Args:
        datapath:
        split_file:
        batch_size:
        val_batch_size:
        test_batch_size:
        cutoff:
        **kwargs: other parameters for AtomsDataModule
    """
    dm = AtomsDataModule(
        datapath,
        split_file=split_file,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        distance_unit="Ang",
        property_units={"wan": "Ang"},
        transforms=[trn.ASENeighborList(cutoff=cutoff), trn.CastTo32()],
        **kwargs,
    )

    # TODO, check whether dm.setup() is needed

    return dm


def update_model_configs(config: dict) -> dict:
    """
    Copy model parameters from other sections of the config file.

    Some parameters like cutoff are used in different places. To avoid redundancy, which
    is error-prone, we just provide the parameter in one place in the config file and
    copy it to other places where it is needed here.

    Note the update is in place.
    """
    config["model"]["cutoff"] = config["datamodule"]["cutoff"]

    return config


def create_model(
    model_hparams: dict = None,
    # loss_hparams: dict = None,
    task_hparams: dict = None,
    other_hparams: dict = None,
) -> pl.LightningModule:
    # model
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(
        n_rbf=model_hparams["n_rbf"],
        cutoff=model_hparams["cutoff"],
    )
    schnet = spk.representation.PaiNN(
        n_atom_basis=model_hparams["n_atom_basis"],
        n_interactions=model_hparams["n_interactions"],
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(model_hparams["cutoff"]),
    )
    pred_wan = WannierCenter(
        n_in=model_hparams["n_atom_basis"],
        dipole_key=model_hparams["dipole_key"],
    )
    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_wan],
        postprocessors=[trn.CastTo64()],
    )

    # loss
    output_wan = spk.task.ModelOutput(
        name="wan",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.0,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )

    # optimizer and scheduler
    model = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_wan],
        optimizer_cls=get_module(task_hparams["optimizer_cls"]),
        optimizer_args=task_hparams["optimizer_args"],
        scheduler_cls=get_module(task_hparams["scheduler_cls"]),
        scheduler_args=task_hparams["scheduler_args"],
        scheduler_monitor=task_hparams.get("scheduler_monitor", None),
        # other_hparams passed to the model so that they are logged to wandb
        other_hparams=other_hparams,
    )

    return model


def main(config: dict):
    pl.seed_everything(config["seed_everything"])

    dm = get_datamodule(**config["datamodule"])

    config = update_model_configs(config)
    model = create_model(
        model_hparams=config.pop("model"),
        task_hparams=config.pop("atomistic_task"),
        other_hparams=config,
    )

    # create all callbacks and logger
    try:
        callbacks = instantiate_class(config["trainer"].pop("callbacks"))
    except KeyError:
        callbacks = None

    try:
        logger = instantiate_class(config["trainer"].pop("logger"))
    except KeyError:
        logger = None

    trainer = Trainer(callbacks=callbacks, logger=logger, **config["trainer"])

    # fit
    trainer.fit(model, datamodule=dm)
    print("default_root_dir:", trainer.default_root_dir)
    print(f"Best checkpoint path: {trainer.checkpoint_callback.best_model_path}")

    # test
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == "__main__":
    config_file = Path(__file__).parent / "configs" / "config_wannier.yaml"
    config = get_args(config_file)
    main(config)

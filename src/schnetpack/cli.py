import argparse
import logging
import os
import random
import socket
import tempfile
import uuid

import hydra
import numpy as np
import petname
import torch
from ase.db import connect
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger
from tqdm import tqdm

from schnetpack import properties
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.lightning import AtomisticTask, PredictionWriter
from schnetpack.lightning.utils import (
    load_task_from_checkpoint,
    log_hyperparameters,
    trainer_fit_kwargs_for_checkpoint,
)
from schnetpack.model.compile import CompiledPotential
from schnetpack.utils import load_model, str2class
from schnetpack.utils.script import print_config

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()), use_cache=True)
OmegaConf.register_new_resolver("petname", lambda: petname.generate())
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)

header = r"""
   _____      __    _   __     __  ____             __
  / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__
  \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/
 ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<
/____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|
"""


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    log.info("Running on host: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            "Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            """
        Config incomplete! You have to specify at least `data` and `model`!
        For an example, try one of our pre-defined experiments:
        > spktrain experiment=qm9_atomwise
        """
        )
        return

    if os.path.exists("config.yaml"):
        log.info(
            f"Config already exists in given directory {os.path.abspath('.')}."
            + " Attempting to continue training."
        )

        # save old config
        old_config = OmegaConf.load("config.yaml")
        count = 1
        while os.path.exists(f"config.old.{count}.yaml"):
            count += 1
        with open(f"config.old.{count}.yaml", "w") as f:
            OmegaConf.save(old_config, f, resolve=False)

        # resume from latest checkpoint
        if config.run.ckpt_path is None:
            if os.path.exists("checkpoints/last.ckpt"):
                config.run.ckpt_path = "checkpoints/last.ckpt"

        if config.run.ckpt_path is not None:
            log.info(
                f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
            )
    else:
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f, resolve=False)

    # Set matmul precision if specified
    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
    else:
        # choose seed randomly
        with open_dict(config):
            config.seed = random.randint(0, 2**32 - 1)
        log.info(f"Seed randomly with <{config.seed}>")
    seed_everything(seed=config.seed, workers=True)

    if config.get("print_config"):
        print_config(config, resolve=False)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.data,
        train_sampler_cls=(
            str2class(config.data.train_sampler_cls)
            if config.data.train_sampler_cls
            else None
        ),
    )

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    # Init Lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: list[Logger] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                l = hydra.utils.instantiate(lg_conf)

                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=task, trainer=trainer)

    # Train the model
    log.info("Starting training.")
    fit_kwargs = (
        trainer_fit_kwargs_for_checkpoint(trainer)
        if config.run.ckpt_path is not None
        else {}
    )
    trainer.fit(
        model=task,
        datamodule=datamodule,
        ckpt_path=config.run.ckpt_path,
        **fit_kwargs,
    )

    # Lightning's EarlyStopping stops training silently on non-finite metrics
    # (check_finite=True), so fail explicitly instead of testing the model
    for early_stopping in trainer.early_stopping_callbacks:
        if not early_stopping.check_finite:
            continue
        value = trainer.callback_metrics.get(early_stopping.monitor)
        if value is not None and not torch.isfinite(value).all():
            raise RuntimeError(
                f"Training stopped at epoch {early_stopping.stopped_epoch} because the "
                f"monitored metric became non-finite "
                f"({early_stopping.monitor} = {value.item()}). "
                "Skipping testing and model export."
            )

    # Load the best checkpoint through the compatibility helper (it handles
    # `weights_only` across PL versions) and test that task directly, instead
    # of having Lightning re-load the checkpoint internally via
    # ckpt_path="best", whose weights_only handling is version-dependent.
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")
    best_task = load_task_from_checkpoint(type(task), best_path)

    # Evaluate best model on test set after training
    log.info("Starting testing.")
    trainer.test(model=best_task, datamodule=datamodule)

    # Store best model
    log.info("Store best model")
    torch.save(best_task, config.globals.model_path + ".task")

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at {os.path.abspath(config.globals.model_path)}")


@hydra.main(config_path="configs", config_name="predict", version_base="1.2")
def predict(config: DictConfig):
    log.info(f"Load data from `{config.data.datapath}`")
    dataset: ASEAtomsData = hydra.utils.instantiate(config.data)
    loader = AtomsLoader(dataset, batch_size=config.batch_size, num_workers=8)

    model = load_model("best_model")

    model_fn = CompiledPotential(model) if config.get("compile", False) else model

    class WrapperLM(LightningModule):
        def __init__(self, model, enable_grad=config.enable_grad):
            super().__init__()
            self.model = model
            self.enable_grad = enable_grad

        def forward(self, x):
            return model_fn(x)

        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            torch.set_grad_enabled(self.enable_grad)
            results = self(batch)
            results[properties.idx_m] = batch[properties.idx][batch[properties.idx_m]]
            results = {k: v.detach().cpu() for k, v in results.items()}
            return results

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            PredictionWriter(
                output_dir=config.outputdir,
                write_interval=config.write_interval,
                write_idx=config.write_idx_m,
            )
        ],
        default_root_dir=".",
        _convert_="partial",
    )
    trainer.predict(
        WrapperLM(model, config.enable_grad),
        dataloaders=loader,
        ckpt_path=config.ckpt_path,
    )


def convert():
    """
    Update the metadata of an ASE DB dataset in place, e.g. to make a dataset
    created with SchNetPack 1.0 loadable by the current version.

    - Converts old-style atom references (array + `atref_labels`) to a dict
      mapping each property to its per-element references.
    - Sets the distance unit (`--distunit`) and property units (`--propunit`).
    - Optionally prepends a dimension to the given properties in every row
      (`--expand_property_dims`), as needed for old FieldSchNet datasets.

    The database is modified in place, so back it up first.

    """
    parser = argparse.ArgumentParser(
        description="Set units of an ASE dataset, e.g. to convert from SchNetPack 1.0 to the new format."
    )
    parser.add_argument(
        "data_path",
        help="Path to ASE DB dataset",
    )
    parser.add_argument(
        "--distunit",
        help="Distance unit as string, corresponding to ASE units (e.g. `Ang`)",
    )
    parser.add_argument(
        "--propunit",
        help="Property units as string, corresponding "
        "to ASE units (e.g. `kcal/mol/Ang`), in the form: `property1:unit1,property2:unit2`",
    )
    parser.add_argument(
        "--expand_property_dims",
        default=[],
        nargs="+",
        help="Expanding the first dimension of the given property "
        "(required for example for old FieldSchNet datasets). "
        "Add property names here in the form 'property1 property2 property3'",
    )
    args = parser.parse_args()
    with connect(args.data_path) as db:
        meta = db.metadata
    print(meta)

    if "atomrefs" not in meta.keys():
        meta["atomrefs"] = {}
    elif "atref_labels" in meta.keys():
        old_atref = np.array(meta["atomrefs"])
        new_atomrefs = {}
        labels = meta["atref_labels"]
        if type(labels) is str:
            labels = [labels]
        for i, label in enumerate(labels):
            print(i, label, old_atref[:, i])
            new_atomrefs[label] = list(old_atref[:, i])
        meta["atomrefs"] = new_atomrefs
        del meta["atref_labels"]

    if args.distunit:
        if args.distunit == "A":
            raise ValueError(
                "The provided unit (A for Ampere) is not a valid distance unit according to the ASE unit"
                " definitions. You probably mean `Ang`/`Angstrom`. Please also check your property units!"
            )
        meta["_distance_unit"] = args.distunit

    if args.propunit:
        if "_property_unit_dict" not in meta.keys():
            meta["_property_unit_dict"] = {}

        for p in args.propunit.split(","):
            prop, unit = p.split(":")
            meta["_property_unit_dict"][prop] = unit

    with connect(args.data_path) as db:
        db.metadata = meta

    if args.expand_property_dims is not None and len(args.expand_property_dims) > 0:
        with connect(args.data_path) as db:
            for i in tqdm(range(len(db))):
                atoms_row = db.get(i + 1)
                data = {}
                for p, v in atoms_row.data.items():
                    if p in args.expand_property_dims:
                        data[p] = np.expand_dims(v, 0)
                    else:
                        data[p] = v
                db.update(i + 1, data=data)


def deploy():
    """
    Compile a trained potential with AOTInductor into a package for the LAMMPS pair
    style in ``interfaces/lammps`` (see
    :func:`~schnetpack.interfaces.lammps_interface.deploy_lammps`).

    The package runs only with the torch version and on the device it was compiled
    for, so run this on the machine LAMMPS runs on.
    """
    from schnetpack.interfaces.lammps_interface import deploy_lammps

    parser = argparse.ArgumentParser(
        description="Compile a SchNetPack potential for LAMMPS (pair_style schnetpack)."
    )
    parser.add_argument("model_path", help="Path to the trained model, e.g. best_model")
    parser.add_argument(
        "deployed_model_path", help="Where the package is written, ending in .pt2"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device LAMMPS runs the model on, e.g. cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Precision of the deployed model (default: float32)",
    )
    args = parser.parse_args()

    model = load_model(args.model_path, device=args.device)
    deploy_lammps(
        model,
        args.deployed_model_path,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    print(f"stored deployed model at {args.deployed_model_path}.")

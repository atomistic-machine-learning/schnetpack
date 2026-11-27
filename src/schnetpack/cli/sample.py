"""spk sample: sample structures with a generative model (``configs/sample.yaml``)."""

import logging
import os
import shutil
import tempfile
import time

import hydra
import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf

from schnetpack import properties
from schnetpack.cli import _run_config
from schnetpack.data import ASEAtomsData
from schnetpack.model_format import load_hyperparameters, load_model

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="sample", version_base="1.2")
def main(config: DictConfig):
    """spk sample: sample structures with the generative model ``model`` (a model
    folder or a spk train run folder) and write one ASE db per number of steps to
    ``out``. The compositions are those of ``db``, or else of the run's test split.
    The settings are the model's ``sample:`` block over the ``sample:`` block of
    configs/sample.yaml, under the command line's ``sample.*`` keys.

    ``sample.condition`` holds values written into every structure, by batch key: the
    condition of a conditional model, the target of a guidance term. A conditional
    model whose condition is not given runs without it (NaN). ``sample.cfg_scale``
    is the scale of the classifier-free guidance, and an update's ``guidance`` lists
    the terms added to the field it steps on."""
    from schnetpack.dynamics import DatasetStructures, FieldCalculator, Sampler
    from schnetpack.nn import ConditionEmbedding

    run = None
    if os.path.exists(os.path.join(config.model_dir, "config.yaml")):
        run = _run_config(config.model_dir, config.data_dir)
    blocks = load_hyperparameters(config.model_dir)
    if "generative" not in blocks:
        raise ValueError(
            f"{config.model_dir} has no generative: block: not a generative model"
        )
    sample = _settings(config.sample, blocks.get("sample", {}))

    if config.db is not None:
        dataset = ASEAtomsData(config.db, load_properties=[], transforms=[])
    elif run is not None:
        test_idx = np.load(run.data.split_file)["test_idx"].tolist()
        dataset = hydra.utils.instantiate(
            run.data.dataset, load_properties=[], transforms=[]
        ).subset(test_idx)
    else:
        raise ValueError(
            f"{config.model_dir} is a model folder: give the compositions as db=<ASE db>"
        )
    if sample.seed is None and run is not None:
        sample.seed = run.get("seed")
    if sample.seed is None:
        raise ValueError("no seed: give sample.seed=<seed>")
    out = config.out
    if out is None:
        out = (
            os.path.join(config.model_dir, "samples") if run is not None else "samples"
        )

    # each variable built once: the updates and the prior draws must hold the same
    # objects
    generative = hydra.utils.instantiate(blocks["generative"], _convert_="all")
    updates = [
        hydra.utils.instantiate(update, _convert_="all")(generative[name])
        for name, update in sample.updates.items()
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for update in updates:
        for term in update.guidance:
            term.to(device)
    model_path = os.path.abspath(config.model_dir)
    model = load_model(model_path, device=device).eval()
    condition_keys = [
        module.property_key
        for module in model.modules()
        if isinstance(module, ConditionEmbedding)
    ]
    if sample.cfg_scale != 0.0 and not condition_keys:
        raise ValueError(
            f"sample.cfg_scale={sample.cfg_scale} needs a conditional model; "
            f"{config.model_dir} has no ConditionEmbedding"
        )
    # a condition that is not given is NaN: the model's unconditional branch
    condition = {key: float("nan") for key in condition_keys}
    condition.update(sample.condition or {})

    def write_condition(batch):
        n_structures = batch[properties.n_atoms].shape[0]
        for key, value in condition.items():
            batch[key] = torch.full((n_structures,), float(value))
        return batch

    sampler = Sampler(
        FieldCalculator(
            model,
            device=device,
            compile=sample.compile,
            cfg_scale=sample.cfg_scale,
            condition_keys=condition_keys,
        ),
        updates=updates,
        start=[generative[name] for name in sample.updates],
        grid=hydra.utils.instantiate(sample.grid),
        structures=DatasetStructures(
            dataset,
            transforms=[
                *hydra.utils.instantiate(sample.structure_transforms),
                write_condition,
            ],
        ),
    )
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    os.makedirs(out, exist_ok=True)

    for n_steps in sample.n_steps:
        torch.manual_seed(sample.seed)
        results, seconds = [], []
        start = time.perf_counter()
        for batch in sampler.sample_batches(
            sample.n_samples, n_steps, sample.batch_size
        ):
            n_atoms = batch[properties.n_atoms].cpu()  # waits for the run
            results.append((n_atoms, batch[properties.Z].cpu(), batch[properties.R]))
            seconds.append(time.perf_counter() - start - sum(seconds))
        total = sum(seconds)
        log.info(
            f"{sample.n_samples} structures, {n_steps} steps: {total:.1f} s "
            f"(first batch {seconds[0]:.2f} s"
            f"{', with the compile' if sample.compile else ''}), {gpu}"
        )

        # written to a local file and moved: sqlite on a shared file system is slow
        name = f"steps{n_steps}_seed{sample.seed}.db"
        local = os.path.join(tempfile.mkdtemp(), name)
        with connect(local) as db:
            for n_atoms, zs, rs in results:
                for z, r in zip(
                    zs.split(n_atoms.tolist()),
                    rs.cpu().double().split(n_atoms.tolist()),
                ):
                    db.write(Atoms(numbers=z.numpy(), positions=r.numpy()))
            db.metadata = {
                "model": model_path,
                "n_samples": sample.n_samples,
                "n_steps": n_steps,
                "seed": sample.seed,
                "batch_size": sample.batch_size,
                "compile": sample.compile,
                "seconds": total,
                "seconds_first_batch": seconds[0],
                "gpu": gpu,
                "sample": OmegaConf.to_container(sample, resolve=True),
                "variables": {
                    name: blocks["generative"][name] for name in sample.updates
                },
            }
        path = os.path.join(out, name)
        shutil.move(local, path)
        log.info(f"Samples stored at {path}")


def _settings(defaults: DictConfig, saved: dict) -> DictConfig:
    """The model's saved ``sample:`` block over ``defaults`` (the command's
    ``sample:`` block), and the ``sample.*`` keys given on the command line over
    both."""
    settings = OmegaConf.merge(OmegaConf.to_container(defaults), saved)
    task = HydraConfig.get().overrides.task
    for override in OverridesParser.create().parse_overrides(task):
        key = override.key_or_group.removeprefix("sample.")
        if key != override.key_or_group:
            value = OmegaConf.select(defaults, key)
            OmegaConf.update(settings, key, value, merge=False)
    return settings

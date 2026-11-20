"""spksample: sample molecules from a finished generative spktrain run."""

import argparse
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
from omegaconf import OmegaConf

from schnetpack import properties
from schnetpack.cli import _run_config
from schnetpack.model_format import load_model

log = logging.getLogger(__name__)


def main():
    """spksample: sample molecules from a generative spktrain run's best_model, with
    the compositions of its test split, and write one ASE db per number of steps to
    ``sample.out_dir``. The settings are the ``sample:`` block of configs/sample.yaml,
    under the run's own ``sample:`` block and the command line's overrides."""
    from schnetpack.dynamics import DatasetStructures, FieldCalculator, Sampler

    parser = argparse.ArgumentParser(
        description="Sample molecules from a generative spktrain run."
    )
    parser.add_argument("run_dir", help="the run folder spktrain wrote")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="key=value changes, e.g. sample.n_samples=1000 run.data_dir=/data",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    defaults = OmegaConf.load(
        os.path.join(os.path.dirname(__file__), "..", "configs", "sample.yaml")
    )
    config = _run_config(args.run_dir, args.overrides, defaults)
    sample = config.sample
    if sample.seed is None:
        raise ValueError("the run has no seed: give sample.seed=<seed>")

    # each variable built once: the updates and the prior draws must hold the same
    # objects
    generative = hydra.utils.instantiate(config.generative, _convert_="all")
    updates = [
        hydra.utils.instantiate(update)(generative[name])
        for name, update in sample.updates.items()
    ]
    test_idx = np.load(config.data.split_file)["test_idx"].tolist()
    dataset = hydra.utils.instantiate(
        config.data.dataset, load_properties=[], transforms=[]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(os.path.abspath(args.run_dir), "best_model")
    sampler = Sampler(
        FieldCalculator(
            load_model(model_path, device=device).eval(),
            device=device,
            compile=sample.compile,
        ),
        updates=updates,
        start=[generative[name] for name in sample.updates],
        grid=hydra.utils.instantiate(sample.grid),
        structures=DatasetStructures(
            dataset.subset(test_idx),
            transforms=hydra.utils.instantiate(sample.structure_transforms),
        ),
    )
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    os.makedirs(sample.out_dir, exist_ok=True)

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
            f"{sample.n_samples} molecules, {n_steps} steps: {total:.1f} s "
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
                    name: OmegaConf.to_container(config.generative[name], resolve=True)
                    for name in sample.updates
                },
            }
        out = os.path.join(sample.out_dir, name)
        shutil.move(local, out)
        log.info(f"Samples stored at {out}")

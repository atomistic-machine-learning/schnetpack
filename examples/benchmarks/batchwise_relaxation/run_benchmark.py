"""Time the batch-wise ``dynamics.LBFGS`` against sequential ase ``LBFGS``.

Both optimizers relax the same batches of noisy structures over a range of batch sizes.
The relaxed structures and the wall times are written to the output directory, which
``plot_benchmark.py`` reads to plot the relaxation time against the batch size.

Only the relaxation itself is timed: building the calculator, converting the structures
and constructing the relaxer all happen before the clock starts. Every batch size is
repeated with a different noise seed per repeat, and both optimizers relax the identical
batch in each repeat (paired samples).
"""

import argparse
import json
import os
import socket
import time
from copy import deepcopy

import numpy as np
import torch
from ase.io import read, write
from ase.optimize import LBFGS

import schnetpack as spk
from schnetpack.dynamics import ForceCalculator
from schnetpack.interfaces.ase_interface import (
    SpkCalculator,
    atoms_to_batch,
    batch_to_atoms,
)
from schnetpack.utils.compatibility import load_model

TESTDATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "testdata"
)
METHODS = ("batch-wise", "sequential")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        default=os.path.join(TESTDATA, "md_ethanol.model"),
        help="SchNetPack model that predicts energies and forces",
    )
    parser.add_argument(
        "--structures",
        default=os.path.join(TESTDATA, "ethanol_conformers.xyz"),
        help="starting structures, cycled through to fill a batch",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--fmax", type=float, default=0.001, help="convergence criterion in eV/Ang"
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="standard deviation of the Gaussian noise on the positions, in Ang",
    )
    parser.add_argument("--cutoff-skin", type=float, default=0.3)
    parser.add_argument("--energy-unit", default="kcal/mol")
    parser.add_argument("--position-unit", default="Ang")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default="./batchwise_relaxation_benchmark")
    return parser.parse_args()


def make_structures(conformers, n_structures, noise, seed=None):
    """``n_structures`` starting geometries, cycling through the conformers.

    Every copy gets Gaussian noise on its atomic positions, so the structures of a batch
    are similar but not identical. Pass a ``seed`` to get a reproducible set.
    """
    rng = np.random.default_rng(seed)
    structures = []
    for idx in range(n_structures):
        noisy_atoms = conformers[idx % len(conformers)].copy()
        noisy_atoms.positions += rng.normal(
            scale=noise, size=noisy_atoms.positions.shape
        )
        structures.append(noisy_atoms)
    return structures


class Benchmark:
    """Builds both optimizers from the same settings and times their relaxations."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        model = load_model(args.model, device=self.device)
        self.cutoff = model.representation.cutoff.item()

    def batchwise_relaxer(self):
        """An ``LBFGS`` that relaxes a whole batch in a single model call per step."""
        calculator = ForceCalculator(
            self.args.model,
            neighbor_list=spk.dynamics.BatchNeighborList(
                neighbor_list=spk.transform.MatScipyNeighborList(cutoff=self.cutoff),
                cutoff_skin=self.args.cutoff_skin,
            ),
            device=self.device,
            dtype=torch.float32,
            energy_unit=self.args.energy_unit,
            position_unit=self.args.position_unit,
        )
        return spk.dynamics.LBFGS(calculator, fmax=self.args.fmax)

    def ase_calculator(self):
        """A plain ase calculator, which evaluates one structure per call."""
        return SpkCalculator(
            model=self.args.model,
            neighbor_list=spk.transform.MatScipyNeighborList(cutoff=self.cutoff),
            device=self.device,
            energy_unit=self.args.energy_unit,
            position_unit=self.args.position_unit,
        )

    def time_batchwise(self, structures):
        """Wall time of relaxing the whole batch in parallel. The setup is not timed."""
        relaxer = self.batchwise_relaxer()
        inputs = relaxer.calculator.prepare(
            atoms_to_batch(structures, device=self.device)
        )

        start = time.perf_counter()
        relaxed = relaxer.run(inputs, self.args.max_steps)
        elapsed = time.perf_counter() - start

        # a run that is fast because it never converged is not a faster run
        max_force = relaxer.calculator(relaxed)["forces"].norm(dim=-1).max()
        assert max_force < self.args.fmax, "batch-wise run hit the step limit"
        return elapsed, batch_to_atoms(relaxed)

    def time_sequential(self, structures):
        """Wall time of relaxing the same structures one after another."""
        calculator = self.ase_calculator()
        structures = deepcopy(structures)  # LBFGS relaxes the Atoms objects in place

        start = time.perf_counter()
        steps = []
        for structure in structures:
            structure.calc = calculator
            lbfgs = LBFGS(structure, logfile=None)
            lbfgs.run(fmax=self.args.fmax, steps=self.args.max_steps)
            steps.append(lbfgs.nsteps)
        elapsed = time.perf_counter() - start

        assert max(steps) < self.args.max_steps, "a sequential run hit the step limit"
        # the relaxed positions only, without the calculator attached
        for structure in structures:
            structure.calc = None
        return elapsed, structures


def main():
    args = parse_args()
    structures_dir = os.path.join(args.output, "structures")
    os.makedirs(structures_dir, exist_ok=True)

    conformers = read(args.structures, index=":")
    benchmark = Benchmark(args)
    timers = {
        "batch-wise": benchmark.time_batchwise,
        "sequential": benchmark.time_sequential,
    }

    # The first relaxation pays for lazy torch initialisation, which would otherwise
    # land entirely on the smallest batch size and fake a slow start.
    warmup = make_structures(conformers, 2, args.noise, seed=999)
    for timer in timers.values():
        timer(warmup)

    records = []
    for n_structures in args.batch_sizes:
        times = {method: [] for method in METHODS}
        for repeat in range(args.repeats):
            # both optimizers relax the identical batch, so the repeats are paired
            structures = make_structures(conformers, n_structures, args.noise, repeat)
            for method in METHODS:
                elapsed, relaxed = timers[method](structures)
                structures_file = os.path.join(
                    "structures", f"{method}_n{n_structures:03d}_r{repeat:02d}.xyz"
                )
                write(os.path.join(args.output, structures_file), relaxed)
                records.append(
                    dict(
                        method=method,
                        batch_size=n_structures,
                        repeat=repeat,
                        seed=repeat,
                        time_s=elapsed,
                        structures_file=structures_file,
                    )
                )
                times[method].append(elapsed)

        print(
            f"batch size {n_structures:>3}: "
            f"batch-wise {np.mean(times['batch-wise']):6.2f} s, "
            f"sequential {np.mean(times['sequential']):6.2f} s",
            flush=True,
        )

    meta = dict(
        model=os.path.abspath(args.model),
        structures=os.path.abspath(args.structures),
        device=benchmark.device.type,
        # the hardware, so runs on different nodes can be told apart
        gpu=(
            torch.cuda.get_device_name(benchmark.device)
            if benchmark.device.type == "cuda"
            else None
        ),
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        fmax=args.fmax,
        max_steps=args.max_steps,
        noise=args.noise,
        cutoff_skin=args.cutoff_skin,
        torch_version=torch.__version__,
        schnetpack_version=spk.__version__,
    )
    timings_file = os.path.join(args.output, "timings.json")
    with open(timings_file, "w") as f:
        json.dump(dict(meta=meta, records=records), f, indent=2)
    print(f"results written to {timings_file}")


if __name__ == "__main__":
    main()

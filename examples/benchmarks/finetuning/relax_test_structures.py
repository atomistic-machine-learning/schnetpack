"""Relax rMD17 ethanol test structures with the batch-wise ``dynamics.LBFGS`` on three
MACE models, and score each minimum by its RMSD to the closest DFT conformer.

The models are those of the scratch vs. fine-tuning comparison trained on the same
structures (``lc_{finetune,scratch}_n<N>_r<REP>``), plus MACE-OFF23 zero shot as the
reference. Its E0 correction shifts the energy only, so the geometries do not depend on
it and the bare archive is relaxed.

The starting structures are drawn from the test set of the replicate's published split,
leaving out every structure in the training pool (``known``) of any published split:
no model of the comparison, of any replicate, has trained or validated on them. Each
model relaxes all of them as one batch, in one model call per step.

The reference conformers are relaxed at PBE/def2-SVP, the level of theory of rMD17: the
fine-tuned and the scratch model are trained on it, MACE-OFF23 (wB97M-D3BJ/def2-TZVPPD)
is not, so its minima are expected to sit off the reference by the change of level.

The RMSD is minimized over the reference conformers, the rigid-body motion and the
permutations of equivalent hydrogens, and is a true RMSD (divided by the square root of
the number of atoms).

    python relax_test_structures.py [--runs RUNS] [--datapath DB] [--mace-off PT2]
"""

import argparse
import json
import os
import socket
import time
from itertools import permutations, product

import numpy as np
import torch
from ase import Atoms
from ase.build import minimize_rotation_and_translation
from ase.io import read, write

import schnetpack as spk
from schnetpack import properties
from schnetpack.data import ASEAtomsData
from schnetpack.dynamics import ForceCalculator
from schnetpack.dynamics.hooks import Hook
from schnetpack.interfaces.ase_interface import atoms_to_batch, batch_to_atoms
from schnetpack.model import MaceModel, load_mace
from schnetpack.utils.compatibility import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("finetune", "scratch", "zeroshot")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs",
        default="/home/Jonas/schnetpack_debugging/finetuning/runs",
        help="runs directory of the scratch vs. fine-tuning comparison",
    )
    parser.add_argument(
        "--datapath",
        default="/home/Jonas/datasets/schnetpack/rmd17/ethanol.db",
        help="rMD17 ethanol database, with the published splits in its metadata",
    )
    parser.add_argument(
        "--mace-off",
        default="/home/Jonas/schnetpack_debugging/teacher_models/"
        "MACE-OFF23_medium_mace-0.3.16.pt2",
        help="MACE-OFF23 archive (.pt2) for the zero-shot model",
    )
    parser.add_argument(
        "--conformers",
        default=os.path.join(HERE, "ethanol_conformers_dft.xyz"),
        help="reference conformers relaxed at PBE/def2-SVP (rMD17's level), in the "
        "atom order of rMD17",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--num-train", type=int, default=100)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--n-structures", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fmax", type=float, default=0.001, help="convergence criterion in eV/Ang"
    )
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--cutoff-skin", type=float, default=0.3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory; defaults to relaxation_n<N>_r<REP> in the runs directory",
    )
    return parser.parse_args()


def run_dir(args, arm):
    return os.path.join(args.runs, f"lc_{arm}_n{args.num_train}_r{args.rep}")


def test_structures(args, models):
    """The dataset indices and structures to relax.

    Drawn from the test set of split ``rep``, minus the training pool of every
    published split, so that no run of the comparison has seen them.
    """
    dataset = ASEAtomsData(args.datapath, load_properties=[])
    splits = dataset.metadata["splits"]
    seen = set(np.concatenate(splits["known"]).tolist())
    pool = np.array(sorted(set(splits["test"][args.rep]) - seen))

    # the run's own split file is what it trained on: check against it, too
    for arm in ("finetune", "scratch"):
        if arm not in models:
            continue
        split = np.load(os.path.join(run_dir(args, arm), "split.npz"))
        overlap = np.intersect1d(
            pool, np.concatenate([split["train_idx"], split["val_idx"]])
        )
        assert overlap.size == 0, f"{arm} trained on test structures {overlap}"

    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(pool, size=args.n_structures, replace=False))
    structures = []
    for idx in indices:
        sample = dataset[int(idx)]
        structures.append(
            Atoms(
                numbers=sample[properties.Z].numpy(),
                positions=sample[properties.R].numpy(),
            )
        )
    return indices, structures, len(pool)


def load(args, name, device):
    """The model ``name`` as a ``MaceModel``, float32 as it was trained.

    The model is cast here and not by the calculator: ``MaceModel`` casts only MACE
    itself, so its per-element reference energies stay in float64.
    """
    if name == "zeroshot":
        model = MaceModel(load_mace(args.mace_off), dtype="float32")
        path = args.mace_off
    else:
        path = os.path.join(run_dir(args, name), "best_model")
        model = load_model(path, device=device)
    return model.to(device), os.path.abspath(path)


class StepCounter(Hook):
    """The step at which each structure last moved: its number of steps.

    The optimizer holds converged structures in place, so this costs no model call.
    """

    def __init__(self):
        self.steps = None
        self._positions = None

    def before_step(self, batch, step, n_steps):
        self._positions = batch[properties.R]
        if self.steps is None:
            self.steps = torch.zeros(
                int(batch[properties.idx_m].max()) + 1, dtype=torch.long
            )
        return batch

    def after_step(self, batch, step, n_steps):
        moved = (batch[properties.R] != self._positions).any(dim=-1)
        moved_structures = batch[properties.idx_m][moved].unique().cpu()
        self.steps[moved_structures] = step
        return batch


def equivalent_hydrogens(reference):
    """Groups of hydrogens bound to the same heavy atom, which may swap labels."""
    numbers = reference.get_atomic_numbers()
    heavy = np.flatnonzero(numbers != 1)
    groups = {}
    for h in np.flatnonzero(numbers == 1):
        distances = np.linalg.norm(
            reference.positions[heavy] - reference.positions[h], axis=1
        )
        groups.setdefault(heavy[distances.argmin()], []).append(h)
    return [group for group in groups.values() if len(group) > 1]


def relabellings(reference):
    """Every atom order that leaves the molecule unchanged, by swapping equivalent
    hydrogens: 2! * 3! = 12 for ethanol."""
    groups = equivalent_hydrogens(reference)
    orders = []
    for choice in product(*(permutations(group) for group in groups)):
        order = np.arange(len(reference))
        for group, permuted in zip(groups, choice):
            order[group] = permuted
        orders.append(order)
    return orders


def closest_conformer_rmsd(atoms, conformers, orders):
    """``(rmsd, index)`` of the closest conformer in Angstrom.

    Minimized over the conformers, the relabellings of equivalent hydrogens and the
    rigid-body motion. ``minimize_rotation_and_translation`` applies a proper rotation,
    so the two gauche conformers, mirror images of each other, stay distinct.
    """
    best = (np.inf, None)
    for order in orders:
        mobile = atoms.copy()
        mobile.positions = atoms.positions[order]
        for idx, conformer in enumerate(conformers):
            minimize_rotation_and_translation(conformer, mobile)
            rmsd = np.sqrt(
                ((mobile.positions - conformer.positions) ** 2).sum(-1).mean()
            )
            best = min(best, (rmsd, idx), key=lambda x: x[0])
    return float(best[0]), best[1]


def relax(model, structures, args, device):
    """Relax all structures as one batch; returns the relaxed structures, the steps
    and the final maximum force per structure, and the wall time."""
    calculator = ForceCalculator(
        model,
        neighbor_list=spk.dynamics.BatchNeighborList(
            neighbor_list=spk.transform.MatScipyNeighborList(cutoff=model.r_max),
            cutoff_skin=args.cutoff_skin,
        ),
        device=device,
    )
    counter = StepCounter()
    relaxer = spk.dynamics.LBFGS(calculator, fmax=args.fmax, hooks=[counter])
    batch = atoms_to_batch(structures, device=device)

    # the first model call pays for lazy initialisation, which is not the relaxation
    relaxer.run(batch, 1)
    counter.steps = None
    batch = atoms_to_batch(structures, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    relaxed = relaxer.run(batch, args.max_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    forces = relaxer.calculator(relaxed)["forces"].norm(dim=-1).detach()
    max_force = forces.new_zeros(len(structures)).index_reduce(
        0, relaxed[properties.idx_m], forces, "amax", include_self=False
    )
    return (
        batch_to_atoms(relaxed),
        counter.steps.numpy(),
        max_force.cpu().numpy(),
        elapsed,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    output = args.output or os.path.join(
        args.runs, f"relaxation_n{args.num_train}_r{args.rep}"
    )
    os.makedirs(os.path.join(output, "structures"), exist_ok=True)

    conformers = read(args.conformers, index=":")
    orders = relabellings(conformers[0])
    indices, structures, pool_size = test_structures(args, args.models)
    assert all((s.numbers == conformers[0].numbers).all() for s in structures), (
        "the conformers are not in the atom order of the dataset"
    )
    write(os.path.join(output, "structures", "start.xyz"), structures)
    start = [closest_conformer_rmsd(s, conformers, orders) for s in structures]

    rows = [
        dict(
            model="start",
            dataset_idx=int(idx),
            rmsd=rmsd,
            conformer=conformer,
            steps=0,
            max_force=np.nan,
            converged=False,
        )
        for idx, (rmsd, conformer) in zip(indices, start)
    ]
    meta = dict(
        runs=os.path.abspath(args.runs),
        datapath=os.path.abspath(args.datapath),
        conformers=os.path.abspath(args.conformers),
        num_train=args.num_train,
        rep=args.rep,
        seed=args.seed,
        pool_size=pool_size,
        dataset_idx=indices.tolist(),
        fmax=args.fmax,
        max_steps=args.max_steps,
        cutoff_skin=args.cutoff_skin,
        device=device.type,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        torch_version=torch.__version__,
        schnetpack_version=spk.__version__,
        models={},
    )

    for name in args.models:
        model, path = load(args, name, device)
        relaxed, steps, max_force, elapsed = relax(model, structures, args, device)
        write(os.path.join(output, "structures", f"{name}.xyz"), relaxed)
        meta["models"][name] = dict(path=path, time_s=elapsed)
        for idx, atoms, n, force in zip(indices, relaxed, steps, max_force):
            rmsd, conformer = closest_conformer_rmsd(atoms, conformers, orders)
            rows.append(
                dict(
                    model=name,
                    dataset_idx=int(idx),
                    rmsd=rmsd,
                    conformer=conformer,
                    steps=int(n),
                    max_force=float(force),
                    converged=bool(force < args.fmax),
                )
            )
        print(
            f"{name}: relaxed {len(structures)} structures in {elapsed:.1f} s",
            flush=True,
        )

    csv = os.path.join(output, "relaxation.csv")
    with open(csv, "w") as fh:
        fh.write(",".join(rows[0]) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row.values()) + "\n")
    with open(os.path.join(output, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(
        f"\n{len(structures)} test structures of split {args.rep} "
        f"(seed {args.seed}, from {pool_size} in no training pool)"
    )
    print(
        f"{'model':>9} {'RMSD median':>12} {'mean':>8} {'max':>8} "
        f"{'conformers':>11} {'steps':>6} {'converged':>10}"
    )
    print(f"{'':>9} {'[A]':>12} {'[A]':>8} {'[A]':>8}")
    for name in ["start", *args.models]:
        sel = [r for r in rows if r["model"] == name]
        rmsd = np.array([r["rmsd"] for r in sel])
        counts = np.bincount([r["conformer"] for r in sel], minlength=len(conformers))
        steps = np.median([r["steps"] for r in sel])
        converged = (
            f"{sum(r['converged'] for r in sel)}/{len(sel)}" if name != "start" else "-"
        )
        print(
            f"{name:>9} {np.median(rmsd):12.4f} {rmsd.mean():8.4f} {rmsd.max():8.4f} "
            f"{'/'.join(map(str, counts)):>11} {steps:6.0f} {converged:>10}"
        )
    print(f"\nresults written to {csv}")


if __name__ == "__main__":
    main()

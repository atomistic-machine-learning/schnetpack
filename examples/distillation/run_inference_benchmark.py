"""Time the inference of a distillation teacher against its students.

Every model predicts energies and forces for the same batches of structures over a range
of batch sizes. The wall times are written to ``timings.json`` in the output directory.

Only the model call is timed. The batch, with its neighbor list at the model's own
cutoff, is built before the clock starts; how long that build takes is recorded
separately, since a student with a smaller cutoff saves there as well. Within each batch
size the models take turns, so that drifts of the node (clocks, temperature) hit all of
them alike. A model that runs out of GPU memory is recorded as such and skips the larger
batch sizes.
"""

import argparse
import json
import os
import socket
import time

import numpy as np
import torch
from ase.io import read

import schnetpack as spk
from schnetpack.dynamics import ForceCalculator
from schnetpack.interfaces.ase_interface import AtomsConverter
from schnetpack.model import MaceModel
from schnetpack.model.mace import load_mace
from schnetpack.utils.compatibility import load_model

TESTDATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "testdata"
)
MODEL_FORMATS = ("mace", "schnetpack")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--teacher",
        required=True,
        help="the teacher: a MACE TorchScript archive or a SchNetPack model",
    )
    parser.add_argument(
        "--teacher-format",
        choices=MODEL_FORMATS,
        default="mace",
        help="how the teacher is loaded",
    )
    parser.add_argument(
        "--students",
        nargs="+",
        default=[],
        metavar="[LABEL=]PATH",
        help="SchNetPack models distilled from the teacher, e.g. the best_model of a "
        "spktrain run; an optional label names them in the results",
    )
    parser.add_argument(
        "--structures",
        default=os.path.join(TESTDATA, "ethanol_conformers.xyz"),
        help="structures in any format ase reads, cycled through to fill a batch",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--repeats", type=int, default=20, help="timed model calls per batch size"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="untimed model calls per batch size"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="standard deviation of the Gaussian noise on the positions, in Ang",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default="./inference_benchmark")
    return parser.parse_args()


def make_structures(conformers, n_structures, noise, seed=None):
    """``n_structures`` geometries, cycling through the conformers.

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


def parse_students(entries):
    """``{label: path}`` from ``[LABEL=]PATH`` entries."""
    students = {}
    for idx, entry in enumerate(entries):
        label, sep, path = entry.partition("=")
        if not sep:
            label, path = f"student_{idx}", entry
        if label in students or label == "teacher":
            raise ValueError(f"the label {label!r} is used twice")
        students[label] = path
    return students


class TimedModel:
    """A model loaded for inference, with the cutoff its neighbor list needs."""

    def __init__(self, label, path, model_format, device, dtype):
        self.label = label
        self.path = os.path.abspath(path)
        self.model_format = model_format
        if model_format == "mace":
            # the core alone is cast, so the E0s stay in float64
            model = MaceModel(load_mace(path), dtype=dtype)
            self.cutoff = model.r_max
        else:
            model = load_model(path, device="cpu").to(dtype)
            self.cutoff = model.representation.cutoff.item()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.n_parameters = sum(p.numel() for p in model.parameters())
        self.calculator = ForceCalculator(model, device=device)
        self.converter = AtomsConverter(
            neighbor_list=spk.transform.MatScipyNeighborList(cutoff=self.cutoff),
            device=device,
            dtype=dtype,
        )

    def meta(self):
        return dict(
            path=self.path,
            format=self.model_format,
            cutoff=self.cutoff,
            n_parameters=self.n_parameters,
        )


def is_out_of_memory(err):
    """Whether ``err`` is CUDA running out of memory. Inside a TorchScript model, such
    as a MACE archive, it is raised as a plain ``RuntimeError``."""
    return isinstance(err, torch.OutOfMemoryError) or "CUDA out of memory" in str(err)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_mb(device, *, peak=False, reserved=False):
    """The GPU memory torch has allocated, or reserved in its caching allocator, now or
    at its peak, in MiB; None on the cpu."""
    if device.type != "cuda":
        return None
    if reserved:
        memory = torch.cuda.max_memory_reserved if peak else torch.cuda.memory_reserved
    else:
        memory = (
            torch.cuda.max_memory_allocated if peak else torch.cuda.memory_allocated
        )
    return memory(device) / 2**20


def time_model(model, structures, args, device):
    """Wall times of ``args.repeats`` energy and force calls on one batch, after
    ``args.warmup`` untimed ones, together with the time the batch took to build and
    the GPU memory of the calls.

    Every model stays on the device for the whole run, so all memory figures include
    the weights of all of them: ``resident_memory_mb`` is what is allocated before the
    batch is built, ``allocated_memory_mb`` what is allocated with the batch in place,
    before the calls. ``peak_memory_mb`` is the most allocated during the timed calls,
    and ``peak_reserved_memory_mb`` the most the caching allocator held then, which is
    closer to what ``nvidia-smi`` shows.
    """
    if device.type == "cuda":
        # the cache still holds the blocks of the previous model's batch
        torch.cuda.empty_cache()
    resident_memory_mb = memory_mb(device)
    start = time.perf_counter()
    batch = model.converter(structures)
    synchronize(device)
    neighbor_list_s = time.perf_counter() - start
    allocated_memory_mb = memory_mb(device)

    for _ in range(args.warmup):
        model.calculator(batch)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        outputs = model.calculator(batch)
        synchronize(device)
        times.append(time.perf_counter() - start)

    # a call that is fast because it returned nonsense is not a faster call
    assert torch.isfinite(outputs["forces"]).all(), f"{model.label}: non-finite forces"
    return dict(
        times_s=times,
        neighbor_list_s=neighbor_list_s,
        resident_memory_mb=resident_memory_mb,
        allocated_memory_mb=allocated_memory_mb,
        peak_memory_mb=memory_mb(device, peak=True),
        peak_reserved_memory_mb=memory_mb(device, peak=True, reserved=True),
    )


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    models = [
        TimedModel("teacher", args.teacher, args.teacher_format, device, dtype)
    ] + [
        TimedModel(label, path, "schnetpack", device, dtype)
        for label, path in parse_students(args.students).items()
    ]
    conformers = read(args.structures, index=":")

    meta = dict(
        models={model.label: model.meta() for model in models},
        structures=os.path.abspath(args.structures),
        device=device.type,
        # the hardware, so runs on different nodes can be told apart
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        hostname=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_partition=os.environ.get("SLURM_JOB_PARTITION"),
        cpu_threads=torch.get_num_threads(),
        batch_sizes=args.batch_sizes,
        repeats=args.repeats,
        warmup=args.warmup,
        noise=args.noise,
        dtype=args.dtype,
        torch_version=torch.__version__,
        schnetpack_version=spk.__version__,
    )
    timings_file = os.path.join(args.output, "timings.json")

    records = []
    out_of_memory = set()
    for n_structures in args.batch_sizes:
        structures = make_structures(conformers, n_structures, args.noise, seed=0)
        n_atoms = sum(len(structure) for structure in structures)
        summary = []
        for model in models:
            if model.label in out_of_memory:
                continue
            try:
                result = time_model(model, structures, args, device)
            except RuntimeError as err:
                if not is_out_of_memory(err):
                    raise
                result = None
            if result is None:
                # outside the except block, whose traceback holds the failed batch
                torch.cuda.empty_cache()
                out_of_memory.add(model.label)
                records.append(
                    dict(model=model.label, batch_size=n_structures, oom=True)
                )
                summary.append(f"{model.label} out of memory")
                continue
            records.append(
                dict(
                    model=model.label,
                    batch_size=n_structures,
                    n_atoms=n_atoms,
                    oom=False,
                    **result,
                )
            )
            median = np.median(result["times_s"])
            summary.append(
                f"{model.label} {1e3 * median:8.2f} ms "
                f"({n_structures / median:9.1f} structures/s)"
            )

        print(f"batch size {n_structures:>5}: " + ", ".join(summary), flush=True)
        # after every batch size, so a run that dies at a large batch keeps the rest
        with open(timings_file, "w") as f:
            json.dump(dict(meta=meta, records=records), f, indent=2)

    print(f"results written to {timings_file}")


if __name__ == "__main__":
    main()

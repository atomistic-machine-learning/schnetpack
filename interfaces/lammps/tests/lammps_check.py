#!/usr/bin/env python
"""Check pair_style schnetpack in LAMMPS against SchNetPack in Python.

Run it where LAMMPS with pair_style schnetpack and SchNetPack with the same torch are
installed, e.g. in a container (the CI job in .github/workflows/lammps.yml runs it too):

    apptainer exec schnetpack-lammps.sif python lammps_check.py
    apptainer exec --nv schnetpack-lammps.sif python lammps_check.py --device cuda

With the plugin build, point LAMMPS_PLUGIN_PATH at the directory of
schnetpackplugin.so.

It builds a small random PaiNN potential, deploys it with spkdeploy, writes a rattled
SiC crystal and runs LAMMPS:

1. one force evaluation with domain decomposition on each rank count of --ranks and
   with no_domain_decomposition on one rank, each compared to the model in Python:
   energy, per-atom energies, forces and virial (pressure without kinetic part);
2. 50 NVE steps in each of those setups, compared to each other, which covers
   reneighboring and atoms moving between ranks; the time per step is reported;
3. no_domain_decomposition on two ranks must fail;
4. 20 NPT steps must run and give finite results.

The model is random, so the numbers are not physical; only their agreement counts.
Exits with 1 if a check fails. Inputs, logs and dumps are kept in --workdir.
"""

import argparse
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import ase.build
import ase.io
import numpy as np
import torch

import schnetpack as spk
from schnetpack import properties
from schnetpack.interfaces.lammps_interface import LammpsPotential

CUTOFF = 3.5
# LAMMPS atom types in order, with their atomic numbers
TYPES = {"C": 6, "Si": 14}
# metal units: eV / Angstrom^3 -> bar, as LAMMPS' force->nktv2p
NKTV2P = 1.6021765e6
# relative tolerance of the static checks, and absolute tolerance (Angstrom) of the
# positions after 50 NVE steps, which grow chaotically from round-off
TOLERANCES = {"float64": (1e-8, 1e-6), "float32": (1e-4, 1e-3)}

HEADER = """units metal
atom_style atomic
boundary p p p
read_data data.lmp
pair_style schnetpack{style}
pair_coeff * * model.pt2 {types}
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
compute pea all pe/atom
compute pesum all reduce sum c_pea
compute pvir all pressure NULL virial
# every compute that a print or dump reads after a run must be evaluated during
# the run, which thermo output does; etotal needs the thermo temperature
thermo_style custom step temp pe etotal c_pesum press vol c_pvir[1] c_pvir[2] &
    c_pvir[3] c_pvir[4] c_pvir[5] c_pvir[6]
thermo_modify norm no
"""

STATIC = """run 0
print "RESULT $(pe:%.17g) $(c_pesum:%.17g) $(c_pvir[1]:%.17g) $(c_pvir[2]:%.17g) \
$(c_pvir[3]:%.17g) $(c_pvir[4]:%.17g) $(c_pvir[5]:%.17g) $(c_pvir[6]:%.17g)"
write_dump all custom static.{tag}.dump id fx fy fz c_pea modify sort id format float %.17g
"""

NVE = """velocity all create 300.0 4928459 dist gaussian loop geom
fix 1 all nve
timestep 0.0005
thermo 10
run 50
write_dump all custom nve.{tag}.dump id xu yu zu fx fy fz modify sort id format float %.17g
print "RESULT $(pe:%.17g) $(etotal:%.17g)"
"""

NPT = """velocity all create 300.0 4928459 dist gaussian loop geom
fix 1 all npt temp 300.0 300.0 0.05 iso 0.0 0.0 0.5
timestep 0.0005
thermo 5
run 20
print "RESULT $(pe:%.17g) $(press:%.17g) $(vol:%.17g)"
"""


class Report:
    def __init__(self):
        self.failed = []

    def check(self, name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
        if not ok:
            self.failed.append(name)


def build_model():
    """A random PaiNN potential in float64 with forces, stress and energy offsets."""
    torch.manual_seed(0)
    offsets = spk.transform.AddOffsets(
        properties.energy,
        add_mean=True,
        add_atomrefs=True,
        atomrefs=torch.linspace(-1.0, 1.0, 100),
        property_mean=torch.tensor([-0.5]),
    )
    model = spk.model.NeuralNetworkPotential(
        representation=spk.model.PaiNN(
            n_atom_basis=32,
            n_interactions=2,
            radial_basis=spk.nn.GaussianRBF(n_rbf=20, cutoff=CUTOFF),
            cutoff_fn=spk.nn.CosineCutoff(CUTOFF),
        ),
        input_modules=[spk.model.Strain(), spk.model.PairwiseDistances()],
        output_modules=[
            spk.model.Atomwise(n_in=32, output_key=properties.energy),
            spk.model.Forces(calc_stress=True),
        ],
        postprocessors=[spk.transform.CastTo64(), offsets],
        input_dtype_str="float64",
    )
    return model.double().eval()


def crystal():
    atoms = ase.build.bulk("SiC", "zincblende", a=4.36, cubic=True).repeat((3, 3, 2))
    atoms.rattle(0.05, seed=1)
    atoms.wrap()
    return atoms


def reference(model, atoms):
    """Energy, per-atom energies, forces and virial pressure of the model in Python."""
    converter = spk.interfaces.AtomsConverter(
        neighbor_list=spk.transform.ASENeighborList(cutoff=CUTOFF),
        dtype=torch.float64,
    )
    inputs = converter(atoms)
    out = model({k: v.clone() for k, v in inputs.items()})
    _, energy_atom, _, _ = LammpsPotential(model)(
        inputs[properties.R],
        inputs[properties.Z],
        inputs[properties.idx_i],
        inputs[properties.idx_j],
        inputs[properties.offsets],
        torch.ones(len(atoms), dtype=torch.float64),
    )
    # LAMMPS' virial is -stress * volume, so its virial pressure is -stress
    stress = -out[properties.stress][0].detach().numpy() * NKTV2P
    return {
        "energy": out[properties.energy][0].item(),
        "energy_atom": energy_atom.numpy(),
        "forces": out[properties.forces].detach().numpy(),
        # xx, yy, zz, xy, xz, yz, as compute pressure
        "pressure": stress[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
    }


def cores():
    """Cores this process may use; under Slurm that is the job's share of the node."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def run_lammps(args, workdir, name, script, ranks):
    """Run LAMMPS on `script` and return its exit code, output and RESULT values.

    A run that takes longer than --timeout is killed, and it returns the exit code None
    like a run on another number of MPI ranks than `ranks`.
    """
    (workdir / f"in.{name}").write_text(script)
    command = [args.lmp, "-in", f"in.{name}", "-log", f"log.{name}"]
    if ranks > 1:
        command = [args.mpirun, "-np", str(ranks)] + command
    env = dict(os.environ)
    # ranks share the cores, unless the caller decided
    env.setdefault("OMP_NUM_THREADS", str(max(1, cores() // ranks)))
    # in a session of its own, so that a hanging mpirun is killed with its ranks
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.timeout)
        code = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        output += f"\nkilled after {args.timeout:g} s, see log.{name}"
        code = None
    # if mpirun and the MPI library of LAMMPS do not match, each process runs LAMMPS
    # on its own as rank 0 of 1
    grids = re.findall(r"(\d+) by (\d+) by (\d+) MPI processor grid", output)
    used = {math.prod(int(n) for n in grid) for grid in grids}
    if used - {ranks}:
        output += (
            f"\nran on {', '.join(map(str, sorted(used)))} MPI rank(s), not {ranks}"
        )
        code = None
    results = [line for line in output.splitlines() if line.startswith("RESULT")]
    values = [float(v) for v in results[-1].split()[1:]] if results else None
    return code, output, values


def read_dump(path):
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
    columns = lines[start].split()[2:]
    data = np.array([[float(v) for v in line.split()] for line in lines[start + 1 :]])
    return {column: data[:, k] for k, column in enumerate(columns)}


def loop_time(output):
    """Seconds per step of the last run, from LAMMPS' 'Loop time' line."""
    found = re.findall(r"Loop time of (\S+) on \d+ procs for (\d+) steps", output)
    if not found:
        return None
    seconds, steps = found[-1]
    return float(seconds) / max(int(steps), 1)


def relative(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def tail(output, n=15):
    return "\n".join("      " + line for line in output.strip().splitlines()[-n:])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=list(TOLERANCES),
        help="(default: float64)",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="MPI rank counts with domain decomposition (default: 1 2 4)",
    )
    parser.add_argument(
        "--workdir", default="lammps_check", help="(default: lammps_check)"
    )
    parser.add_argument("--lmp", default="lmp", help="LAMMPS binary (default: lmp)")
    parser.add_argument("--mpirun", default="mpirun", help="(default: mpirun)")
    parser.add_argument("--spkdeploy", default="spkdeploy", help="(default: spkdeploy)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="seconds after which a LAMMPS run counts as hanging (default: 300)",
    )
    args = parser.parse_args()

    for tool in (args.lmp, args.mpirun, args.spkdeploy):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found")
    static_tol, position_tol = TOLERANCES[args.dtype]
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    report = Report()
    print(
        f"torch {torch.__version__}, schnetpack {spk.__version__}, work dir {workdir}"
    )

    # model, structure and the Python reference
    model = build_model()
    torch.save(model, workdir / "best_model")
    ase.io.write(
        workdir / "data.lmp",
        crystal(),
        format="lammps-data",
        specorder=list(TYPES),
        masses=True,
        units="metal",
        atom_style="atomic",
    )
    # read back, so that Python sees the positions exactly as LAMMPS does
    atoms = ase.io.read(
        workdir / "data.lmp",
        format="lammps-data",
        Z_of_type={k + 1: z for k, z in enumerate(TYPES.values())},
        atom_style="atomic",
        units="metal",
    )
    ref = reference(model, atoms)
    print(f"{len(atoms)} atoms, cell {np.diag(atoms.cell).round(3).tolist()} Angstrom")

    deploy = subprocess.run(
        [args.spkdeploy, "best_model", "model.pt2"]
        + ["--device", args.device, "--dtype", args.dtype],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    report.check(
        "spkdeploy",
        deploy.returncode == 0,
        "" if deploy.returncode == 0 else "\n" + tail(deploy.stdout + deploy.stderr),
    )
    if deploy.returncode != 0:
        sys.exit(1)

    types = " ".join(str(z) for z in TYPES.values())
    setups = [("dd", ranks, "") for ranks in args.ranks]
    setups.append(("nodd", 1, " no_domain_decomposition"))

    # 1. one force evaluation against Python
    for style, ranks, keyword in setups:
        tag = f"{style}{ranks}"
        script = HEADER.format(style=keyword, types=types) + STATIC.format(tag=tag)
        code, output, values = run_lammps(args, workdir, f"static.{tag}", script, ranks)
        if code != 0 or values is None:
            report.check(f"{tag} runs", False, "\n" + tail(output))
            continue
        energy, energy_sum, *pressure = values
        dump = read_dump(workdir / f"static.{tag}.dump")
        forces = np.stack([dump["fx"], dump["fy"], dump["fz"]], axis=1)
        for name, a, b in [
            ("energy", energy, ref["energy"]),
            ("sum of per-atom energies", energy_sum, energy),
            ("per-atom energies", dump["c_pea"], ref["energy_atom"]),
            ("forces", forces, ref["forces"]),
            ("virial pressure", pressure, ref["pressure"]),
        ]:
            error = relative(a, b)
            report.check(f"{tag} {name}", error < static_tol, f"rel. error {error:.1e}")

    # 2. NVE trajectories agree between the setups
    trajectories = {}
    for style, ranks, keyword in setups:
        tag = f"{style}{ranks}"
        script = HEADER.format(style=keyword, types=types) + NVE.format(tag=tag)
        code, output, values = run_lammps(args, workdir, f"nve.{tag}", script, ranks)
        if code != 0 or values is None:
            report.check(f"{tag} NVE runs", False, "\n" + tail(output))
            continue
        dump = read_dump(workdir / f"nve.{tag}.dump")
        trajectories[tag] = np.stack([dump["xu"], dump["yu"], dump["zu"]], axis=1)
        step = loop_time(output)
        timing = f"{step * 1e3:.1f} ms/step" if step is not None else "no timing"
        print(f"      {tag}: {timing} on {ranks} rank(s), etotal {values[1]:.6f} eV")
    if len(trajectories) > 1:
        first, positions = next(iter(trajectories.items()))
        for tag, other in list(trajectories.items())[1:]:
            error = float(np.abs(other - positions).max())
            report.check(
                f"NVE {tag} vs {first}",
                error < position_tol,
                f"max. position difference {error:.1e} Angstrom after 50 steps",
            )

    # 3. no_domain_decomposition refuses several ranks
    script = HEADER.format(style=" no_domain_decomposition", types=types) + "run 0\n"
    code, output, _ = run_lammps(args, workdir, "nodd2", script, 2)
    ok = code not in (0, None) and "runs on one MPI rank" in output
    report.check(
        "no_domain_decomposition refuses 2 ranks",
        ok,
        "" if ok else "it ran" if code == 0 else "\n" + tail(output),
    )

    # 4. NPT runs with the virial
    ranks = max(args.ranks)
    script = HEADER.format(style="", types=types) + NPT
    code, output, values = run_lammps(args, workdir, "npt", script, ranks)
    ok = code == 0 and values is not None and all(math.isfinite(v) for v in values)
    detail = (
        f"pe {values[0]:.4f} eV, press {values[1]:.1f} bar, vol {values[2]:.2f} A^3"
        if ok
        else "\n" + tail(output)
    )
    report.check(f"NPT on {ranks} rank(s)", ok, detail)

    print()
    if report.failed:
        print(f"{len(report.failed)} check(s) failed: {', '.join(report.failed)}")
        print(f"inputs, logs and dumps are in {workdir}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()

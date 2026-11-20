# Changelog

All notable changes to SchNetPack are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [3.0.0] - Unreleased

3.0 reorganizes the package, so code, configs and saved run directories written
for 2.x need updating. The steps are listed under
[Migrating from 2.x](#migrating-from-2x) below.

### Highlights

- **One model package.** `schnetpack.atomistic` and `schnetpack.representation`
  are merged into `schnetpack.model`.
- **Training without Lightning.** Loss assembly is in `schnetpack.objectives`
  and depends only on PyTorch, so a model can be trained in a plain PyTorch loop.
  Everything that uses PyTorch Lightning is in `schnetpack.lightning`, and
  `import schnetpack` no longer imports Lightning.
- **New data API.** Datasets such as `QM9` are now datasets
  (`ASEAtomsData` subclasses) and no longer data modules. They are passed to
  `schnetpack.lightning.AtomsDataModule`, which owns splitting, statistics and
  loaders.
- **`schnetpack.dynamics`** (new): batched optimizers, samplers, calculators,
  hooks and guidance. These loops move many structures with one model at once.
- **`schnetpack.generative`** (new): building blocks for diffusion and flow
  matching on atomistic systems, in pure PyTorch.
- **Distillation and MACE.** A model can be trained from a frozen teacher, and
  MACE models can be used as the teacher or fine-tuned directly.

### Migrating from 2.x

**Imports.** Every class from `schnetpack.atomistic` and
`schnetpack.representation` is now available from `schnetpack.model`:

| 2.x                                          | 3.0                                       |
| -------------------------------------------- | ----------------------------------------- |
| `schnetpack.representation.PaiNN` (and `SchNet`, `SO3net`, `FieldSchNet`, …) | `schnetpack.model.PaiNN`, … |
| `schnetpack.atomistic.Atomwise` (and `Forces`, `PairwiseDistances`, `DipoleMoment`, …) | `schnetpack.model.Atomwise`, … |
| `schnetpack.task.ModelOutput`                | `schnetpack.ModelOutput`                  |
| `schnetpack.task.ConsiderOnlySelectedAtoms(selection_name)` | `schnetpack.AtomMask(key)`  |
| `ModelOutput(..., constraints=[...])`        | `ModelOutput(..., masks=[...])`           |
| `schnetpack.AtomisticTask`                   | `schnetpack.lightning.AtomisticTask`      |
| `schnetpack.data.AtomsDataModule`            | `schnetpack.lightning.AtomsDataModule`    |
| `schnetpack.train.ModelCheckpoint`, `ExponentialMovingAverage`, `PredictionWriter` | `schnetpack.lightning.…` |
| `schnetpack.datasets.GDMLDataModule`         | `schnetpack.datasets.md17.GDMLDataset`    |
| `schnetpack.data.create_dataset(...)`        | `schnetpack.data.ASEAtomsData.create(...)` |
| `schnetpack.data.load_dataset(...)`          | `schnetpack.data.ASEAtomsData(...)`       |

The old import paths of `AtomisticTask`, `AtomsDataModule` and the callbacks
still work but emit a `DeprecationWarning`. The old `atomistic` and
`representation` paths raise an `ImportError`.

**Datasets and the data module.** Settings for splitting and loading move from
the dataset to the data module:

```python
# 2.x
qm9 = QM9("qm9.db", batch_size=100, num_train=110000, num_val=10000,
          transforms=[...])

# 3.0
from schnetpack.lightning import AtomsDataModule

qm9 = AtomsDataModule(
    dataset=QM9("qm9.db", transforms=[...]),
    batch_size=100, num_train=110000, num_val=10000,
)
```

The dataset classes `QM9`, `MD17`, `MD22`, `rMD17`, `QM7X`, `ANI1`, `ISO17`,
`MaterialsProject` and `OrganicMaterialsDatabase` therefore no longer accept
`batch_size`, `num_train`, `num_val`, `num_test`, `split_file`, the
`*_batch_size` and `*_workers` arguments, or `format`. `data_workdir` is removed.
`AtomsDataModule(num_workers=...)` now defaults to `0` (it was `8`).
`calculate_stats` and `estimate_atomrefs` take a dataset instead of a
dataloader.

**Hydra configs.** The data config now nests the dataset under `data.dataset`.
This changes command-line overrides:

```bash
# 2.x
spktrain experiment=qm9_atomwise data.datapath=/data/qm9.db
# 3.0
spktrain experiment=qm9_atomwise data.dataset.datapath=/data/qm9.db
```

Custom configs and the `config.yaml` saved in existing run directories still
refer to the old `_target_` paths. Update them using the table above,
e.g. `schnetpack.representation.PaiNN` → `schnetpack.model.PaiNN`, and
`schnetpack.data.AtomsDataModule` → `schnetpack.lightning.AtomsDataModule`.

**Saved models.** Models pickled by 2.x store the old module paths. Load them
with `schnetpack.utils.load_model(path)`, which maps those paths onto the new
ones. A plain `torch.load` fails with `ModuleNotFoundError`. Models deployed for
LAMMPS with `spkdeploy` are unchanged.

### Added

- `schnetpack.objectives`: `ModelOutput`, `UnsupervisedModelOutput`, the loss
  masks `LossMask` and `AtomMask`, and the functions `extract_targets` and
  `calculate_loss`. `predict_without_postprocessing` runs a model with its
  postprocessing switched off.
- `schnetpack.lightning`: `AtomisticTask`, `AtomsDataModule`, the callbacks,
  and `load_task_from_checkpoint`.
- `schnetpack.dynamics`:
  - Optimizers: `LBFGS` (one inverse-Hessian approximation per structure, so
    structures of different size relax in one batch), `GradientDescent` and
    GPFF's `DirectDenoising`.
  - `Sampler`, with SDE, ODE and ancestral updates on a time grid.
  - Force, score, velocity, noise and x0 calculators.
  - Hooks such as `FreezeScaffold`, and guidance terms such as
    `HarmonicRestraint`.
  - Structure providers, and `BatchNeighborList`.
- `schnetpack.generative`:
  - Noise schedules (VE, VP, flow matching).
  - Forward noising, including a periodic interpolant for positions in a cell.
  - Couplings (OT, permutation, PC-variance).
  - Parametrizations (eps, x0, score, velocity, pseudo-force) and `MatchingLoss`.
  - Priors, and the `Diffuse` transform, which noises inside the dataloader.
- `schnetpack.uncertainty`: ensemble disagreement per structure
  (`AbsoluteUncertainty`, `RelativeUncertainty`), shared by the ASE and dynamics
  ensembles. `schnetpack.model.NNEnsemble` evaluates several models together.
- Knowledge distillation:
  - `AtomisticTask(teacher=...)` and `schnetpack.train.Distillation`.
  - The teachers `SchNetPackTeacher` and `MaceTeacher`.
  - `DistillationNeighborList`.
  - The experiment config `experiment=distillation`.
- MACE:
  - `schnetpack.model.MaceModel`, which runs MACE on SchNetPack batches.
  - `load_mace`, and `ExportedMace` for models compiled with `torch.export`.
  - Fine-tuning via `experiment=mace_finetune`.
  - Compiling MACE models needs the new `mace` extra
    (`pip install schnetpack[mace]`). Loading a compiled model does not.
- Model and transforms:
  - `AtomwiseVector` (per-atom vector outputs).
  - `TimeConditioning`.
  - `PaiNN(norm_epsilon=...)`.
  - The transforms `AllToAllNeighborList` and `NiggliReduce`.
- Data:
  - `ChunkedRandomSampler`, which splits a pass over a large dataset into
    shorter epochs.
  - `StatsAtomrefProvider`, which caches training-set statistics and atom
    references next to the split file.
  - `prune_neighbors` and `DownloadableASEAtomsData`.
  - The `MPTraj` dataset (MPtrj): 1.58 M DFT relaxation frames of bulk crystals
    with energies, forces, stresses and magnetic moments, and the data config
    `data=mptraj`, which splits by material.
- Configs: a progress bar callback, and a wandb logger setup (#787).
- How-tos for batch-wise relaxations, ensemble calculations and guidance.
  The tutorials and how-tos are now stored as jupytext `py:percent` files.
- `py.typed`. SchNetPack now ships its type annotations.

### Changed

- Console scripts (`spktrain`, `spkpredict`, `spkmd`, `spkconvert`,
  `spkdeploy`) are `[project.scripts]` entry points. `spkconvert` and
  `spkdeploy` moved into `schnetpack.cli`.
- `spkdeploy model package.pt2 [--device cuda] [--dtype float64]` compiles a
  potential with AOTInductor (`schnetpack.interfaces.lammps_interface`) instead
  of writing TorchScript. The package holds energy, per-atom energies, forces
  and the virial, and runs only with the torch version and on the device it was
  built for, so it is built on the machine LAMMPS runs on. The LAMMPS pair
  style is being rewritten to load it.
- Optional dependency extras: `test`, `dev`, `doc` and `mace`. `black`,
  `pytest`, `pre-commit`, `protobuf` and `progressbar` are no longer installed
  as runtime dependencies.
- Raised minimum versions: `ase>=3.23`, `hydra-core>=1.3,<2`,
  `pytorch_lightning>=2.6.1,<3` and, in the `test` extra,
  `pytest-benchmark>=5.3.1`. Before Lightning 2.6.1, random transforms in
  DataLoader workers replayed the same random stream every epoch under
  `seed_everything(workers=True)`, as `spktrain` uses it.
- The TensorBoard logger config writes to a fixed `version: 0`, so a resumed run
  stays one TensorBoard run.
- `schnetpack.datasets` passes on the data API again (e.g.
  `schnetpack.datasets.ASEAtomsData`), as in 2.x.
- New documentation: a Markdown (MyST) site in the look of the Flax and JAX docs
  (`sphinx_book_theme`, light and dark mode), with Getting started, Tutorials,
  Guides, Concepts, Migrating, API reference and Project sections. The tutorials
  and how-tos moved from `examples/` into `docs/tutorials` and `docs/guides` as
  jupytext `.py` notebooks; Read the Docs executes them on every build, so the
  pages show their outputs. No `.ipynb` is committed.
- Ensemble uncertainty calculations for ASE and dynamics are refactored onto
  `schnetpack.uncertainty`.
- `ASEAtomsData.add_systems` writes all systems over one database connection
  instead of one per system, 20x faster (2.6 to 0.13 ms per periodic system).

### Fixed

- `SO3net` and `PaiNN` in float64 created some tensors in float32, so
  `SO3net` used `sqrt(0.5)` rounded to float32 (a relative error of about 2e-8).
- `spkmd` was broken in every installed wheel and sdist, because the
  `md_configs` YAML files were not packaged.
- `spktrain` raises an error if the early-stopping metric becomes NaN or
  infinite. Before, it reported the run as finished (#802).
- `Atomwise` with `aggregation_mode="avg"` failed for multiple outputs (#801).
- Resumed training restores the callback and training state (#788).
- Hydra config search paths no longer depend on `${oc.env:PWD}` (#794).
- Neighbor-list cache errors are logged instead of being swallowed, and
  unreadable cache files are rebuilt. All neighbor lists convert torch inputs
  to numpy the same way.
- The MD neighbor list now runs on `BatchNeighborList`, which it shares with
  batch-wise relaxation. This fixes its skin handling and the collection of
  atom triples.
- Invalid escape sequences in the source, a future `SyntaxError`.
- Mutable default arguments (`[]`, `{}`) in the MD calculators, simulator
  hooks and `StaticExternalFields` are replaced by `None`.
- `schnetpack.utils.load_model(path, weights_only=...)` raised a `TypeError`.
  `weights_only` still defaults to `False`.
- `Forces` and `Response` divided the stress by the signed cell volume, which
  flipped its sign for left-handed cells.
- `Response(map_properties=...)` listed the original names as model outputs,
  while the results are stored under the mapped names, and changed the
  caller's dict.
- `MatScipyNeighborList` gave atoms outside the cell offsets along non-periodic
  directions, to images that do not exist. A periodic system whose cell has no
  volume now raises an error instead of losing its lattice vectors.
- `QM7X.FPBE0` had the name of `FMBD`, so the PBE0 forces were never stored and
  loading `FPBE0` returned the MBD forces. Databases built before hold no PBE0
  forces and have to be rebuilt. An interrupted extraction no longer leaves a
  partial file that later runs reuse.
- With Lightning >= 2.6, `last.ckpt` stayed at the best epoch once the
  monitored metric stopped improving, so a resumed run repeated the epochs since
  then.
- A resumed run trains like an uninterrupted one: checkpoints hold the RNG
  states and, with `cache=True`, the state of the training order.
- `ExponentialMovingAverage` continued training from the weights of the last
  validation when validation ran less than once per epoch, and from the EMA
  average when it ran within an epoch.
- `PredictionWriter` with `write_interval="epoch"`, the default of
  `spkpredict`, failed.
- `spktrain` crashed with two loggers; the config is now logged to every
  logger. A resumed run keeps its seed unless a new one is given.
  `trainer=debug_trainer` no longer passes the removed `gpus` argument, and the
  `aim` logger config no longer doubles its repo path.

### Removed

- `schnetpack.atomistic` and `schnetpack.representation`, merged into
  `schnetpack.model`.
- `BaseAtomsData`, `AtomsDataFormat`, `resolve_format`, `create_dataset`,
  `load_dataset` and `AtomsDataModuleError`. `ASEAtomsData` is the only
  dataset backend.
- The `TMQM` dataset.
- `tox.ini` and the `black` workflow. Linting and formatting use `ruff` via
  pre-commit, and CI runs in a single `ci.yml`.

## [2.2.0] - 2025-12-19

See the [GitHub release](https://github.com/atomistic-machine-learning/schnetpack/releases/tag/v2.2.0).

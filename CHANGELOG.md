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
- Configs: a progress bar callback, and a wandb logger setup (#787).
- How-tos for batch-wise relaxations, ensemble calculations and guidance.
  The tutorials and how-tos are now stored as jupytext `py:percent` files.
- `py.typed`. SchNetPack now ships its type annotations.

### Changed

- Console scripts (`spktrain`, `spkpredict`, `spkmd`, `spkconvert`,
  `spkdeploy`) are `[project.scripts]` entry points. `spkconvert` and
  `spkdeploy` moved into the package as `schnetpack.cli_scripts`.
- Optional dependency extras: `test`, `dev`, `doc` and `mace`. `black`,
  `pytest`, `pre-commit`, `protobuf` and `progressbar` are no longer installed
  as runtime dependencies.
- Raised minimum versions: `ase>=3.23`, `hydra-core>=1.3,<2`,
  `pytorch_lightning>=2.2,<3`.
- The legacy notebook tutorials moved to `examples/tutorials_deprecated`.
- Ensemble uncertainty calculations for ASE and dynamics are refactored onto
  `schnetpack.uncertainty`.

### Fixed

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

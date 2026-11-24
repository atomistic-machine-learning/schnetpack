# SchNetPack

```{div} sd-text-left sd-font-italic
Neural networks for atomistic systems
```

---

SchNetPack is a toolbox for the development and application of deep neural networks
that predict potential energy surfaces and other quantum-chemical properties of
molecules and materials. It contains the building blocks of atomistic neural
networks, manages their training, and gives simple access to common benchmark
datasets, so new models are easy to implement and evaluate.

## Features

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Neural network potentials
:class-card: spk-feature
:shadow: none
SchNet, PaiNN and SO3net as building blocks, trained on energies, forces and
other properties.
:::

:::{grid-item-card} Molecular dynamics
:class-card: spk-feature
:shadow: none
Batched MD, relaxations and ensembles in `schnetpack.md` and
`schnetpack.dynamics`, plus ASE and LAMMPS interfaces.
:::

:::{grid-item-card} Generative modelling
:class-card: spk-feature
:shadow: none
Diffusion and flow matching on positions, cells and atom types in
`schnetpack.generative`.
:::

:::{grid-item-card} Fast training
:class-card: spk-feature
:shadow: none
Compiled models, CUDA graphs and data held in memory, with or without
PyTorch Lightning.
:::
::::

## Basic usage

```bash
pip install --upgrade schnetpack
mkdir spk_workdir && cd spk_workdir
spk train experiment=qm9_atomwise
```

---

## Learn more

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card} {material-regular}`rocket_launch;2em` Getting started
:link: getting_started/installation
:link-type: doc
:::

:::{grid-item-card} {material-regular}`school;2em` Tutorials
:link: tutorials/data
:link-type: doc
:::

:::{grid-item-card} {material-regular}`build;2em` Guides
:link: guides/configs
:link-type: doc
:::

:::{grid-item-card} {material-regular}`lightbulb;2em` Concepts
:link: concepts/overview
:link-type: doc
:::

:::{grid-item-card} {material-regular}`swap_horiz;2em` Migrating from 2.x
:link: migrating
:link-type: doc
:::

:::{grid-item-card} {material-regular}`menu_book;2em` API reference
:link: api/schnetpack
:link-type: doc
:::
::::

```{toctree}
:hidden:
:caption: Getting started

getting_started/installation
getting_started/quickstart
getting_started/cli
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/data
tutorials/properties
tutorials/force_fields
tutorials/molecular_dynamics
tutorials/materials
```

```{toctree}
:hidden:
:caption: Guides

guides/configs
guides/saving_models
guides/training_speed
guides/md_cli
guides/batchwise_relaxations
guides/ensembles_uncertainty
guides/guidance
guides/lammps
```

```{toctree}
:hidden:
:caption: Concepts

concepts/overview
concepts/generative/index
concepts/glossary
```

```{toctree}
:hidden:
:caption: Migrating

migrating
```

```{toctree}
:hidden:
:caption: API reference

api/schnetpack
api/data
api/datasets
api/transform
api/model
api/nn
api/objectives
api/lightning
api/train
api/md
api/dynamics
api/generative
api/interfaces
api/utils
api/cli
```

```{toctree}
:hidden:
:caption: Project

contributing
citing
changelog
```

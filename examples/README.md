# Examples

In this directory, you find code examples and tutorials to demonstrate the functionality of SchNetPack

## Running them as notebooks

The tutorials and how-tos are stored as [jupytext](https://jupytext.readthedocs.io)
`py:percent` files instead of `.ipynb`, so that they diff and merge like normal source
code. To work through one interactively:

```
pip install jupytext
jupytext --to notebook tutorials/tutorial_01_preparing_data.py
```

This writes `tutorial_01_preparing_data.ipynb` next to it, which you open in Jupyter as
usual. The generated notebook is git-ignored, so any outputs you produce stay local.

With jupytext installed you can also skip the conversion and open the `.py` directly from
the JupyterLab file browser (*right-click → Open With → Notebook*). Your edits are then
written back to the `.py` automatically.

If you just want to read them, the rendered versions are at
[schnetpack.readthedocs.io](https://schnetpack.readthedocs.io).

## Which ones the docs build executes

The documentation build runs every notebook here so that the rendered pages show real
outputs. The four training tutorials are too expensive for that -- they download datasets
and train models -- so each opts out with a header at the top of its `.py`:

```
# ---
# jupyter:
#   nbsphinx:
#     execute: never
# ---
```

Add that header to any new example that cannot run in a couple of minutes on a CPU;
without it, the example has to keep working, because the weekly `docs` CI job and Read
the Docs both execute it.

## Tutorials
Jupyter notebooks demonstrating general concepts and workflows

[Preparing and loading your data](tutorials/tutorial_01_preparing_data.py)

[Training a neural network on QM9
](tutorials/tutorial_02_qm9.py)

[Training a model on forces and energies](tutorials/tutorial_03_force_models.py)

[Molecular dynamics in SchNetPack](tutorials/tutorial_04_molecular_dynamics.py)

[Force fields for materials](tutorials/tutorial_05_materials.py)


## How-To
Short notebooks showing a particular use-case or functionality

[Batch-wise Structure Relaxation
](howtos/howto_batchwise_relaxations.py)

[Uncertainty Estimation with Model Ensembles
](howtos/howto_ensemble_calculation.py)

# Examples

In this directory, you find code examples and tutorials to demonstrate the functionality of SchNetPack

## Running them as notebooks

The tutorials and how-tos are stored as [jupytext](https://jupytext.readthedocs.io)
`py:percent` files instead of `.ipynb`, so that they diff and merge like normal source
code. To work through one interactively:

```
pip install jupytext
jupytext --to notebook howtos/howto_batchwise_relaxations.py
```

This writes `howto_batchwise_relaxations.ipynb` next to it, which you open in Jupyter as
usual. The generated notebook is git-ignored, so any outputs you produce stay local.

With jupytext installed you can also skip the conversion and open the `.py` directly from
the JupyterLab file browser (*right-click → Open With → Notebook*). Your edits are then
written back to the `.py` automatically.

If you just want to read them, the rendered versions are at
[schnetpack.readthedocs.io](https://schnetpack.readthedocs.io).

## Which ones CI executes

Read the Docs and pull requests only render the examples. The weekly `docs` CI job
executes every one of them, so an example has to keep working with the current code. The
deprecated tutorials are excluded with a header at the top of their `.py`:

```
# ---
# jupyter:
#   nbsphinx:
#     execute: never
# ---
```

Add that header to any new example that cannot run in a couple of minutes on a CPU.

## Tutorials (deprecated)
Notebooks demonstrating general concepts and workflows. They were written for
SchNetPack 2.x and have not been updated for SchNetPack 3 yet, so parts of their code may
not run with the current version.

[Preparing and loading your data](tutorials_deprecated/tutorial_01_preparing_data.py)

[Training a neural network on QM9](tutorials_deprecated/tutorial_02_qm9.py)

[Training a model on forces and energies](tutorials_deprecated/tutorial_03_force_models.py)

[Molecular dynamics in SchNetPack](tutorials_deprecated/tutorial_04_molecular_dynamics.py)

[Force fields for materials](tutorials_deprecated/tutorial_05_materials.py)


## How-To
Short notebooks showing a particular use-case or functionality

[Batch-wise Structure Relaxation](howtos/howto_batchwise_relaxations.py)

[Uncertainty Estimation with Model Ensembles](howtos/howto_ensemble_calculation.py)

[Restraining a Relaxation](howtos/howto_guidance.py)

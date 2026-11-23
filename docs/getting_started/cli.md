# Training from the command line

Installing SchNetPack puts the `spk` command on your PATH; `spk train` trains a
SchNetPack model [^schnetpack2] [^schnetpack] on a benchmark dataset with one
command. The CLI is built on [Hydra](https://hydra.cc/) and follows the
[Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template), so
every part of the model, the data and the training can be changed from the command
line. [Configuration and CLI](../guides/configs.md) explains the configs in detail.

## Example 1: QM9

This example trains on the QM9 dataset [^qm]; the same steps work for the other
benchmark datasets. First, create a working directory for the data and the runs:

```bash
mkdir spk_workdir
cd spk_workdir
```

Then start the training of a SchNet model with the default settings for QM9:

```bash
spk train experiment=qm9_atomwise
```

The script prints the config of the experiment `qm9_atomwise`, downloads the dataset
to `spk_workdir/data` if it is not there yet, and starts the training.

Every value of the config can be changed from the command line, including the
directories for runs and data. By default, a run is stored in a subdirectory of
`spk_workdir/runs` named by a unique run id. To change that:

```bash
spk train experiment=qm9_atomwise run.data_dir=/my/data/dir run.path=~/all_my_runs run.id=this_run
```

`spk train experiment=qm9_atomwise --help` prints the full config with every
parameter. Nested parameters are set with dots:

```bash
spk train experiment=qm9_atomwise data.batch_size=64
```

Hydra groups parameters into config groups, so a whole dataset, model or
representation can be swapped at once. To replace the default SchNet
representation [^schnet1] [^schnet2] [^schnet3] by PaiNN [^painn]:

```bash
spk train experiment=qm9_atomwise model/representation=painn
```

The slash loads a predefined config group; the dot changes a single value. The
config group `model/representation=painn` corresponds to this part of the config:

```yaml
model:
  representation:
    _target_: schnetpack.model.PaiNN
    n_atom_basis: 128
    n_interactions: 3
    shared_interactions: false
    shared_filters: false
    radial_basis:
      _target_: schnetpack.nn.radial.GaussianRBF
      n_rbf: 20
      cutoff: ${globals.cutoff}
    cutoff_fn:
      _target_: schnetpack.nn.cutoff.CosineCutoff
      cutoff: ${globals.cutoff}
```

Both can be combined, for example to load PaiNN with five interactions:

```bash
spk train experiment=qm9_atomwise model/representation=painn model.representation.n_interactions=5
```

More on config groups is in the
[Hydra docs](https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/).

## Example 2: Potential energy surfaces

The same command trains a force field. The experiment `md17` selects a
representation and output modules that predict the energy and its derivatives, the
forces. To train on one molecule of the MD17 dataset:

```bash
spk train experiment=md17 data.molecule=uracil
```

MD17 has reference energies and forces, so the loss is a weighted sum of both. The
weights are part of the output definitions in the `task` config group:

```yaml
task:
  outputs:
    - _target_: schnetpack.ModelOutput
      name: ${globals.energy_key}
      loss_fn:
        _target_: torch.nn.MSELoss
      metrics:
        mae:
          _target_: torchmetrics.regression.MeanAbsoluteError
        rmse:
          _target_: torchmetrics.regression.MeanSquaredError
          squared: False
      loss_weight: 0.01
    - _target_: schnetpack.ModelOutput
      name: ${globals.forces_key}
      loss_fn:
        _target_: torch.nn.MSELoss
      metrics:
        mae:
          _target_: torchmetrics.regression.MeanAbsoluteError
        rmse:
          _target_: torchmetrics.regression.MeanSquaredError
          squared: False
      loss_weight: 0.99
```

For training on energies and forces, a stronger weight on the forces works best.
The defaults are 0.01 for the energy and 0.99 for the forces. To change them:

```bash
spk train experiment=md17 data.molecule=uracil task.outputs.0.loss_weight=0.05 task.outputs.1.loss_weight=0.95
```

## Logging

Besides the command line output, SchNetPack logs through PyTorch Lightning. The
TensorBoard logger is on by default; open the logs of a run with:

```bash
tensorboard --logdir=<rundir>
```

```{image} tensorboard.png
:alt: TensorBoard with the metrics, learning rate and losses of a QM9 run
```

SchNetPack also has configs for a CSV logger, [Aim](https://github.com/aimhubio/aim)
and [Weights & Biases](https://wandb.ai/). Aim and Weights & Biases need a separate
install. Select a logger with:

```bash
spk train experiment=md17 logger=csv
```

## References

[^schnetpack2]: K.T. Schütt, S.S.P. Hessmann, N.W.A. Gebauer, J. Lederer, M. Gastegger.
    [SchNetPack 2.0: A neural network toolbox for atomistic machine learning](https://doi.org/10.1063/5.0138367).
    The Journal of Chemical Physics **158** (14), 144801, 2023.

[^schnetpack]: K.T. Schütt, P. Kessel, M. Gastegger, K.A. Nicoli, A. Tkatchenko, K.-R. Müller.
    [SchNetPack: A Deep Learning Toolbox For Atomistic Systems](https://doi.org/10.1021/acs.jctc.8b00908).
    Journal of Chemical Theory and Computation **15** (1), 448–455, 2019.

[^schnet1]: K.T. Schütt, F. Arbabzadah, S. Chmiela, K.-R. Müller, A. Tkatchenko.
    [Quantum-chemical insights from deep tensor neural networks](https://www.nature.com/articles/ncomms13890).
    Nature Communications **8**, 13890, 2017.

[^schnet2]: K.T. Schütt, P.-J. Kindermans, H.E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
    [SchNet: A continuous-filter convolutional neural network for modeling quantum interactions](https://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions).
    Advances in Neural Information Processing Systems **30**, 992–1002, 2017.

[^schnet3]: K.T. Schütt, P.-J. Kindermans, H.E. Sauceda, A. Tkatchenko, K.-R. Müller.
    [SchNet – A deep learning architecture for molecules and materials](https://doi.org/10.1063/1.5019779).
    The Journal of Chemical Physics **148** (24), 241722, 2018.

[^painn]: K.T. Schütt, O.T. Unke, M. Gastegger.
    [Equivariant message passing for the prediction of tensorial properties and molecular spectra](https://proceedings.mlr.press/v139/schutt21a.html).
    ICML 2021.

[^qm]: [quantum-machine.org](http://www.quantum-machine.org/data)

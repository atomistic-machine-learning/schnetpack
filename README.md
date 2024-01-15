# SchNetPack - Deep Neural Networks for Atomistic Systems
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)


SchNetPack is a toolbox for the development and application of deep neural networks to the prediction of potential energy surfaces and other quantum-chemical properties of molecules and materials. It contains basic building blocks of atomistic neural networks, manages their training and provides simple access to common benchmark datasets. This allows for an easy implementation and evaluation of new models.

The documentation can be found [here](https://schnetpack.readthedocs.io).

##### Features

- SchNet - an end-to-end continuous-filter CNN for molecules and materials [1-3]
- PaiNN - equivariant message-passing for molecules and materials [4]
- Output modules for dipole moments, polarizability, stress, and general response properties
- Modules for electrostatics, Ewald summation, ZBL repulsion
- GPU-accelerated molecular dynamics code incl. path-integral MD, thermostats, barostats

## Installation

### Install with pip

The simplest way to install SchNetPack is through pip which will automatically get the source code from PyPI:
```
pip install schnetpack
```

### Install from source

You can also install the most recent code from our repository:

```
git clone https://github.com/atomistic-machine-learning/schnetpack.git
cd schnetpack
pip install .
```

### Visualization with Tensorboard

SchNetPack supports multiple logging backends via PyTorch Lightning. The default logger is Tensorboard. SchNetPack also supports TensorboardX.


## Getting started

The best place to get started is training a SchNetPack model on a common benchmark dataset via the command line
interface (CLI).
When installing SchNetPack, the training script `spktrain` is added to your PATH.
The CLI uses [Hydra](https://hydra.cc/) and is based on the PyTorch Lightning/Hydra template that can be found
[here](https://github.com/ashleve/lightning-hydra-template).
This enables a flexible configuration of the model, data and training process.
To fully take advantage of these features, it might be helpful to have a look at the Hydra and PyTorch Lightning docs.

### Example 1: QM9

In the following, we focus on using the CLI to train on the QM9 dataset, but the same
procedure applies for the other benchmark datasets as well.
First, create a working directory, where all data and runs will be stored:

```
mkdir spk_workdir
cd spk_workdir
```

Then, the training of a SchNet model with default settings for QM9 can be started by:

```
spktrain experiment=qm9_atomwise
```

The script prints the defaults for the experiment config `qm9_atomwise`.
The dataset will be downloaded automatically to `spk_workdir/data`, if it does not exist yet.
Then, the training will be started.

All values of the config can be changed from the command line, including the directories for run and data.
By default, the model is stored in a directory with a unique run id hash as a subdirectory of `spk_workdir/runs`.
This can be changed as follows:

```
spktrain experiment=qm9_atomwise run.data_dir=/my/data/dir run.path=~/all_my_runs run.id=this_run
```

If you call `spktrain experiment=qm9_atomwise --help`, you can see the full config with all the parameters
that can be changed.
Nested parameters can be changed as follows:

```
spktrain experiment=qm9_atomwise run.data_dir=<path> data.batch_size=64
```

Hydra organizes parameters in config groups which allows hierarchical configurations consisting of multiple
yaml files. This allows to easily change the whole dataset, model or representation.
For instance, changing from the default SchNet representation to PaiNN, use:

```
spktrain experiment=qm9_atomwise run.data_dir=<path> model/representation=painn
```

It is a bit confusing at first when to use "." or "/". The slash is used, if you are loading a preconfigured config
group, while the dot is used changing individual values. For example, the config group "model/representation"
corresponds to the following part of the config:

```
    model:
      representation:
        _target_: schnetpack.representation.PaiNN
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

If you would want to additionally change some value of this group, you could use:

```
spktrain experiment=qm9_atomwise run.data_dir=<path> model/representation=painn model.representation.n_interactions=5
```

For more details on config groups, have a look at the
[Hydra docs](https://hydra.cc/docs/next/tutorials/basic/your_first_app/config_groups).


### Example 2: Potential energy surfaces

The example above uses `AtomisticModel` internally, which is a
`pytorch_lightning.LightningModule`, to predict single properties.
The following example will use the same class to predict potential energy surfaces,
in particular energies with the appropriate derivates to obtain forces and stress tensors.
This works since the pre-defined configuration for the MD17 dataset,
provided from the command line by `experiment=md17`, is selecting the representation and output modules that
`AtomisticModel` is using.
A more detailed description of the configuration and how to build your custom configs can be
found [here](https://schnetpack.readthedocs.io/en/latest/userguide/configs.html).

The `spktrain` script can be used to train a model for a molecule from the MD17 datasets

```
spktrain experiment=md17 data.molecule=uracil
```

In the case of MD17, reference calculations of energies and forces are available.
Therefore, one needs to set weights for the losses of those properties.
The losses are defined as part of output definitions in the `task` config group:

```
    task:
      outputs:
        - _target_: schnetpack.task.ModelOutput
          name: ${globals.energy_key}
          loss_fn:
            _target_: torch.nn.MSELoss
          metrics:
            mae:
              _target_: torchmetrics.regression.MeanAbsoluteError
            mse:
              _target_: torchmetrics.regression.MeanSquaredError
          loss_weight: 0.005
        - _target_: schnetpack.task.ModelOutput
          name: ${globals.forces_key}
          loss_fn:
            _target_: torch.nn.MSELoss
          metrics:
            mae:
              _target_: torchmetrics.regression.MeanAbsoluteError
            mse:
              _target_: torchmetrics.regression.MeanSquaredError
          loss_weight: 0.995
```

For a training on *energies** and *forces*, we recommend to put a stronger
weight on the loss of the force prediction during training.
By default, the loss weights are set to 0.005 for the energy and 0.995 for forces.
This can be changed as follow:

```
spktrain experiment=md17 data.molecule=uracil task.outputs.0.loss_weight=0.005 task.outputs.1.loss_weight=0.995
```

### Logging

Beyond the output of the command line, SchNetPack supports multiple logging backends over PyTorch Lightning.
By default, the Tensorboard logger is activated.
If TensorBoard is installed, the results can be shown by calling:

```
tensorboard --logdir=<rundir>
```

Furthermore, SchNetPack comes with configs for a CSV logger and [Aim](https://github.com/aimhubio/aim).
These can be selected as follows:

```
spktrain experiment=md17 logger=csv
```

## LAMMPS interface

SchNetPack comes with an interface to LAMMPS. A detailed installation guide is linked in the [How-To section of our documentation](https://schnetpack.readthedocs.io/en/latest/howtos/lammps.html).

## Extensions

SchNetPack can be used as a base for implementations of advanced atomistic neural networks and training tasks.
For example, there exists an [extension package](https://github.com/atomistic-machine-learning/schnetpack-gschnet) called `schnetpack-gschnet` for the most recent version of cG-SchNet [5], a conditional generative model for molecules.
It demonstrates how a complex training task can be implemented in a few custom classes while leveraging the hierarchical configuration and automated training procedure of the SchNetPack framework.


## Citation

If you are using SchNetPack in your research, please cite:

K.T. Schütt, S.S.P. Hessmann, N.W.A. Gebauer, J. Lederer, M. Gastegger.
SchNetPack 2.0: A neural network toolbox for atomistic machine learning.
J. Chem. Phys. 2023, 158 (14): 144801.
[10.1063/5.0138367](https://doi.org/10.1063/5.0138367).

K.T. Schütt, P. Kessel, M. Gastegger, K. Nicoli, A. Tkatchenko, K.-R. Müller.
SchNetPack: A Deep Learning Toolbox For Atomistic Systems.
J. Chem. Theory Comput. 2019, 15 (1): 448-455.
[10.1021/acs.jctc.8b00908](http://dx.doi.org/10.1021/acs.jctc.8b00908).

    @article{schutt2023schnetpack,
        author = {Sch{\"u}tt, Kristof T. and Hessmann, Stefaan S. P. and Gebauer, Niklas W. A. and Lederer, Jonas and Gastegger, Michael},
        title = "{SchNetPack 2.0: A neural network toolbox for atomistic machine learning}",
        journal = {The Journal of Chemical Physics},
        volume = {158},
        number = {14},
        pages = {144801},
        year = {2023},
        month = {04},
        issn = {0021-9606},
        doi = {10.1063/5.0138367},
        url = {https://doi.org/10.1063/5.0138367},
        eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0138367/16825487/144801\_1\_5.0138367.pdf},
    }
    @article{schutt2019schnetpack,
        author = {Sch{\"u}tt, Kristof T. and Kessel, Pan and Gastegger, Michael and Nicoli, Kim A. and Tkatchenko, Alexandre and Müller, Klaus-Robert},
        title = "{SchNetPack: A Deep Learning Toolbox For Atomistic Systems}",
        journal = {Journal of Chemical Theory and Computation},
        volume = {15},
        number = {1},
        pages = {448-455},
        year = {2019},
        doi = {10.1021/acs.jctc.8b00908},
        URL = {https://doi.org/10.1021/acs.jctc.8b00908},
        eprint = {https://doi.org/10.1021/acs.jctc.8b00908},
    }



## Acknowledgements

CLI and hydra configs for PyTorch Lightning are adapted from this template: [![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)


## References

* [1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.
*Quantum-chemical insights from deep tensor neural networks.*
Nature Communications **8**. 13890 (2017) [10.1038/ncomms13890](http://dx.doi.org/10.1038/ncomms13890)

* [2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
*SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.*
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [Paper](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)

* [3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
*SchNet - a deep learning architecture for molecules and materials.*
The Journal of Chemical Physics 148(24), 241722 (2018) [10.1063/1.5019779](https://doi.org/10.1063/1.5019779)

* [4] K. T. Schütt, O. T. Unke, M. Gastegger
*Equivariant message passing for the prediction of tensorial properties and molecular spectra.*
International Conference on Machine Learning (pp. 9377-9388). PMLR, [Paper](https://proceedings.mlr.press/v139/schutt21a.html).

* [5] N. W. A. Gebauer, M. Gastegger, S. S. P. Hessmann, K.-R. Müller, K. T. Schütt
*Inverse design of 3d molecular structures with conditional generative neural networks.*
Nature Communications **13**. 973 (2022) [10.1038/s41467-022-28526-y](https://doi.org/10.1038/s41467-022-28526-y)

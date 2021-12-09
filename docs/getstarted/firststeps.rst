===========
First steps
===========

SchNetPack aims to provide accessible atomistic neural networks
that can be trained and applied out-of-the-box, while still being
extensible to custom atomistic architectures.

Installation
============

.. _requirement:

Requirements
^^^^^^^^^^^^

* `Python <http://www.python.org/>`_ (>=3.8)
* `PyTorch <https://pytorch.org/docs/stable/index.html>`_ (>=1.9)
* `PyTorchLightning <https://www.pytorchlightning.ai/>`_ (>=1.4.5)
* `Hydra <https://hydra.cc/>`_ (>=1.1.0)
* `ASE <https://wiki.fysik.dtu.dk/ase/index.html>`_ (>=3.21)

..
    Installing using pip
    ^^^^^^^^^^^^^^^^^^^^
    .. highlight:: bash


    The simplest way to install SchNetPack is through pip which will automatically get the source code from PyPI_::

        $ pip install --upgrade schnetpack

    Now, once all the requirements are satisfied, you should be ready to use SchNetPack.


Building from source
^^^^^^^^^^^^^^^^^^^^

You can also install the most recent code from our repository::

   $ git clone https://github.com/atomistic-machine-learning/schnetpack.git <dest_dir>
   $ cd <dest_dir>

Install both requirements and SchNetPack::

   $ pip install -r requirements.txt
   $ pip install .

You are ready to go!


Visualization with Tensorboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SchNetPack supports multiple logging backends over PyTorch Lightning.
The default logger is Tensorboard, which can be installed via::

   $ pip install tensorboard


Training pre-defined models
===========================

The best place to get started is training a SchNetPack model on a common benchmark dataset via the command line
interface (CLI).
When installing SchNetPack, the training script ``spktrain`` is added to your PATH.
The CLI is based on `Hydra <https://hydra.cc/>`_ and oriented on the PyTorch Lightning/Hydra template that can be found
`here <https://github.com/ashleve/lightning-hydra-template>`_.
This enables a flexible configuration of the model, data and training process.
To fully take advantage of these features, it might be helpful for have a look at the Hydra and PyTorch Lightning docs.


Example 1: QM9
^^^^^^^^^^^^^^

In the following, we focus on using the CLI to train on the QM9 dataset, but the same
procedure applies for the other benchmark datasets as well.
First, create a working directory, where all data and runs will be stored::

    $ mkdir spk_workdir
    $ cd spk_workdir

Them, the training of a SchNet model with default settings for QM9 can be started by::

   $ spktrain experiment=qm9

The script prints the defaults for the experiment config ``qm9``.
The dataset will be downloaded automatically to ``spk_workdir/data``, if it does not exist yet.
Then, the training will be started.

All values of the config can be changed from the command line, including the directories for run and data.
By default, the model is stored in a directory with a unique run id hash as a subdirectory of ``spk_workdir/runs``.
This can be changed as follows::

   $ spktrain experiment=qm9 run.data_dir=/my/data/dir run.path=~/all_my_runs run.id=this_run

If you call ``spktrain experiment=qm9 --help``, you can see the full config with all the parameters
that can be changed.
Nested parameters can be changed as follows::

   $ spktrain experiment=qm9 data_dir=<path> data.batch_size=64

Hydra organizes parameters in config groups which allows hierarchical configurations consisting of multiple
yaml files. This allows to easily change the whole dataset, model or representation.
For instance, changing from the default SchNet representation to PaiNN, use::

   $ spktrain experiment=qm9 data_dir=<path> model/representation=painn

It is a bit confusing at first when to use "." or "/". The slash is used, if you are loading a preconfigured config
group, while the dot is used changing individual values. For example, the config group "model/representation"
corresponds to the following part of the config: ::

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

If you would want to additionally change some value of this group, you could use: ::

    $ spktrain experiment=qm9 data_dir=<path> model/representation=painn model.representation.n_interactions=5

For more details on config groups, have a look at the
`Hydra docs <https://hydra.cc/docs/next/tutorials/basic/your_first_app/config_groups>`_.


Example 2: Potential energy surfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example above uses :class:`AtomisticModel` internally, which is a
:class:`pytorch_lightning.LightningModule`, to predict single properties.
The following example will use the same class to predict potential energy surfaces,
in particular energies with the appropriate derivates to obtain forces and stress tensors.
This works since the pre-defined configuration for the MD17 dataset,
provided from the command line by ``experiment=md17``, is selecting the representation and output modules that
:class:`AtomisticModel` is using.
A more detailed description of the configuration and how to build your custom configs can be
found :ref:`here <configs>`.

The ``spktrain`` script can be used to train a model for a molecule from the MD17 datasets::

   $ spktrain experiment=md17 data.molecule=uracil

In the case of MD17, reference calculations of energies and forces are available.
Therefore, one needs to set weights for the losses of those properties.
The losses are defined as part of output definitions in the ``task`` config group: ::

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

For a training on *energies* and *forces*, we recommend to put a stronger
weight on the loss of the force prediction during training.
By default, the loss weights are set to 0.005 for the energy and 0.995 for forces.
This can be changed as follow::

    $ spktrain experiment=md17 data.molecule=uracil task.outputs.0.loss_weight=0.005 \
        task.outputs.1.loss_weight=0.995


Logging
^^^^^^^
Beyond the output of the command line, SchNetPack supports multiple logging backends over PyTorch Lightning.
By default, the Tensosboard logger is activated.
If TensorBoard is installed, the results can be shown by calling::

    $ tensorboard --logdir=<rundir>

Furthermore, SchNetPack comes with configs for a CSV logger and `Aim <https://github.com/aimhubio/aim>`_.
These can be selected as follows::

   $ spktrain experiment=md17 logger=csv


References
==========

.. [#schnetpack] K.T. Schütt, P. Kessel, M. Gastegger, K.A. Nicoli, A. Tkatchenko, K.-R. Müller.
   `SchNetPack: A Deep Learning Toolbox For Atomistic Systems <https://doi.org/10.1021/acs.jctc.8b00908>`_.
   Journal of Chemical Theory and Computation **15** (1), pp. 448-455. 2018.

.. [#schnet1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.
   `Quantum-chemical insights from deep tensor neural networks <https://www.nature.com/articles/ncomms13890>`_
   Nature Communications **8** (13890). 2017.

.. [#schnet2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
   `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions
   <http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions>`_
   Advances in Neural Information Processing Systems **30**, pp. 992-1002. 2017.

.. [#schnet3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
   `SchNet - a deep learning architecture for molecules and materials <https://aip.scitation.org/doi/10.1063/1.5019779>`_
   The Journal of Chemical Physics **148** (24), 241722, 2018.

.. [#painn1a] Schütt, Unke, Gastegger:
   Equivariant message passing for the prediction of tensorial properties and molecular spectra.
   ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

.. [#qm] `Quantum-Machine.org <http://www.quantum-machine.org/data>`_

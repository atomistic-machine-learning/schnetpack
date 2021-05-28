SchNetPack aims to provide accessible atomistic neural networks
that can be trained and applied out-of-the-box, while still being
extensible to custom atomistic architectures.

============
Installation
============

.. _requirement:

Requirements
^^^^^^^^^^^^

* `Python <http://www.python.org/>`_ (>=3.6)
* `PyTorch <https://pytorch.org/docs/stable/index.html>`_ (>=1.7)
* `PyTorchLightning <https://www.pytorchlightning.ai/>`_ (>=1.3.3)
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

You can directly build a path from source through git clone. To do so, just type::

   $ git clone https://github.com/atomistic-machine-learning/schnetpack.git <dest_dir>

then move in the new directory ``<dest_dir>``::

   $ cd <dest_dir>

install both requirements and schnetpack::

   $ pip install -r requirements.txt
   $ pip install .

and run tests to be sure everything runs as expected::

   $ pytest

Once that's done, you are ready to go!


Visualization with Tensorboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SchNetPack supports multiple logging backends over PyTorch Lightning.
The default logger is Tensorboard, which can be installed via::

   $ pip install tensorboard


======================
Command line interface
======================

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
procedure applies for the other benchmark datasets as well. The training can be
started using::

   $ spktrain +experiment=qm9 data_dir=<path>

This will print the defaults for the experiment config ``qm9`` and set the data directory to the chosen location.
The dataset will be downloaded automatically if it does not exist there.
Then, the training will be started.

All values of the config can be changed from the command line.
For example, the model will be stored in a directory with a unique run id as a subdirectory of the
current working directory which is by default called ``runs``.
This can be changed as follows::

   $ spktrain +experiment=qm9 data_dir=<path> run_dir=~/all_my_runs run_id=this_run

If you call ``spktrain +experiment=qm9 --help``, you can see the full config with all the parameters
that can be changed.
Nested parameters can be changed as follows::

   $ spktrain +experiment=qm9 data_dir=<path> data.batch_size=64

Hydra organizes parameters in config groups which allows hierarchical configurations consisting of multiple
yaml files. This allows to easily change the whole dataset, model or representation.
For instance, changing from the default SchNet representation to PaiNN, use::

   $ spktrain +experiment=qm9 data_dir=<path> model/representation=painn

For more details on config groups, have a look at the
`Hydra docs <https://hydra.cc/docs/next/tutorials/basic/your_first_app/config_groups>`_.


Example 2: Potential energy surfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example above uses the :class:`SinglePropertyModel` internally, which is a
:class:`pytorch_lightning.LightningModule` for predicting single properties.
The following example will use the PESModel, which can be used for predicting potential energy surfaces,
in particular energies with the appropriate derivates to obtain forces and stress tensors.
For more details on the available models, see :ref:`here<schnetpack.model>`

The ``spktrain`` script can be used to train a model for a molecule from the MD17 datasets::

   $ spktrain data_dir=<path> +experiment=md17 data.molecule=uracil

In the case of MD17, reference calculations of energies and forces are available.
Therefore, one needs to set weights for the losses of those properties.
For a training on *energies* and *forces*, we recommend to put a stronger
weight on the loss of the force prediction during training.
By default, the loss weights are set to 0.05 for the energy and 0.95 for forces.
This can be changed as follow::

    $ spktrain data_dir=<path> +experiment=md17 data.molecule=uracil model.output.energy.loss_weight=0.01 \
        model.output.forces.loss_weight=0.99



===============
Representations
===============

SchNetPack currently supports SchNet, PaiNN and (w)ACSF.

SchNet
^^^^^^

SchNet [#schnet1]_ [#schnet2]_ [#schnet3]_ is an end-to-end deep neural network architecture based on continuous-filter convolutions.
It follows the deep tensor neural network framework, i.e. atom-wise representations are constructed by starting from
embedding vectors that characterize the atom type before introducing the configuration of the system by a series of
interaction blocks.

PaiNN
^^^^^

PaiNN [#painn1]_ is the successor to SchNet, overcoming limitations of invariant representations
by using equivariant representations.
It improves over previous networks in terms of accuracy and/or data efficiency.

ACSF & (w)ACSF
^^^^^^^^^^^^^^

ACSFs [#wacsf1]_ [#wacsf2]_  describe the local chemical environment around a central atom via a combination of radial and angular
distribution functions. Those model come from Behler–Parrinello networks, based on atom centered symmetry functions (ACSFs).
Moreover, wACSF comes as an extensions of this latest. It uses weighted atom-centered symmetry functions (wACSF).
Whereas for SchNet, features are learned by the network, for ACSFs (and wACSFs) we need to introduce some handcrafted
features before training.

==================
Benchmark Datasets
==================

SchNetPack provides convenient interfaces to popular benchmark datasets in order to train and test models.

QM9
^^^
The ``qm9`` dataset contains 133,885 organic molecules with up to nine heavy atoms from C, O, N and F [#qm9]_.

MD17
^^^^
The ``md17`` dataset allows to do molecular dynamics of small molecules containing molecular forces [#qm]_.

ANI1
^^^^
The ``ani1`` dataset consists of more than 20 million conformations for 57454 small organic molecules from C, O and N [#ani]_.

Materials Project
^^^^^^^^^^^^^^^^^
A repository of bulk crystals containing atom types ranging across the whole periodic table up to Z = 94 [#mp]_.

OMDB
^^^^
The ``omdb`` dataset contains data from Organic Materials Database (OMDB) of bulk organic crystals.
This database contains DFT (PBE) band gap (OMDB-GAP1 database) for 12500 non-magnetic materials.
The registration to the OMDB is free for academic users. [#omdb]_.



==========
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

.. [#painn1] Schütt, Unke, Gastegger:
   Equivariant message passing for the prediction of tensorial properties and molecular spectra.
   ICML 2021 (to appear)

.. [#wacsf1] M. Gastegger, L. Schwiedrzik, M. Bittermann, F. Berzsenyi, P. Marquetand.
   `wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials <https://aip.scitation.org/doi/10.1063/1.5019667>`_
   The Journal of Chemical Physics **148** (24), 241709. 2018.

.. [#wacsf2] J. Behler, M. Parrinello.
   `Generalized neural-network representation of high-dimensional potential-energy surfaces <https://link.aps.org/doi/10.1103/PhysRevLett.98.146401>`_
   Physical Review Letters **98** (14), 146401. 2007.

.. [#qm9] R. Ramakrishnan, P.O. Dral, M. Rupp, O. A. von Lilienfeld.
   `Quantum chemistry structures and properties of 134 kilo molecules <https://doi.org/10.1038/sdata.2014.22>`_
   Scientific Data **1** (140022). 2014.

.. [#ani] J.S. Smith, O. Isayev, A.E. Roitberg.
    `ANI-1, A data set of 20 million calculated off-equilibrium conformations for organic molecules. <https://doi.org/10.1038/sdata.2017.193>`_
    Scientific Data **4** (170193). 2017.

.. [#qm] `Quantum-Machine.org <http://www.quantum-machine.org/data>`_

.. [#omdb] `Organic Materials Database (OMDB) <https://omdb.mathub.io/dataset/>`_

.. [#mp] A. Jain, S.P. Ong, G. Hautier, W. Chen, W.D. Richards, S. Dacek,
    S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson.
    `The Materials Project: A materials genome approach to accelerating materials innovation <https://doi.org/10.1063/1.4812323>`_
    APL Materials **1** (1), 011002. 2013.

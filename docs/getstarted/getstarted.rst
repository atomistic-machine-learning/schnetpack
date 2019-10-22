SchNetPack aims to provide accessible atomistic neural networks
that can be trained and applied out-of-the-box, while still being
extensible to custom atomistic architectures.

============
Installation
============

.. _requirement:

Requirements
^^^^^^^^^^^^

* Python_ (>=3.6)
* NumPy_
* Pytorch_ (>=1.1)
* ASE_ (>=3.16)
* TensorboardX_ (For improved training visualization)
* h5py_
* tqdm_
* PyYaml_

.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _Pytorch: https://pytorch.org/docs/stable/index.html#
.. _TensorboardX: https://github.com/lanpa/tensorboardX
.. _h5py: https://www.h5py.org
.. _ASE: https://wiki.fysik.dtu.dk/ase/index.html
.. _tqdm: https://github.com/tqdm/tqdm
.. _PyYaml: https://pyyaml.org/


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


.. note::

   If your OS doesn't have ``numpy``, ``pytorch``, and ``ase`` packages
   installed, and the previous command didn't work for you, you can install those requirements through::

        $ pip install --upgrade --user numpy torch ase

Visualization with Tensorboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While SchNetPack is based on PyTorch, it is possible to use Tensorboard, which comes with TensorFlow,
to visualize the learning progress.
While this is more convenient for visualization, you need to install TensorBoard
in order to view the event files SchNetPack produces.
Even though there is a standalone version, the easiest way to get Tensorboard is by installing TensorFlow, e.g. using pip::

   $ pip install tensorflow

===============================
Scripts for benchmark data sets
===============================

 The best place to start is training a SchNetPack model on a common benchmark dataset.
 Scripts for common datasets are provided by SchNetPack and inserted into your PATH during installation.

The example script allows to train and evaluate both SchNet and wACSF neural networks.
In the following, we focus on using the script for the QM9 dataset, but the same
procedure applies for the other benchmark datasets as well. The training can be
started using::

   $ spk_run.py train <schnet/wacsf> <qm9/ani1/...> <dbpath> <modeldir> --split num_train num_val [--cuda]

where num_train and num_val need to be replaced by the number of training and validation datapoints respectively.
You can choose between SchNet and wACSF networks and have to provide a directory to store the model and the location
of the dataset, which has to be a ASE DB file (``.db`` or ``.json``). It will be downloaded automatically
if it does not exist.

.. note::
   Please be warned that the ANI-1 dataset is huge (more than 20gb).


With the ``--cuda`` flag, you can activate GPU training.
The default hyper-parameters should work fine, however, you can change them through command-line arguments.
Please refer to the help at the command line::

   $ spk_run.py train <schnet/wacsf> --help

The training progress will be logged in ``<modeldir>/log``. The default is a basic logging with **CSV** files.
Advanced logging with **TensorBoard** event files can be activated using ``--logger tensorboard`` (see `above <#visualization-with-tensorboard>`_).

To evaluate the trained model that showed the best validation error during training (i.e., early stopping), call::

   $ spk_run.py eval <datapath> <modeldir> [--split train val test] [--cuda]

which will write a result file ``evaluation.txt`` into the model directory.

.. tip::

   ``<modeldir>`` should point to a directory in which a pre-trained model is stored. As an argument for the --split
   flag for evaluation you should choose among one of training, validation or test subsets.

==================================
Using Scripts with custom Datasets
==================================

The script for benchmark data can also train a model on custom data sets, by using::

   $ spk_run.py train <schnet/wacsf> custom <dbpath> <modeldir> --split num_train num_val --property your_property [--cuda]

Depending on your data you will need to define some settings that have already been
pre-selected for the benchmark data. In order to show how to use the script
on arbitrary data sets, we will use the MD17 data set and treat it as a custom data
set. First of all we need to define the property that we want to use for training.
In this example we will train the model on the *energy* labels. If we want to use the
*forces* during training, we need to add the ``--derivative`` argument and also set
``--negative_dr``, because the gradient of the energy predictions corresponds to the
negative forces.

Defining Output Modules
^^^^^^^^^^^^^^^^^^^^^^^

Since energy is a property that depends on the total number of atoms
we select ``--aggregation_mode sum``. Other properties (e.g. homo, lumo, ...) do not
depend on the total number of atoms and will therefore use the mean aggregation mode.
While most properties should be trained with the ``spk.nn.Atomwise`` output module
which is selected by default, some properties require special output modules.
Models using the ``spk.SchNet`` representation support ``dipole_moment``,
``electronic_spatial_extent``, ``ploarizability`` and ``isotropic_polarizability``.
Note that if your model is based on the ``spk.BehlerSFBlock`` representation you need
to select between ``elemental_atomwise`` and ``elemental_dipole_moment``. The output
module selection is defined with ``--output_module
<atomwise/elemental_atomwise/dipole_moment/...>``.

Loss Tradeoff
^^^^^^^^^^^^^

It can be useful to define a tradeoff between multiple properties of an output
module. For a training on *energies* and *forces*, we recommend to put a stronger
weight on the loss of the force prediction during training. Therefore one can add the
tradeoff parameter ``--rho`` with its arguments as ``key=value``. If no weight is
selected for a key, it gets the weight 1. Afterwards all weights are divided by the
total weight. For including 90% of the force loss and 10% of the energy loss, the
command is ``--rho property=0.1 derivative=0.9``. You can also use the *stress* and
the *contributions* properties during training.

Summary
^^^^^^^

The final command for the MD17 example would be::

   $ spk_run.py train <schnet/wacsf> custom <dbpath> <modeldir> --split num_train num_val --property energy --derivative forces --negative_dr --rho property=0.1 derivative=0.9 --aggregation_mode sum [--cuda]

The command for training a QM9-like data set on dipole moments would be::

   $ spk_run.py train <schnet/wacsf> custom <dbpath> <modeldir> --split num_train num_val --property dipole_moment --output_module dipole_moment --aggregation_mode sum [--cuda]

The evaluation of the trained model uses the same commands as any pre-implemented
data set.

=================================
Using Argument Files for Training
=================================

An argument file with all training arguments is created at the beginning of every
training session and can be found at *<modeldir>/args.json*. These argument
files can be modified and used for new training sessions. In order to build a file
with default settings run::

   $ spk_run.py train <schnet/wacsf> custom <dbpath> <modeldir>

This will create the <modeldir> which contains the argument file, while the training
session will fail because ``--split`` is not selected. You can now modify the
arguments and use them for training::

   $ spk_run.py from_json <modeldir>/args.json

================
Supported Models
================

SchNetPack currently supports SchNet and (w)ACSF.

SchNet
^^^^^^

SchNet [#schnet1]_ [#schnet2]_ [#schnet3]_ is an end-to-end deep neural network architecture based on continuous-filter convolutions.
It follows the deep tensor neural network framework, i.e. atom-wise representations are constructed by starting from
embedding vectors that characterize the atom type before introducing the configuration of the system by a series of
interaction blocks.


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

SchNetPack provides convenient interfaces to popular benchmark datasets in order to train and test its model.

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

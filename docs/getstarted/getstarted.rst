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
* Pytorch_ (>=0.4.1)
* Ase_ (>=3.16)
* TensorboardX_ (tensorboard logging instead of .csv)
* h5py_
* tqdm_
* PyYaml_

.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _Pytorch: https://pytorch.org/docs/stable/index.html#
.. _TensorboardX: https://github.com/lanpa/tensorboardX
.. _h5py: https://www.h5py.org
.. _Ase: https://wiki.fysik.dtu.dk/ase/index.html
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
   $ python setup.py install

and run tests::

   $ pytest

Once that's done, you are ready to go!


.. note::

   The best place to start is training a SchNetPack model on a common benchmark dataset.
   Scripts for common datasets are provided by SchNetPack and inserted into your PATH during installation.


.. tip::

   If your OS doesn't have ``numpy``, ``pytorch``, and ``ase`` packages
   installed, and the previous command didn't work for you, you can install those requirements through::

        $ pip install --upgrade --user numpy torch ase

================
Provided Scripts
================

QM9 & ANI1
^^^^^^^^^^

The QM9 and ANI1 example scripts allow to train and evaluate both SchNet and wACSF neural networks.
In the following tutorial we focus on the qm9 scripts, but the same procedure applies for the
``schnetpack_ani1.py`` script as well. The training can be started using::

   $ schnetpack_qm9.py train <schnet/wacsf> <dbpath> <modeldir> --split num_train num_val [--cuda]

where num_train and num_val need to be replaced by the number of training and validation datapoints respectively.
You can choose between SchNet and wACSF networks and have to provide a directory to store the model and the location
of the dataset, which has to be a ASE DB file (``.db`` or ``.json``). It will be downloaded automatically
if it does not exist.

.. note::
   Please be warned that the ANI-1 dataset its huge (more than 20gb).


With the ``--cuda`` flag, you can activate GPU training.
The default hyper-parameters should work fine, however, you can change them through command-line arguments.
Please refer to the help at the command line::

   $ schnetpack_qm9.py train <schnet/wacsf> --help

The training progress will be logged in ``<modeldir>/log``, either as **CSV**
(default) or as **TensorBoard** event files. For the latter, you need to install TensorBoard in order to view the event files.
This first comes by installing the version included in TensorFlow::

   $ pip install tensorflow

To evaluate the trained model that showed the best validation error during training (i.e., early stopping), call::

   $ schnetpack_qm9.py eval <schnet/wacsf> <dbpath> <modeldir> [--split train val test] [--cuda]

which will write a result file ``evaluation.txt`` into the model directory.

.. tip::

   ``<modeldir>`` should point to a directory in which a pre-trained model is stored. As an argument for the --split
   flag for evaluation you should choose among one of training, validation or test subsets.

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

ANI1
^^^^
The ``ani1`` dataset consists of more than 20 million conformations for 57454 small organic molecules from C, O and N [#ani]_.

MD17
^^^^
The ``md17`` dataset allows to do molecular dynamics of small molecules containing molecular forces [#qm]_.

..
    ISO17
    ^^^^^
    The ``iso17`` dataset contains data for molecular dynamics of C7 O2 H10 isomers.
    It contains 129 isomers with 5000 conformational geometries and their corresponding energies and forces [#qm]_.

Materials Project
^^^^^^^^^^^^^^^^^
A repository of bulk crystals containing atom types ranging across the whole periodic table up to Z = 94 [#mp]_.


==========
References
==========

.. [#schnetpack] Schnetpack -  Add reference once paper is accepted

.. [#schnet1] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.
   `Quantum-chemical insights from deep tensor neural networks <https://www.nature.com/articles/ncomms13890>`_
   Nature Communications 8. 13890 (2017)

.. [#schnet2] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
   `SchNet: A continuous-filter convolutional neural network for modeling quantum interactions
   <http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions>`_
   Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017)

.. [#schnet3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
   `SchNet - a deep learning architecture for molecules and materials <https://aip.scitation.org/doi/10.1063/1.5019779>`_
   The Journal of Chemical Physics 148(24), 241722 (2018)

.. [#wacsf1] M. Gastegger, L. Schwiedrzik, M. Bittermann, F. Berzsenyi, P. Marquetand.
   `wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials <https://aip.scitation.org/doi/10.1063/1.5019667>`_
   The Journal of Chemical Physics, 148(24), 241709. (2018)

.. [#wacsf2] J. Behler, M. Parrinello.
   `Generalized neural-network representation of high-dimensional potential-energy surfaces <https://link.aps.org/doi/10.1103/PhysRevLett.98.146401>`_
   Physical Review Letters, 98(14), 146401. (2007)

.. [#qm9] R. Ramakrishnan, P.O. Dral, M. Rupp, O. A. von Lilienfeld.
   `Quantum chemistry structures and properties of 134 kilo molecules <https://doi.org/10.1038/sdata.2014.22>`_
   Scientific data, 1, 140022. (2014)

.. [#ani] J.S. Smith, O. Isayev, A.E. Roitberg.
    `ANI-1, A data set of 20 million calculated off-equilibrium conformations for organic molecules. <https://doi.org/10.1038/sdata.2017.193>`_
    Scientific data, 4, 170193. (2017)

.. [#qm] `Quantum-Machine.org <http://www.quantum-machine.org/data>`_

.. [#mp] A. Jain, S.P. Ong, G. Hautier, W. Chen, W.D. Richards, S. Dacek,
    S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. Persson.
    `The Materials Project: A materials genome approach to accelerating materials innovation <https://doi.org/10.1063/1.4812323>`_
    APL Materials 1(1), 011002 (2013)

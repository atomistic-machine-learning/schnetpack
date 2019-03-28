.. _tut etha:

Tutorial: Using SchNetPack with custom Data
===========================================

This tutorial will explain how to use SchNetPack for training a SchNet model
on custom datasets and how the trained model can be used for further
experiments. We will start by creating a new environment for the installation of
SchNetPack. Therefore open a new terminal and use::

    conda create -n schnet python

to create a new environment and activate it with::

    source activate schnet

SchNetPack can directly be installed into the new environment by using ``pip``::

    pip install schnetpack

For this tutorial we will create a new folder. Therefore ``cd`` to your
desired location and create a new directory::

    mkdir schnet_tutorial
    cd schnet_tutorial


.. _tut etha prep:

Prepare the Data for Training
-----------------------------

This tutorial will use an ethanol dataset which can be downloaded
`here <http://quantum-machine.org/gdml/data/xyz/ethanol_dft.zip>`_. First of all you
will need to create a data directory in the tutorial folder. Therefore run::

    mkdir data

Move the downloaded dataset to the data folder and unzip it with::

    unzip data/ethanol_dft.zip

The dataset should be provided as an xyz-file. In order to use the dataset for
training you will need to use the parsing script. This will convert the xyz-file to
an ``ase.db`` which is suitable for SchNetPack. Run the script with::

    spk_parse.py with forces file_path=data/ethanol.xyz db_path=data/ethanol.db

You will end up with a new file in your data directory.


.. _tut etha train:

Train a Model on the Ethanol Dataset
------------------------------------

TODO


.. _tut etha monitoring:

Monitoring your Training Session
--------------------------------


We recommend to use TensorBoard for monitoring your training session. Therefore
you will need to open a new terminal window and ``cd` to the directory of this
tutorial. Activate your environment and install TensorBoard with::

    pip install tensorflow

In order to run the TensorBoard use::

    tensorboard --logdir=training

Your terminal will display a message which contains a URL to your board. Copy it into
your browser and the TensorBoard should show up:

.. |TensorBoard| image:: ../pictures/tensorboard.png
  :width: 600
  :alt: Screenshot of a running TensorBoard

|TensorBoard|


.. _tut etha eval:

Evaluating Datasets with a trained Model
----------------------------------------

When the training session has ended you can use the trained model to predict
properties for other datasets. You will find a small dataset with ethanol molecules here
xxrefxx. Download the snippet and store it in the data directory of your tutorial
folder. If you open the dataset you will notice that the molecules do not contain any
values for energy or forces. The missing properties will be predicted by using the
trained model::

    spk_eval.py with in_path=data/ethanol_missing.xyz out_path=data/ethanol_predicted.db model_dir=training_ethanol

The script will look inside the ``model_dir`` and find the best model of the training
session, which will automatically be used for the predictions. You will end up with a
new ``ase.db`` file in your data directory, which contains the energy values in the
*data* column.


.. _tut etha calc:

Using a trained Model as a Calculator for ASE
---------------------------------------------

The trained model can also be used as a calculator for ``ase``. For the purpose of
this tutorial we will write a small example script which reads a molecule from our
test snippet that has been downloaded in section :ref:`tut etha eval` and predict its
properties. Therefore we start with the necessary imports::

    import torch
    from ase.io import read
    from schnetpack.ase_interface import SpkCalculator

Secondly build an ``ase`` calculator from our model. Therefore you will need to load
the model and use the ``SpkCalculator`` class::

    # load model
    path_to_model = 'training_ethanol/best_model'
    model = torch.load(path_to_model)
    # build calculator
    calculator = SpkCalculator(model, device='cpu')

Afterwards you will need to load an ``ase.Atoms`` object from the database and set
the calculator::

    atoms = read('ethanol_missing.xyz')
    # set calculator
    atoms.set_calculator(calculator)

At last just print the result::

    print('energy', atoms.get_total_energy())
    print('forces', atoms.forces())

Execute the script and you should see the energy prediction.
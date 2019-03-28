.. _tutorial qm9:

Tutorial: Using SchNetPack with QM9
===================================

This tutorial will explain how to use SchNetPack for training a SchNet model
on the QM9 dataset and how the trained model can be used for further
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


.. _tut qm9 train::

Training a Model on QM9
-----------------------

TODO


.. _monitoring tut qm9:

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


.. _tut qm9 eval:

Evaluating Datasets with a trained Model
----------------------------------------

When the training session has ended you can use the trained model to predict
properties for other datasets. You will find a small database with QM9 molecules here
xxrefxx. Download the snippet and store it in the data directory of your tutorial
folder. In order to test the trained model, the energy labels of the molecules inside
the database have been removed. For predicting the missing labels you can use the
evaluation script::

    spk_eval.py with in_path=data/qm9_missing.db out_path=data/qm9_predicted.db model_dir=training

The script will look inside the ``model_dir`` and find the best model of the training
session, which will automatically be used for the predictions. You will end up with a
new ``ase.db`` file in your data directory, which contains the energy values in the
*data* column.


.. _tut qm9 calc:

Using a trained Model as a Calculator for ASE
---------------------------------------------

The trained model can also be used as a calculator for ``ase``. For the purpose of
this tutorial we will write a small example script which predicts the energy of an
``ase.Atoms`` object. For this tutorial we will predict the missing energy value of
the first atom in the database snippet that has been downloaded in :ref:`tut qm9 eval`.
First of all you will need to open your favorite editor and create a new Python file.
The file should be stored at your tutorial directory. Start the file by doing the
necessary imports::

    import torch
    from ase.db import connect
    from schnetpack.ase_interface import SpkCalculator

Secondly build an ``ase`` calculator from our model. Therefore you will need to load
the model and use the ``SpkCalculator`` class::

    # load model
    path_to_model = 'training/best_model'
    model = torch.load(path_to_model)
    # build calculator
    calculator = SpkCalculator(model, device='cpu')

Afterwards you will need to load an ``ase.Atoms`` object from the database and set
the calculator::

    # connect to database
    conn = connect('data/qm9_missing.db')
    # get first molecule
    atoms = conn.get_atoms(1)
    # set calculator
    atoms.set_calculator(calculator)

At last just print the result::

    print('energy', atoms.get_total_energy())

Execute the script and you should see the energy prediction.
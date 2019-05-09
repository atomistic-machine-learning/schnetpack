.. _tutorial qm9:

Tutorial: Using SchNetPack with QM9
===================================

This tutorial will explain how to use SchNetPack for training a SchNet model
on the QM9 dataset and how the trained model can be used for further
experiments. For this tutorial we will create a new folder. Therefore ``cd`` to your
desired location and create a new directory::

    mkdir schnet_tutorial
    cd schnet_tutorial


.. _tut qm9 train::

Training a Model on QM9
-----------------------


This example trains Schnet model on the energy of QM9 molecules. Here we go through
various sections of example script.

We start by importing modules used in the example::

    import os
    import logging
    from torch.optim import Adam
    import schnetpack as spk
    from schnetpack.datasets import QM9
    from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
    from schnetpack.metrics import MeanAbsoluteError
    from schnetpack.metrics import mse_loss

and adjust the basic setting of the model as described below::

    # basic settings
    model_dir = "qm9_model"  # directory that will be created for storing model
    properties = [QM9.U0]  # properties used for training

this is followed by making a directory to store the model::

    os.makedirs(model_dir)

Train, Validation & Test Sets
.............................

Then, we load the database and the required properties given as a list of strings
(which should match the name of properties used in database file)::

    dataset = QM9("data/qm9.db", properties=[QM9.U0])

in the next step, the dataset is into train, validation and test sets. The
corresponding indices are stored in split.npz file::

    train, val, test = spk.train_test_split(
        data=dataset,
        num_train=1000,
        num_val=100,
        split_file=os.path.join(model_dir, "split.npz"),
    )

the datasets are then used to build the dataloaders. The dataloaders provide batches
of our dataset for the training session::

    train_loader = spk.AtomsLoader(train, batch_size=batch_size)
    val_loader = spk.AtomsLoader(val, batch_size=batch_size)


Model Representation
....................

The `Schnet` network is build for learning the representation by assigning the optional
argument `n_interactions`. To further customize the network see API Documentation::

    representation = spk.SchNet(n_interactions=6)


Model Network
.............

The ``Atomwise`` network is build for accumulating atom-wise property predictions::

    output_modules = [
        spk.Atomwise(
            property=QM9.U0,
            mean=means[QM9.U0],
            stddev=stddevs[QM9.U0],
        )
    ]

The `model` is built by joining the representation network and output networks::

    model = spk.AtomisticModel(representation, output_modules)


Monitor Train Process: Hooks
............................

You can use `hooks` to monitor or control the progress of your training session. For
this tutorial we will use a ``CSVHook`` for monitoring and the ``ReduceLROnPlateauHook``
automatically reduces the learning rate if the training session does not improve any
further. To learn more about customizing hooks see API Documentation::

    metrics = [MeanAbsoluteError(p, p) for p in properties]
    hooks = [
        CSVHook(log_path=model_dir, metrics=metrics),
        ReduceLROnPlateauHook(optimizer)
    ]


Train Model
...........

Before, we train the model, the loss function is defined for the properties we are training on.
This loss function measures the discrepancy between batch predictions and actual results::

    loss = mse_loss(properties)

Now, the model can be trained for the given number of epochs on the specified device.
This will save the best_model as well as checkpoints in the model directory specified above.
To learn more about customizing trainer see the API Documentation::

    trainer = Trainer(
        model_dir,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=Adam(params=model.parameters(), lr=1e-4),
        train_loader=train_loader,
        validation_loader=val_loader,
    )
    trainer.train(device="cpu", n_epochs=1000)

.. _tut qm9 monitoring:

Monitoring your Training Session
--------------------------------

We recommend to use TensorBoard for monitoring your training session. Therefore
you will need to open add the ``TensorboardHook`` to the list of hooks::

        TensorboardHook(log_path=model_dir, metrics=metrics)

In order to use the TensorBoard you will need to install ``tensorflow`` in your
environment::

    pip install tensorflow

and ``cd`` to the directory of this tutorial. Make sure that your environment is
activated and run TensorBoard::

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
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

Train a Model on the Ethanol Database
-------------------------------------

This example trains Schnet model on energy and forces of ethanol conformations.
Here we go through various sections of example script.

We start by importing modules used in the example::

    import os
    import logging
    from shutil import rmtree
    from torch.optim import Adam
    from schnetpack.atomistic import AtomisticModel
    from schnetpack.output_modules import Atomwise
    from schnetpack.data import AtomsData, AtomsLoader, train_test_split
    from schnetpack.representation import SchNet
    from schnetpack.train import Trainer, TensorboardHook, CSVHook, ReduceLROnPlateauHook
    from schnetpack.metrics import MeanAbsoluteError
    from schnetpack.utils import loss_fn

and adjust the basic setting of the model as described below::

    db_path = "data/md17/ethanol.db"    # relative path to the database file
    model_dir = "ethanol_model"         # directory that will be created for storing model
    properties = ["energy", "forces"]   # properties used for training
    num_train, num_val = 1000, 100      # number of training and validating samples
    batch_size = 64                     # batch size used in training
    device = "cpu"                      # device used, choose between 'cpu' & 'gpu'

this is followed by making a directory to store the model::

    if os.path.exists(model_dir):
        rmtree(model_dir)
    os.makedirs(model_dir)


Train, Validation & Test Sets
.............................

Then, we load the database and the required properties given as a list of strings
(which should match the name of properties used in database file)::

    dataset = AtomsData(db_path, required_properties=properties)

in the next step, the database in split into train, validation and test and the
corresponding indices are stored in split.npz file::

    train, val, test = train_test_split(
        data=dataset,
        num_train=num_train,
        num_val=num_val,
        split_file=os.path.join(model_dir, "split.npz"),
    )

these indices are used to load train, validation and test data into batches::

    train_loader = AtomsLoader(train, batch_size=batch_size)
    val_loader = AtomsLoader(val, batch_size=batch_size)
    test_loader = AtomsLoader(test, batch_size=batch_size)



Model Representation
....................

The `Schnet` network is build for learning the representation by assigning the optional
argument `n_interactions`. To further customize the network see API Documentation::

    representation = SchNet(n_interactions=6)


Model Network
.............

The `Atomiwise` network is build for accumulating atom-wise property predictions::

    output_modules = [
        Atomwise(
            property="energy",
            derivative="forces",
            mean=means["energy"],
            stddev=stddevs["energy"],
            negative_dr=True,
        )
    ]

The `model` is built by joining the representation network and output networks::

    model = AtomisticModel(representation, output_modules)


Monitor Train Process: Hooks
............................

The `hooks` is built for monitoring training process which is a list of 3 types
of hooks here. To learn more about customizing hooks see API Documentation::

    metrics = [MeanAbsoluteError(p, p) for p in properties]
    logging_hooks = [
        TensorboardHook(log_path=model_dir, metrics=metrics),
        CSVHook(log_path=model_dir, metrics=metrics),
    ]
    scheduling_hooks = [ReduceLROnPlateauHook(patience=25, window_length=3, factor=0.8)]
    hooks = logging_hooks + scheduling_hooks


Train Model
...........

Before, we train the model, the loss function is defined for the properties we are training on.
This loss function measures the discrepancy between batch predictions and actual results::

    loss = loss_fn(properties)

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
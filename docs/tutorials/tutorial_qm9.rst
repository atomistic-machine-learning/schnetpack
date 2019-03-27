.. _tutorial qm9:

Tutorial: Using SchNetPack with QM9
===================================

This tutorial will explain how to use SchNetPack for training a SchNet model
on the QM9 dataset and how the trained model can be used for further
experiments. For this tutorial we will start by creating a new environment
for the installation of SchNetPack. Therefore open a new terminal and use::

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

In order to get started with ``schnetpack`` we will need to train a model.
For this tutorial we will make use of the training script and train a new
model on the QM9 dataset. The script will automatically download the dataset
and create a new directory with the training outputs. Run the script by
calling::

    spk_train.py with model_dir=training dataset.qm9 device=<cpu/cuda>

This will automatically start the training session and store all outputs to
the directory that has been defined with ``model_dir``. Inside the model
directory you will find a file called *csv.log*. You can use the log file to
monitor the progress of your training session. If you train the model on a
CPU it could take a long time, before you see any changes in the log file.
Therefore we recommend training the model on a GPU by setting ``device=cuda``.


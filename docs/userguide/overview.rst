========
Overview
========
.. _overview:

SchNetPack is built so that it can be used from the command line and configured with
config files as well as used as a Python library.
In this section, we will explain the overall structure of SchNetPack, which will be
helpful for both these use cases.
SchNetPack is based on PyTorch and uses `PyTorchLightning <https://www.pytorchlightning.ai/>`_ as a training framework.
This heavily influences the structure described here.
Additionally, `Hydra <https://hydra.cc/>`_ is used to configure SchNetPack for command-line usage,
which will be described in the next chapter.

Data
====
.. currentmodule:: data

SchNetPack currently supports data sets stored in ASE format using
:class:`ASEAtomsData`, but other formats can be added by implementing
:class:`BaseAtomsData`. These classes are compatible with PyTorch dataloaders and
provide an additional interface to store metadata, e.g. property units and
single-atom reference values.

An important aspect are the transforms that can be passed to the data classes. Those
are PyTorch modules that perform preprocessing task on the data *before* batching.
Typically, this is performed on the CPU as part of the multi-processing of PyTorch
dataloaders.
Important preprocessing :class:`Transform`s include removing of offsets from target properties
and calculation of neighbor lists.

Furthermore, we support PyTorch Lightning datamodules through :class:`AtomsDataModule`,
which combines :class:`ASEAtomsData` with code for preparation, setup and partitioning
into train/validation/test splits. We provide specific implementations of
:class:`AtomsDataModule` for several benchmark datasets.


Model
=====
.. currentmodule:: model

A core component of SchNetPack is the :class:`AtomisticModel`, which is the base
class for all models implemented in SchNetPack. It is essentially a PyTorch module with
some additional functionality for specific to atomistic machine learning.

The particular features and requirements are:

* **Input dictionary:**
   To support a flexible interface, each model is supposed to take an input dictionary
   mapping strings to PyTorch tensors and returns a modified dictionary as output.

* **Automatic collection of required dervatives:**
   Each layer that requires derivatives w.r.t to some input, should list them as strings
   in `layer.required_derivatives = ["input_key"]`. The `requires_grad` of the input
   tensor is then set automatically.

.. currentmodule::transform
* **Post-processing:**
   The atomistic model can take a list of non-trainable :class:`Transform`s that are
   used to post-process the output dictionary. These are not applied during training.
   A common use case are energy values that a large offsets and require double
   precision. To be able to still run a single precision model on GPU, one can substract
   the offset from the reference data during a preprocessing stage and then add it
   to the model prediction in post-processing after casting to double.

.. currentmodule:: model
While :class:`AtomisticModel` is a fairly general class, the models provided in
SchNetPack follow a structure defined in the subclass :class:`NeuralNetworkPotential`:

#. **Input modules:**: the input dictionary is sequentially passed to a list of PyTorch
   modules that return a modified dictionary

#. **Representation:**: the input dictionary is passed to a representation module that
   computes atomwise representation, e.g. SchNet or PaiNN. The representation is added
   to the dictionary

#. **Output modules:**: the dictionary is sequentially passed to a list of PyTorch
   modules that store the outputs in the dictionary

Adhering to the structure of :class:`NeuralNetworkPotential` makes it easier to define
config templates with Hydra and it is therefore recommended to subclass it wherever
possible.

Task
====
.. currentmodule:: task

The :class:`AtomisticTask` ties the model, outputs, loss and optimizers together and defines
how the neural network will be trained. While the model is a vanilla PyTorch module,
the task is a :class:`LightningModule` that can be directly passed to the
PyTorch Lightning :class:`Trainer`.

To define an :class:`AtomisticTask`, you need to provide:

* a model as described above

* a list of :class:`ModelOutput` which map output dictionary keys to target properties and assigns a loss function and other metrics

* (optionally) optimizer and learning rate schedulers

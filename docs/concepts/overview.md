# Overview

SchNetPack can be used from the command line, configured with config files, and
used as a Python library. This page explains the overall structure of SchNetPack,
which helps with both. SchNetPack is based on PyTorch and uses
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) as its training
framework, which shapes the structure described here. [Hydra](https://hydra.cc/)
configures SchNetPack for the command line, as described in
[Configuration and CLI](../guides/configs.md).

## Data

```{currentmodule} schnetpack.data
```

SchNetPack stores data sets in ASE databases ({class}`ASEDBStorage`) and reads them
with {class}`AtomsDataset`, structure by structure. Data sets that fetch their raw
data from a remote source subclass it and define a `download` method. These classes
are compatible with PyTorch dataloaders and provide an additional interface to store
metadata, e.g. property units and single-atom reference values.
{class}`InMemoryDataset` holds a data set in memory and returns whole batches.

An important aspect are the transforms that can be passed to the data classes. Those
are PyTorch modules that preprocess the data *before* batching. Typically, this runs
on the CPU as part of the multi-processing of PyTorch dataloaders. Important
preprocessing {class}`~schnetpack.transform.Transform` modules remove offsets from
target properties and compute neighbor lists.

PyTorch Lightning datamodules are supported through
{class}`~schnetpack.lightning.AtomsDataModule`, which combines {class}`AtomsDataset`
with code for preparation, setup and partitioning into train/validation/test splits.
The datamodule also owns the training statistics (per-property mean/std and atom
references): it exposes them via `get_stats` and `get_atomrefs`, initializes all
transforms that require them, and persists computed values next to the split file
so reruns read instead of recompute.

## Model

```{currentmodule} schnetpack.model
```

A core component of SchNetPack is the {class}`AtomisticModel`, the base class for
all models implemented in SchNetPack. It is essentially a PyTorch module with some
additional functionality specific to atomistic machine learning:

- **Input dictionary:** each model takes an input dictionary mapping strings to
  PyTorch tensors and returns a modified dictionary as output.
- **Automatic collection of required derivatives:** each layer that requires
  derivatives with respect to some input lists them as strings in
  `layer.required_derivatives = ["input_key"]`. The `requires_grad` of the input
  tensor is then set automatically.
- **Post-processing:** the atomistic model can take a list of non-trainable
  {class}`~schnetpack.transform.Transform` modules that post-process the output
  dictionary. These are not applied during training. A common use case are energy
  values with large offsets that require double precision. To still run a single
  precision model on GPU, subtract the offset from the reference data during
  preprocessing and add it back to the model prediction in post-processing, after
  casting to double.

While {class}`AtomisticModel` is a fairly general class, the models provided in
SchNetPack follow a structure defined in the subclass {class}`NeuralNetworkPotential`:

1. **Input modules:** the input dictionary is passed through a list of PyTorch
   modules that return a modified dictionary.
2. **Representation:** the input dictionary is passed to a representation module
   that computes atomwise representations, e.g. SchNet or PaiNN. The representation
   is added to the dictionary.
3. **Output modules:** the dictionary is passed through a list of PyTorch modules
   that store the outputs in the dictionary.

Following the structure of {class}`NeuralNetworkPotential` makes it easier to define
config templates with Hydra, so subclass it wherever possible.

## Task

```{currentmodule} schnetpack.objectives
```

The {class}`~schnetpack.lightning.AtomisticTask` ties the model, outputs, loss and
optimizers together and defines how the neural network is trained. While the model
is a plain PyTorch module, the task is a Lightning `LightningModule` that can be
passed directly to the PyTorch Lightning `Trainer`.

To define an {class}`~schnetpack.lightning.AtomisticTask`, you provide:

- a model as described above,
- a list of {class}`ModelOutput`, which map output dictionary keys to target
  properties and assign a loss function and metrics,
- optionally, optimizers and learning rate schedulers.

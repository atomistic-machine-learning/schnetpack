===============
Training faster
===============

With the default settings, small and medium models on molecules (QM9, MD17) keep the
GPU busy only 5-30% of the time: each training step waits for the DataLoader workers
to read, transform and collate the next batch, and for Python to launch the step's
hundreds of kernels. The switches below remove that overhead. Each one can be turned
on or off on its own; the combination for a given training is in the next section.

Switches
========

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - switch
     - ``spktrain`` / Python
     - what it does
   * - DataLoader workers
     - ``data.num_workers=8``
       ``AtomsDataModule(num_workers=8)``
     - read and transform structures in 8 processes. With 2 workers a QM9 batch of
       100 took 55 ms per training step, with 8 workers 26 ms.
   * - compile
     - ``task.compile=True``
       ``AtomisticTask(compile=True)``
     - run the model and its derivatives (forces, stress) as one compiled graph
       (:class:`~schnetpack.model.CompiledPotential`, torch >= 2.14). Fewer, fused
       kernels; less GPU memory.
   * - fused optimizer
     - ``task.optimizer_args.fused=True``
       ``optimizer_args={"fused": True}``
     - update all parameters in one kernel (Adam, AdamW on CUDA).
   * - data on the GPU
     - ``data.data_device=cuda``
       ``AtomsDataModule(data_device="cuda")``
     - convert the dataset once into flat arrays, hold them on the GPU and build the
       batches there, padded to fixed shapes: no workers, no reads per structure.
       The dataset's neighbor list runs once at the conversion; its other
       transforms run on whole batches on the GPU.
   * - budget batching
     - ``data.batching=budget``
       ``AtomsDataModule(batching="budget")``
     - with the data on the GPU: pack training batches to the atoms and pairs
       ``batch_size`` structures hold on average, instead of padding every batch for
       its largest case. Batches hold ±10% structures around ``batch_size``.
   * - CUDA graph step
     - ``task.cuda_graph=True``
       ``AtomisticTask(cuda_graph=True)``
     - with the data on the GPU: record the whole training step (building the
       batch, the transforms, the model, the loss, the backward pass) once as a CUDA
       graph and replay it every step: one launch instead of hundreds, no host
       synchronization. Compiles the model.
   * - banded filter (PaiNN)
     - ``model.representation.banded_filter=True``
       ``PaiNN(banded_filter=True)``
     - evaluate PaiNN's filter network on each pair's band of Gaussians: a large
       saving with many narrow Gaussians (600 on 150 Å for GPFF: 2.25× per epoch).
       Needs an evenly spaced, non-trainable ``GaussianRBF``; no gain when the
       distances need a gradient (forces).

The gradient clipping and loss-skip options of the task, ``grad_clip`` and
``skip_loss_above``, work in every mode; with ``cuda_graph`` and a fused optimizer the
skip is decided on the GPU, without a host sync.

Which settings for which case
=============================

Measured with a few epochs on one H100 PCIe, training step in ms. "Before" is the
default route with the QM9 config's former 2 workers; "default" is the default route
now (8 workers); "compile" adds ``task.compile`` and a fused optimizer; "fastest" is
the data on the GPU with budget batching and the CUDA graph step:

.. list-table::
   :header-rows: 1
   :widths: 40 12 12 12 12

   * - case
     - before
     - default
     - compile
     - fastest
   * - QM9 energy, PaiNN 3 x 128, batch 100
     - 57.5
     - 20.2
     - 19.5
     - 2.4
   * - QM9 energy, SchNet 6 x 128
     - 56.5
     - 21.1
     - 19.7
     - 2.5
   * - QM9 energy, PaiNN 64 x 3 (small model)
     - 55.9
     - 21.5
     - 20.5
     - 1.6
   * - QM9 energy, PaiNN 512 x 3 (large model)
     - 55.3
     - 26.0
     - 20.6
     - 10.5
   * - QM9 energy, SO3net 3 x 128, lmax 2
     - 79.2
     - 76.2
     - 20.3
     - 9.8
   * - MD17 aspirin, energy + forces, batch 100
     - 40.4
     - 33.0
     - 25.7
     - 5.6
   * - MD17 aspirin, energy + forces, batch 10
     - 35.3
     - 27.9
     - 18.6
     - 1.8
   * - QM9 position denoising (``Diffuse``), PaiNN 3 x 128
     -
     - 39.6
     - 38.0
     - 3.8
   * - Materials Project, formation energy, PaiNN 3 x 128, batch 32
     -
     - 14.6
     - 10.5
     - 2.6

- **Every case: the data on the GPU, budget batching, the CUDA graph step and a fused
  optimizer.** The gain over the default route is largest for small models and small
  batches (13-15×), where the default route is all overhead, and smallest for models
  that keep the GPU busy on their own (2.5× for PaiNN 512 x 3, 7.8× for SO3net).
- **Budget batching** is 20-30% faster than fixed batches on molecules, and 2.9× on
  crystals, whose sizes vary too much for one padded shape.
- **Compile alone** gains little as long as the DataLoader is the limit: the default
  route stays at ~20 ms per QM9 step with 8 workers, whatever the model. It helps where
  the model itself is slow, as SO3net (3.8×) and forces (1.3-1.5×).
- **The data on the GPU without the CUDA graph** (``data_device`` with ``compile``) is
  the middle ground, for losses or callbacks the graph does not take: 3.1× faster than
  compile alone on QM9.
- **Crystals:** with the data on the GPU, the dataset's neighbor list (e.g.
  ``MatScipyNeighborList``) runs once at the conversion, and its periodic images are
  kept as cell shifts.
- **Generative training:** ``Diffuse`` and the centering transforms run on the padded
  batches on the GPU; leave out the neighbor list (or use ``AllToAllNeighborList``), so
  that every batch holds all pairs of the noised structures.

Limits
======

- **The dataset must fit on the GPU** with the model; QM9 takes ~0.2 GB.
- **Transforms must work on collated batches**, as the centering, offset, cast and
  ``Diffuse`` transforms do; transforms that only work on one structure need the
  DataLoader route.
- **Stored pairs are for fixed geometries:** a neighbor list runs once at the
  conversion. Noised structures need all pairs (no neighbor list).
- **Outputs:** padded batches need ``MSELoss`` or ``L1Loss`` without loss masks, or a
  :class:`~schnetpack.generative.VariableOutput`. No knowledge distillation.
- **The first epoch** pays for compiling and recording the graph (10-40 s), and the
  first run on a dataset for the conversion (cached next to the db, or in
  ``array_dir``).
- **Very short steps** (MD17 with batch 10, small models) keep the GPU 75-87% busy:
  what remains is the host's fixed cost per step (the graph launch and the optimizer
  step).
- **TF32** (``+matmul_precision=high``) is faster again but changes the numbers; on QM9
  energies in Hartree it cost accuracy. It is not a default.

Example
=======

.. code-block:: bash

   spktrain experiment=qm9_atomwise data.data_device=cuda data.batching=budget \
       task.compile=True task.cuda_graph=True task.optimizer_args.fused=True

Without Lightning
=================

The same speedups work in a training loop of your own, and nothing below imports
Lightning:

- the data on the GPU: :class:`~schnetpack.data.device.DeviceAtomsData` and
  :class:`~schnetpack.data.device.DeviceAtomsLoader`;
- their statistics: :class:`~schnetpack.data.device.ArrayStats`;
- the CUDA graph step: :class:`~schnetpack.train.GraphStep`;
- the parameter average: :class:`~schnetpack.train.EMA`.

.. code-block:: python

   import gc

   import torch

   import schnetpack as spk
   import schnetpack.transform as trn
   from schnetpack.data.device import ArrayStats, DeviceAtomsData, DeviceAtomsLoader
   from schnetpack.train import EMA, GraphStep

   # converted once (cached next to the db); the neighbor list runs at the conversion
   data = DeviceAtomsData.from_dataset(dataset, "cuda", trn.MatScipyNeighborList(5.0))
   stats = ArrayStats(data, train_idx, dataset.atomrefs)
   transforms = [
       trn.RemoveOffsets("energy_U0", remove_mean=True, remove_atomrefs=True),
       trn.CastTo32(),
   ]
   for t in transforms:  # before the loaders, which move them to the GPU
       t.initialize(stats)
   train = DeviceAtomsLoader(data, train_idx, 100, transforms, batching="budget")
   val = DeviceAtomsLoader(data, val_idx, 100, transforms)

   model.initialize_transforms(stats)  # its postprocessors, e.g. AddOffsets
   model.to("cuda")
   optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, fused=True)
   ema = EMA(model.parameters(), decay=0.995)
   step = GraphStep(
       spk.model.CompiledPotential(model, dynamic=False),
       outputs,
       optimizer,
       train.build_batch,
       skip_loss_above=1e3,
   )

   gc.collect()
   gc.freeze()  # no garbage-collector stalls between the short steps
   for epoch in range(n_epochs):
       for sel in train.index_batches:  # the step builds the batch itself
           loss = step(sel)
           ema.update(gate=step.keep)  # not on skipped steps
       with ema.average_parameters():
           for batch in val:
               ...

The batches are padded:

- compute the loss with :func:`~schnetpack.objectives.calculate_padded_loss`;
- compute metrics on the rows where :func:`~schnetpack.objectives.padding_mask` is 1;
- extract the targets (:func:`~schnetpack.objectives.extract_targets`) before the
  model runs, because the model writes its predictions under the output keys.

Without the CUDA graph, iterate over ``train`` itself: it gives the transformed batches.

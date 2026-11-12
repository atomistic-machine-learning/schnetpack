# Training faster

With the default settings, small and medium models on molecules (QM9, MD17) keep the
GPU busy only 5-30% of the time: each training step waits for the DataLoader workers
to read, transform and collate the next batch, and for Python to launch the step's
hundreds of kernels. The switches below remove that overhead. Each one can be turned
on or off on its own; the combination for a given training is in the next section.

## Switches

| switch | `spktrain` / Python | what it does |
| --- | --- | --- |
| DataLoader workers | `data.num_workers=8`<br>`AtomsDataModule(num_workers=8)` | read and transform structures in 8 processes. With 2 workers a QM9 batch of 100 took 55 ms per training step, with 8 workers 26 ms. |
| compile | `task.compile=True`<br>`AtomisticTask(compile=True)` | run the model and its derivatives (forces, stress) as one compiled graph ({class}`~schnetpack.model.CompiledPotential`, torch >= 2.14). Fewer, fused kernels; less GPU memory. |
| fused optimizer | `task.optimizer_args.fused=True`<br>`optimizer_args={"fused": True}` | update all parameters in one kernel (Adam, AdamW on CUDA). |
| data in memory | `data.in_memory=True`<br>`AtomsDataModule(in_memory=True)` | convert the dataset once into flat arrays, hold them in memory and build whole batches from them, padded to fixed shapes: no workers, no reads per structure. Each batch is gathered on the CPU while the GPU works on the last step, and copied over. The dataset's neighbor list runs once at the conversion and its leading deterministic transforms (centering, offsets, casts) once at setup; the rest run on whole batches on the GPU. |
| budget batching | always, with the data in memory | training batches are packed to the atoms and pairs `batch_size` structures hold on average, instead of padding every batch for its largest case. Batches hold ±10% structures around `batch_size`. |
| CUDA graph step | `task.cuda_graph=True`<br>`AtomisticTask(cuda_graph=True)` | with the data in memory: record the training step (the model, the loss, the backward pass) once as a CUDA graph and replay it every step on the padded batch: one launch instead of hundreds, no host synchronization. Compiles the model. |
| fused kernel (SO3net) | `model.representation.fused_kernel=True`<br>`SO3net(fused_kernel=True)` | sum each convolution's messages in one Triton kernel: 1.9× on energies at `lmax=2` and 4.5× at `lmax=3`, 3× with forces at `lmax=3`; no gain at `lmax=1`. |
| banded filter (PaiNN) | `model.representation.banded_filter=True`<br>`PaiNN(banded_filter=True)` | evaluate PaiNN's filter network on each pair's band of Gaussians: a large saving with many narrow Gaussians (600 on 150 Å for GPFF: 2.25× per epoch). Needs an evenly spaced, non-trainable `GaussianRBF`; no gain when the distances need a gradient (forces). |

The gradient clipping and loss-skip options of the task, `grad_clip` and
`skip_loss_above`, work in every mode; with `cuda_graph` and a fused optimizer the
skip is decided on the GPU, without a host sync.

## Which settings for which case

Measured with four short epochs on one H200 NVL (16 CPU cores), training step in ms.
"Default" is the default route with 8 DataLoader workers; "fastest" is the data in
memory (budget batches), the CUDA graph step and a fused optimizer:

| case | default | fastest | gain |
| --- | --- | --- | --- |
| QM9 energy, PaiNN 3 x 128, batch 100 | 10.8 | 1.9 | 5.7× |
| QM9 energy, PaiNN 3 x 64 (small model) | 9.8 | 1.3 | 7.6× |
| QM9 energy, PaiNN 3 x 512 (large model) | 17.7 | 7.4 | 2.4× |
| QM9 energy, SO3net 3 x 128, lmax 2 |  | 7.5 |  |
| MD17 aspirin, energy + forces, batch 100 | 22.5 | 4.2 | 5.3× |
| MD17 aspirin, energy + forces, batch 10 | 10.7 | 1.6 | 6.5× |
| Materials Project, formation energy, PaiNN 3 x 128, batch 32 | 7.7 | 2.0 | 3.8× |
| QM9 position denoising (`Diffuse`), PaiNN 3 x 128 | 12.9 | 2.4 | 5.5× |
| MP-20 position denoising (`Diffuse`, pairs rebuilt per batch), batch 100 |  | 5.5 |  |

- **Every case: the data in memory, the CUDA graph step and a fused optimizer.** The
  gain over the default route is largest for small models and small batches, where
  the default route is all overhead, and smallest for models that keep the GPU busy
  on their own (2.4× for PaiNN 3 x 512).
- **How close to the limit:** hand-written training loops for single cases, with
  every speedup coded by hand, were 0-6% faster than these numbers (14% for MP-20
  denoising).
- **Budget batching** was 20-30% faster than fixed padded batches on molecules, and
  2.9× on crystals, whose sizes vary too much for one padded shape; it is the only
  training batcher for data in memory.
- **Compile alone** gains little as long as the DataLoader is the limit: the default
  route's step time is set by its workers, whatever the model. It helps where the
  model itself is slow, as SO3net (3.8× on an H100) and forces (1.3-1.5×).
- **The data in memory without the CUDA graph** (`in_memory` with `compile`) is
  the middle ground, for losses or callbacks the graph does not take. For models
  that keep the GPU busy the graph adds 6% (PaiNN 3 x 512) to 29% (SO3net).
- **Crystals:** the dataset's neighbor list (e.g. `MatScipyNeighborList`) runs once
  at the conversion, and its periodic images are kept as cell shifts.
- **Generative training:** `Diffuse` and the centering transforms run on the padded
  batches on the GPU. For molecules, leave out the neighbor list (or use
  `AllToAllNeighborList`), so that every batch holds all pairs of the noised
  structures; filtering them by the cutoff per batch was 37-41% slower on QM9. For
  crystals, put the neighbor list after `Diffuse`: it is then rebuilt on every
  batch, on the GPU ({class}`~schnetpack.transform.BatchCutoffNeighborList`).

## Limits

- **The dataset must fit into the host's memory:** the arrays are loaded as a whole.
  QM9 with its pairs within 5 Å takes 0.3 GB.
- **Large batches:** at batch 1000 on QM9, gathering the batch on the CPU cost 4%
  against a batch built on the GPU.
- **The CPU must keep up on very short steps:** the batch is gathered and handed to
  the GPU by one CPU thread. On a node whose CPU cores were all busy, MD17 with batch
  10 took 3.7 ms per step with the GPU 35% busy, against 1.6 ms on a free node. Give
  such trainings CPU cores of their own.
- **Transforms must work on collated batches**, as the centering, offset, cast and
  `Diffuse` transforms do; transforms that only work on one structure need the
  DataLoader route.
- **Stored pairs are for fixed geometries:** a neighbor list before any noise runs
  once at the conversion. After `Diffuse` it runs on every batch, by brute force
  over all atom pairs and lattice shifts of a structure: made for molecules and the
  small cells of crystal datasets, not for cells of thousands of atoms.
- **Outputs:** padded batches need an elementwise loss with mean reduction
  (`MSELoss`, `L1Loss`, `HuberLoss`, ...) or a
  {class}`~schnetpack.generative.VariableOutput`; loss masks work. No knowledge
  distillation.
- **The first epoch** pays for compiling and recording the graph (10-40 s), and the
  first run on a dataset for the conversion (cached next to the db, or in
  `array_dir`).
- **Very short steps** (MD17 with batch 10, small models) keep the GPU 75-87% busy:
  what remains is mostly idle time between the many small kernels inside the replay,
  which no change on the host removes.
- **TF32** (`+matmul_precision=high`) is faster again but changes the numbers; on QM9
  energies in Hartree it cost accuracy. It is not a default.

## Example

```bash
spktrain experiment=qm9_atomwise data.in_memory=True \
    task.compile=True task.cuda_graph=True task.optimizer_args.fused=True
```

## Without Lightning

The same speedups work in a training loop of your own, and nothing below imports
Lightning:

- the data in memory: `dataset.in_memory(device)`, an
  {class}`~schnetpack.data.InMemoryDataset` that returns whole batches, and
  {class}`~schnetpack.data.InMemoryLoader`, which chooses the structures of each
  batch;
- the statistics: {class}`~schnetpack.data.StatsAtomrefProvider`, as for a dataset
  on the database;
- the CUDA graph step: {class}`~schnetpack.train.GraphStep`;
- the parameter average: {class}`~schnetpack.train.EMA`.

```python
import gc

import torch

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import AtomsDataset, InMemoryLoader, StatsAtomrefProvider
from schnetpack.train import EMA, GraphStep

dataset = AtomsDataset(
    "qm9.db",
    load_properties=["energy_U0"],
    transforms=[
        trn.MatScipyNeighborList(5.0),
        trn.RemoveOffsets("energy_U0", remove_mean=True, remove_atomrefs=True),
        trn.CastTo32(),
    ],
)
# the dataset as arrays in memory: converted once (cached next to the db), the
# neighbor list run at the conversion; batches come out on the GPU
dataset = dataset.in_memory("cuda")
stats = StatsAtomrefProvider(dataset, train_idx)
dataset.initialize_transforms(stats)
dataset.cache_transforms()  # offsets and cast applied once, not per batch
train = InMemoryLoader(dataset.subset(train_idx, "train"), 100, shuffle=True)
val = InMemoryLoader(dataset.subset(val_idx, "val"), 100)

model.initialize_transforms(stats)  # its postprocessors, e.g. AddOffsets
model.to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, fused=True)
ema = EMA(model.parameters(), decay=0.995)
step = GraphStep(
    spk.model.CompiledPotential(model, dynamic=False),
    outputs,
    optimizer,
    skip_loss_above=1e3,
)

gc.collect()
gc.freeze()  # no garbage-collector stalls between the short steps
for epoch in range(n_epochs):
    for batch in train:  # padded batches: the same shapes every step
        loss = step(batch)  # copied into the graph's buffers, then one replay
        ema.update(gate=step.keep)  # not on skipped steps
    with ema.average_parameters():
        for batch in val:
            ...
```

The batches are padded:

- compute the loss with {func}`~schnetpack.objectives.calculate_padded_loss`;
- compute metrics on the rows where {func}`~schnetpack.objectives.padding_mask` is 1;
- extract the targets ({func}`~schnetpack.objectives.extract_targets`) before the
  model runs, because the model writes its predictions under the output keys.

Without the CUDA graph, run the model, the loss and the optimizer on the same batches
yourself.

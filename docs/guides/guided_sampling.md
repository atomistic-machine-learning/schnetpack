# Guided sampling

A {class}`~schnetpack.dynamics.Sampler` moves structures down the reverse process of a
generative model. There are three ways to steer it:

| Way | Where it acts | Needs |
|---|---|---|
| A **guidance term** on an update | added to the field the update steps on, at every model call | a term, e.g. a property model trained on noised structures |
| **Classifier-free guidance** on the calculator | mixes the conditional and the unconditional head of one model | a model trained with a condition that is dropped for part of the data |
| A **hook** on the sampler | edits the structures between two steps | nothing: any model |

The examples below sample the positions of molecules with a model `model` whose eps head
is trained on a variance-exploding schedule:

```python
import torch

import schnetpack as spk
import schnetpack.transform as trn
from schnetpack import properties
from schnetpack.data import ASEAtomsData
from schnetpack.dynamics import (
    AncestralUpdate,
    ClassifierGuidance,
    DatasetStructures,
    FieldCalculator,
    FreezeScaffold,
    Hook,
    KarrasGrid,
    Sampler,
    SDEUpdate,
)
from schnetpack.generative import (
    EpsParametrization,
    GaussianInterpolant,
    GaussianPrior,
    Variable,
    VELinear,
)

positions = Variable(
    name="positions",
    view=properties.R,
    per="atom",
    noising=GaussianInterpolant(
        VELinear(sigma_max=30.0, t_min=0.0), prior=GaussianPrior(std=30.0, centered=False)
    ),
    parametrization=EpsParametrization(),
    output_key="eps_pred",
)
model = spk.load_model("runs/my_diffusion_model/best_model", device="cuda")
dataset = ASEAtomsData("qm9.db", load_properties=[], transforms=[])  # the compositions
grid = KarrasGrid(t_min=0.01 / 30.0, rho=5.0)


def structures(*transforms):
    """The compositions of the dataset as starting structures of a sampler."""
    return DatasetStructures(
        dataset, transforms=[trn.AllToAllNeighborList(), trn.CastTo32(), *transforms]
    )
```

## Guidance terms

A guidance term is a module that reads a batch and returns a tensor shaped like the field
it is added to ({mod}`schnetpack.dynamics.guidance`). An update takes a list of terms and
adds each one, scaled by its `weight`, to its field at every stage of every step:

| Update | Field the terms are added to |
|---|---|
| {class}`~schnetpack.dynamics.SDEUpdate` | the score |
| {class}`~schnetpack.dynamics.ODEUpdate` | the velocity |
| {class}`~schnetpack.dynamics.AncestralUpdate` | the denoised structure x0 |

The update adds the term as it is. The weight carries it into the unit of the field; no
conversion between score, velocity and x0 is applied.

### Classifier guidance

{class}`~schnetpack.dynamics.ClassifierGuidance` steers toward structures with a requested
property. It assumes a Gaussian likelihood of the target around the prediction of a
property model and returns the gradient of its logarithm with respect to the positions,
`-1/2 grad (model(x_t, t) - target)^2`. On the score of an `SDEUpdate` the weight is the
guidance strength divided by the variance of that likelihood, in units of one over the
squared property unit.

```python
def with_target(gap):
    """A batch transform that writes the requested gap of every structure."""

    def transform(batch):
        n_structures = batch[properties.n_atoms].shape[0]
        batch["gap_target"] = torch.full((n_structures,), gap)
        return batch

    return transform


predictor = spk.load_model("runs/my_gap_predictor/best_model", device="cuda")
guidance = ClassifierGuidance(
    predictor, output_key="gap", target_key="gap_target", weight=4.0  # 1/eV^2
)
sampler = Sampler(
    FieldCalculator(model, device="cuda"),
    [SDEUpdate(positions, guidance=[guidance])],
    start=[positions],
    grid=grid,
    structures=structures(with_target(4.8)),
)
batch = sampler.sample(n_samples=500, n_steps=256)
```

The property model sees the noised structure and its time, so it has to be trained on
noised structures of the same noising: put {class}`~schnetpack.generative.Diffuse` into
the batch transforms of its training and give it the time, e.g. with
{class}`~schnetpack.model.TimeConditioning` in front of an
{class}`~schnetpack.model.Atomwise` head. A model trained on clean structures only gives
no useful gradient at high noise. Pass the property model itself, not a compiled one: the
term differentiates it.

A weight that is too large breaks single structures (atoms on top of each other, atoms
flung far away) and lowers the share of valid molecules. Compare a few weights and check
the samples.

### Restraints

{class}`~schnetpack.dynamics.HarmonicRestraint` (see {doc}`guidance`) returns a force in
eV/Å and is added to the field of an update in the same way. On a score its weight is
1/kT in 1/eV.

## Classifier-free guidance

Classifier-free guidance needs no second model. One model is trained with a condition
that is dropped for a share of the structures, so it learns the conditional and the
unconditional head. Sampling mixes the two:

```
head = (1 + w) * conditional - w * unconditional
```

`w = 0` is the conditional model, `w = -1` the unconditional one, and `w > 0` moves
further away from the unconditional head.

**Training.** {class}`~schnetpack.nn.ConditionEmbedding` embeds one scalar per structure
and adds it to the atom embeddings of PaiNN, SchNet or SO3net through their
`electronic_embeddings`. {class}`~schnetpack.generative.DropCondition` replaces the
condition by NaN, which means "no condition", for a share `p` of the structures of every
training batch:

```yaml
data:
  dataset:
    load_properties: [gap]
  batch_transforms:
    - _target_: schnetpack.generative.Diffuse
      ...
    - _target_: schnetpack.generative.DropCondition
      key: gap
      p: 0.1
    - _target_: schnetpack.transform.CastTo32

model:
  representation:
    electronic_embeddings:
      - _target_: schnetpack.nn.ConditionEmbedding
        property_key: gap
        num_features: ${globals.n_atom_basis}
        mean: 6.8  # mean and standard deviation of the training values
        std: 1.3
```

**Sampling.** The condition is a key of the batch, one value per structure. The
{class}`~schnetpack.dynamics.FieldCalculator` takes the scale and the keys it sets to NaN
for the unconditional call:

```python
def with_condition(gap):
    def transform(batch):
        n_structures = batch[properties.n_atoms].shape[0]
        batch["gap"] = torch.full((n_structures,), gap)
        return batch

    return transform


calculator = FieldCalculator(model, device="cuda", cfg_scale=2.0, condition_keys=["gap"])
sampler = Sampler(
    calculator,
    [AncestralUpdate(positions)],
    start=[positions],
    grid=grid,
    structures=structures(with_condition(4.8)),
)
batch = sampler.sample(n_samples=500, n_steps=64)
```

A scale other than 0 costs two model calls per step. A batch whose condition is NaN runs
the unconditional head with one call, so the same model also serves as the unconditional
model, e.g. under classifier guidance.

```{note}
`ConditionEmbedding` is linear in the standardized value. A condition far outside the
values seen in training is not safe: the model can return very large outputs for it.
Keep requested values inside the training range.
```

## Editing structures between steps

A {class}`~schnetpack.dynamics.Hook` edits the batch around every step of a sampler or an
optimizer: `before_step` changes what the model sees, `after_step` what the step
produced. Unlike guidance it does not change the field, so it does not act inside a step
with several stages.

{class}`~schnetpack.dynamics.FreezeScaffold` holds a part of every structure fixed while
the rest is generated. It overwrites the marked atoms with their reference positions
before and after every step, so the model always sees the clean scaffold and the run
ends on it. The mask and the reference are keys of the batch:

```python
def hold_heavy_atoms(batch):
    """Keep the heavy atoms where they are in the dataset; generate the hydrogens."""
    batch[properties.fixed_atoms] = batch[properties.Z] != 1
    batch[properties.R_reference] = batch[properties.R].clone()
    return batch


sampler = Sampler(
    FieldCalculator(model, device="cuda"),
    [AncestralUpdate(positions)],
    start=[positions],
    grid=grid,
    structures=structures(hold_heavy_atoms),
    hooks=[FreezeScaffold()],
)
```

Give the reference in the frame the model was trained in, e.g. centered if the training
structures were centered.

A hook of your own subclasses `Hook` and returns a new batch:

```python
class Recenter(Hook):
    """Move every structure's center of geometry to the origin after each step."""

    def after_step(self, batch, step, n_steps):
        positions, idx_m = batch[properties.R], batch[properties.idx_m]
        n_atoms = batch[properties.n_atoms]
        center = torch.zeros(len(n_atoms), 3).to(positions)
        center = center.index_add_(0, idx_m, positions) / n_atoms.unsqueeze(-1)
        return {**batch, properties.R: positions - center[idx_m]}
```

During sampling the batch a hook sees also holds the current time per atom and per
structure.

## With `spk sample`

`spk sample` takes the same settings as `sample.*` keys:

```bash
# classifier-free guidance toward a gap of 4.8 eV
spk sample model_dir=runs/my_diffusion_model 'sample.condition={gap: 4.8}' sample.cfg_scale=2.0

# classifier guidance on the model's unconditional head
spk sample model_dir=runs/my_diffusion_model \
    'sample.condition={gap_target: 4.8}' \
    sample.updates.positions._target_=schnetpack.dynamics.SDEUpdate \
    'sample.updates.positions.guidance=[{_target_: schnetpack.dynamics.ClassifierGuidance, model: {_target_: schnetpack.load_model, path: runs/my_gap_predictor}, output_key: gap, target_key: gap_target, weight: 4.0}]'
```

- `sample.condition` holds values that are written into every structure, by batch key:
  the condition of a conditional model and the target of a guidance term. A condition of
  the model that is not given is set to NaN, so a conditional model without
  `sample.condition` samples unconditionally.
- `sample.cfg_scale` is the scale `w`.
- `sample.updates.<variable>.guidance` lists the guidance terms of that variable's
  update.

Hooks are not available on the command line.

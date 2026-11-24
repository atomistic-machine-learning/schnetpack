# Saving and loading models

`spk train` stores the best model of a run as a folder, `<run>/best_model/`, with two
files:

- `hyperparameters.json`: the class and constructor arguments of every module, in the
  shape of the Hydra `model:` block, as plain JSON. Next to them: the format number,
  the SchNetPack version, and for a generative model the `generative:` and `sample:`
  settings of its run.
- `model.safetensors`: the weights and statistics (atom references, means, scales),
  each in the dtype it was trained with.

## Loading

```python
import schnetpack as spk

model = spk.load_model("runs/my_run/best_model", device="cuda")
```

{func}`~schnetpack.load_model` builds the model from `hyperparameters.json` with its own
table of SchNetPack classes, loads the weights strictly and returns the model in eval
mode. It runs no code from the files and needs no Hydra. A run folder works as well:
`spk.load_model("runs/my_run")` loads its `best_model`. The commands that take a model
(`spk predict`, `spk sample` and `spk deploy` with `model_dir=`), the ASE calculator
and the MD calculators load it the same way.

{func}`~schnetpack.load_hyperparameters` returns the content of `hyperparameters.json`,
e.g. the `sample` settings of a generative model.

A newer SchNetPack reads the files of older versions. An older SchNetPack refuses a file
of a newer format and asks for an upgrade.

## Saving a model built in Python

```python
spk.save_model(model, "my_model")
```

Every SchNetPack module records the arguments it is built with, and
{func}`~schnetpack.save_model` writes these records and the weights. Before writing,
it builds the model again from the records and compares the two. If a module was added,
removed or replaced after construction, or a weight has another name or shape, the file
would load as a different model: `save_model` then writes nothing and names the
difference. Build the model the way it should be saved, then save it.

A model file holds these constructor arguments:

- JSON values: numbers, strings, booleans, `None`, lists, and dicts with string keys;
- registered modules, such as the representation and the output heads;
- tensors that the class keeps as weights (the file holds their shape, the values come
  from the weights);
- the torch modules `nn.SiLU` and `nn.Embedding`;
- the functions `torch.nn.functional.silu`, `schnetpack.nn.shifted_softplus`,
  `torch.nn.init.xavier_uniform_` and `torch.nn.init.zeros_`.

Any other argument is refused, with its place in the model.

### Classes of your own

A module class of your own (a head, a representation) becomes savable with
{func}`~schnetpack.register`:

```python
import schnetpack as spk
from torch import nn


@spk.register
class MyHead(nn.Module):
    def __init__(self, n_in: int, output_key: str):
        ...
```

Import the module that defines the class before loading a model that contains it.

## Checkpoints

The checkpoints in `<run>/checkpoints/` hold the weights and the training state
(optimizer, scheduler, random states), not the model itself, so they load with
`torch.load(..., weights_only=True)`. `spk train` resumes from `last.ckpt`, and
`spk test run_dir=<run>` tests the best checkpoint; both build the model from the run's
`config.yaml` first. Runs whose checkpoints hold a pickled model, from earlier
versions, cannot be resumed or tested with this version.

## Pickled models

A path that is a file, not a folder, is a model pickled by SchNetPack 2.x or an earlier
3.0 development version. `spk.load_model` loads it as before, through
{func}`schnetpack.utils.load_model`, which maps the old module paths. Unpickling runs
code from the file: load only files you trust. A pickled model cannot be saved in the
new format, since it carries no record of its constructor arguments; build the same
model in Python, load the pickled model's `state_dict` into it, and save that.

MACE models (`MaceModel`) stay pickled files: their architecture is a `torch.export`
archive that the format cannot describe.

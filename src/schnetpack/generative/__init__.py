"""
Generative modeling for atomistic systems: diffusion and flow matching as
composable building blocks. Pure PyTorch, no Lightning.

A model is a configuration of four independent axes plus its training:

- :mod:`~schnetpack.generative.processes`: the forward noising process
  x_t = a(t) x0 + b(t) x1 (the schedule), holding the prior and coupling.
- :mod:`~schnetpack.generative.priors`: what the x1 endpoint is.
- :mod:`~schnetpack.generative.couplings`: how (x0, x1) batches are paired.
- :mod:`~schnetpack.generative.parametrizations`: what the network predicts,
  with the training target and the conversions between fields.
- :mod:`~schnetpack.generative.differential_equations`: the (f, g) SDE chart
  of a process and its reversal.
- :mod:`~schnetpack.generative.times`: which times training draws.
- :mod:`~schnetpack.generative.losses` and
  :mod:`~schnetpack.generative.transforms`: the training step, at tensor
  level and as a data-pipeline transform.

Running a trained model lives in :mod:`schnetpack.dynamics`. Training and
sampling must share the same (process, parametrization) objects. Theory and
design: ``docs_new/README.md``.
"""

from schnetpack.generative.couplings import *
from schnetpack.generative.losses import *
from schnetpack.generative.parametrizations import *
from schnetpack.generative.priors import *
from schnetpack.generative.processes import *
from schnetpack.generative.differential_equations import *
from schnetpack.generative.times import *
from schnetpack.generative.transforms import *

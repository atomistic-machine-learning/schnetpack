"""
Generative modeling for atomistic systems: diffusion and flow matching as
composable building blocks. Pure PyTorch, no Lightning.

A model is a configuration of four independent axes plus its training:

- :mod:`~schnetpack.generative.noising`: how noisy samples are paired with
  clean ones; the interpolant x_t = a(t) x0 + b(t) x1 holds the schedule,
  prior and coupling.
- :mod:`~schnetpack.generative.schedules`: a(t) and b(t), or a(t) and sigma(t).
- :mod:`~schnetpack.generative.priors`: what the x1 endpoint is.
- :mod:`~schnetpack.generative.couplings`: how (x0, x1) batches are paired.
- :mod:`~schnetpack.generative.parametrizations`: what the network predicts,
  with the training target and the conversions between fields.
- :mod:`~schnetpack.generative.differential_equations`: the (f, g) SDE chart
  of a process and its reversal.
- :mod:`~schnetpack.generative.times`: which times training draws.
- :mod:`~schnetpack.generative.variables`: a generated quantity (positions,
  types, a cell) with its noising, parametrization, head and loss; several
  share one time.
- :mod:`~schnetpack.generative.views`: where a variable lives in the
  structure, which is all the batch holds: a batch entry, wrapped periodic
  positions, or the cell as standardized parameters.
- :mod:`~schnetpack.generative.losses` and
  :mod:`~schnetpack.generative.transforms`: the training step, at tensor
  level for one variable and as a data-pipeline transform for a list.

Running a trained model lives in :mod:`schnetpack.dynamics`. Training and
sampling must share the same variables. Theory and
design: ``docs_new/README.md``.
"""

from schnetpack.generative.couplings import *
from schnetpack.generative.differential_equations import *
from schnetpack.generative.losses import *
from schnetpack.generative.noising import *
from schnetpack.generative.parametrizations import *
from schnetpack.generative.priors import *
from schnetpack.generative.schedules import *
from schnetpack.generative.times import *
from schnetpack.generative.transforms import *
from schnetpack.generative.variables import *
from schnetpack.generative.views import *

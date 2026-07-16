"""
Generative modeling for atomistic systems — pure PyTorch, no Lightning imports.

The subpackage factors generative models along four orthogonal axes. VE, VP,
flow matching and EDM are *configurations* of those axes, not classes of their
own:

- :mod:`~schnetpack.generative.paths` — the interpolant
  x_t = alpha(t) x0 + sigma(t) x1, defined by two schedules and their
  derivatives, plus the forward SDE (f, g) and the marginals that follow from
  them. Pure geometry: every method is a function of (alpha, sigma, alpha',
  sigma') and nothing else. This is the primitive; the drift and diffusion are
  not, since for flow matching the diffusion is a sampler choice rather than an
  intrinsic property.
- :mod:`~schnetpack.generative.couplings` — how (x0, x1) endpoint pairs are
  drawn, the joint law the marginals leave open. Independent for VE/VP/FM/EDM;
  nontrivial for OT flow matching, bridges, and any noise living in a
  constrained subspace.
- :mod:`~schnetpack.generative.parametrizations` — what the network predicts
  (score, noise, denoiser, velocity), and everything that follows from that
  choice: the training targets, the conversions between fields and the reverse
  process. Binds to a path and reads its schedule. Preconditioning lives
  alongside in :mod:`~schnetpack.generative.preconditioning` as a net-to-net
  wrapper.
- :mod:`~schnetpack.generative.integrators` — how a reverse process is
  solved, with :mod:`~schnetpack.generative.grids` choosing where the steps go.

The axes stay separate in both directions: adding a parametrization never
touches ``paths.py``, and adding a path never touches ``parametrizations.py``.
The parametrization holds the path rather than the reverse, because the
dependency runs that way — a parametrization is meaningless without a schedule,
while a path is perfectly usable on its own for noising or as a prior.

Around them:

- :mod:`~schnetpack.generative.reverse` — one generic reverse process, derived
  from a parametrization and a model, with a single churn knob spanning the
  probability-flow ODE (churn = 0) and the reverse-time SDE (churn = 1).
  Never implemented per path.
- :mod:`~schnetpack.generative.losses` — score, flow and bridge matching as one
  training step; the EDM objective is a configuration of it.
- :mod:`~schnetpack.generative.transforms` — the same training step as a
  preprocessing transform, for training through the SchNetPack data pipeline
  and an ordinary supervised loss instead.
- :mod:`~schnetpack.generative.priors` — starting distributions for sampling.
- :mod:`~schnetpack.generative.sampler` — composition of the above.
- :mod:`~schnetpack.generative.generate` — high-level generation entry.

The model contract is deliberately minimal::

    model(x, t, cond=None) -> raw output in some parametrization

with x of shape (n_samples, ...) and per-sample t. Nothing here wraps a network
or knows more about it than that, which is what lets the same machinery drive a
toy tensor net and a SchNetPack
:class:`~schnetpack.model.NeuralNetworkPotential` behind an adapter (where the
sample axis is atoms). That adapter, the Schrödinger-bridge orchestrator,
consistency models and optimal-transport couplings are accommodated by the
design but not yet implemented.
"""

from schnetpack.generative import integrators
from schnetpack.generative.couplings import *
from schnetpack.generative.generate import *
from schnetpack.generative.grids import *
from schnetpack.generative.integrators import *
from schnetpack.generative.losses import *
from schnetpack.generative.parametrizations import *
from schnetpack.generative.paths import *
from schnetpack.generative.preconditioning import *
from schnetpack.generative.priors import *
from schnetpack.generative.reverse import *
from schnetpack.generative.sampler import *
from schnetpack.generative.transforms import *

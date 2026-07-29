"""
Generative modeling for atomistic systems — pure PyTorch, no Lightning imports.

The subpackage factors generative models along orthogonal axes. VE, VP,
flow matching and GPFF are *configurations* of those axes, not special
cases with bespoke logic:

- :mod:`~schnetpack.generative.processes` — the forward noising process:
  the interpolant x_t = a(t) x0 + b(t) x1 composed with a prior and a
  coupling. The schedule is the subclass (:class:`VP`, :class:`VE`,
  :class:`FlowMatching`, ... — each defines a(t), b(t) or the TV/SNR pair
  and nothing else, in the literature's vocabulary), while the endpoint and
  the pairing stay swappable constructor arguments. The process owns the
  whole forward side: the noise level sigma(t) = b(t) * std, the SDE
  diffusion, the endpoint draw (:meth:`~processes.Process.perturb`) and the
  sampling start. Whether the one-sided Gaussian kernel holds — what the
  score/noise targets and the closed-form posterior assume — is judged from
  the actual configuration
  (:attr:`~processes.Process.has_gaussian_kernel`), not from the class: the
  same schedule is a Gaussian diffusion under a Gaussian prior and a
  general stochastic interpolant under a structured one.
- :mod:`~schnetpack.generative.priors` — what the x1 endpoint *is*: the
  distribution drawn at both training time (per data sample) and sampling
  time (the start state). Isotropic Gaussian for VE/VP/FM; structured
  (per-molecule covariance, scaffold, second dataset) for GPFF and bridges.
  The endpoint's scale lives here (``GaussianPrior(std=sigma_max)`` is
  where a VE process's noise magnitude sits — not in the schedule),
  declared once.
- :mod:`~schnetpack.generative.couplings` — how (x0, x1) batches are *paired*
  once drawn. Identity for VE/VP/FM; a re-ordering for permutation/OT
  alignment; a data-dependent reshaping for covariance matching. Never draws
  the endpoint — that is the prior's job.
- :mod:`~schnetpack.generative.parametrizations` — what the network predicts
  (score, noise, denoiser, velocity, pseudo-force), and everything that
  follows from that choice: the training targets, the conversions between
  fields and the reverse process. Stateless field math: every method takes
  the process it is applied to, and reads everything from it.
- :mod:`~schnetpack.generative.integrators` — how a reverse process is
  solved, with :mod:`~schnetpack.generative.grids` choosing where the steps go.

The axes stay separate in both directions: adding a parametrization never
touches ``processes.py``, and adding a schedule never touches
``parametrizations.py``. Neither holds the other: a parametrization is
stateless field math, and the consumers that need both — ``Diffuse``,
``MatchingLoss``, ``Sampler``, ``ReverseProcess`` — take the
``(process, parametrization)`` pair explicitly. Validity is a construction
invariant, checked where the pair meets: each consumer calls
``parametrization.validate(process)`` in its constructor, and the
score/noise parametrizations demand the Gaussian kernel, which the process
judges from its prior, coupling and bridge noise
(:meth:`~processes.Process.gaussian_kernel_obstruction` names what is in
the way). The one obligation this leaves the caller: training and sampling
must name the *same* pair — share the objects, don't rebuild them.

Around them:

- :mod:`~schnetpack.generative.reverse` — one generic reverse process, derived
  from a process, a parametrization and a model, with a single churn knob spanning the
  probability-flow ODE (churn = 0) and the reverse-time SDE (churn = 1).
  Never implemented per schedule.
- :mod:`~schnetpack.generative.losses` — score, flow and bridge matching as one
  training step.
- :mod:`~schnetpack.generative.transforms` — the same training step as a
  preprocessing transform, for training through the SchNetPack data pipeline
  and an ordinary supervised loss instead.
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
from schnetpack.generative.priors import *
from schnetpack.generative.processes import *
from schnetpack.generative.reverse import *
from schnetpack.generative.sampler import *
from schnetpack.generative.transforms import *

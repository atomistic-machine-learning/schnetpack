# `schnetpack.generative` — design and theory documentation

Generative modeling for atomistic systems: diffusion, flow matching and
their relatives, as composable building blocks. Pure PyTorch at the core —
no Lightning imports, no atomistic assumptions below the adapter layer.

This documentation explains the mathematics the subpackage implements, the
design it is built around, and how to use it. One file per axis, one for the
sampling machinery, one for training:

| file | contents |
| --- | --- |
| [processes.md](processes.md) | **Axis 1 — the schedule.** The interpolant $x_t = a\,x_0 + b\,x_1$, the two ways to define a schedule, the derived SDE coefficients $f, g^2$, the Gaussian kernel and its closed forms. `VP`, `VE`, `VELinear`, `FlowMatching`, `VPISSNR`. |
| [priors.md](priors.md) | **Axis 2 — the endpoint.** What $x_1$ is, who owns the noise scale, the `gaussian`/`std` declarations, per-molecule centering, structured priors. |
| [couplings.md](couplings.md) | **Axis 3 — the pairing.** The joint law of $(x_0, x_1)$ as a re-pairing of batches, marginal preservation, the `groups` mechanism, OT-style couplings. |
| [parametrizations.md](parametrizations.md) | **Axis 4 — the prediction.** Score, noise, denoiser, velocity and pseudo-force heads: training targets, conversions, validity, loss weighting. |
| [sampling.md](sampling.md) | The reverse process and its churn knob, integrators (Euler–Maruyama, Heun, ancestral), time grids, `Sampler` and `DirectDenoisingSampler`. |
| [training.md](training.md) | The two training routes: `MatchingLoss` (tensor level) and `Diffuse` (data-pipeline transform), time samplers and loss weights. |
| [flow_matching_sde.md](flow_matching_sde.md) | Deep dive: how flow matching is represented in the $(f, g)$ SDE framework, and what changes under a non-Gaussian endpoint. |

The rest of this page states the central idea, shows the shortest usable
example, and argues for the design.


## The central idea: methods are configurations, not classes

A generative model of the diffusion/flow-matching family is assembled from
four independent choices plus a sampling strategy:

1. **The schedule** — *when* the data turns into the endpoint:
   $a(t), b(t)$ in $x_t = a(t)\,x_0 + b(t)\,x_1$.
2. **The endpoint** — *what* $x_1$ is: an isotropic Gaussian, a structured
   shape prior, a scaffold.
3. **The pairing** — *how* data and endpoint batches are matched:
   independently, or re-paired for straighter transport.
4. **The prediction** — *what the network outputs*: score, noise, clean
   sample, velocity, pseudo-force.

Plus, at generation time: how the reverse process is discretized
(integrator), where the steps go (grid), and how much stochasticity is used
(churn).

The literature's named methods are *points in this configuration space*, not
special cases with bespoke logic:

| named method | schedule | endpoint | pairing | prediction | typical sampling |
| --- | --- | --- | --- | --- | --- |
| DDPM / VP diffusion | `VP()` | unit Gaussian (default) | identity | `EpsParametrization` | ancestral, or SDE at churn 1 |
| Score matching (NCSN/SMLD) | `VE(sigma_min, sigma_max)` | Gaussian, `std=sigma_max` (built internally) | identity | `ScoreParametrization` + $b^2$ weight | ancestral / SDE |
| EDM-style denoiser | `VELinear(scale=sigma_max)` | Gaussian | identity | `X0Parametrization` | `Heun`, churn 0 |
| Flow matching / rectified flow | `FlowMatching()` | unit Gaussian | identity (OT later) | `VelocityParametrization` | ODE, churn 0 |
| GPFF | `VE(b_min=..., prior=...)` | shape prior | identity / structured | `PseudoForceParametrization` + clamped $1/b^2$ weight | `DirectDenoisingSampler` |
| TV/SNR ISSNR | `VPISSNR(eta, kappa)` | unit Gaussian | identity | any | any |

Every row shares the same machinery. There is one `perturb`, one training
step, one `ReverseProcess`, one set of integrators — never a
`DDPMSampler` next to an `FMSampler`.


## The shortest usable example

Tensor level, toy data, flow matching:

```python
import torch
from schnetpack.generative import (
    FlowMatching, VelocityParametrization, MatchingLoss, Sampler, Heun,
)

process = FlowMatching()                     # a = 1 - t, b = t, unit Gaussian endpoint
param   = VelocityParametrization()          # the model predicts d/dt x_t
loss_fn = MatchingLoss(process, param)       # validates the pairing at construction

model = torch.nn.Sequential(...)             # any callable (x, t, cond) -> output
for x0 in loader:
    loss = loss_fn(model, x0)                # perturb + target + weighted MSE
    loss.backward(); ...

sampler = Sampler(process, param, Heun(), churn=0.0)   # the SAME two objects
samples = sampler.sample(model, shape=(64, 3), n_steps=50)
```

Swapping `FlowMatching()` for `VP()` and `VelocityParametrization()` for
`EpsParametrization()` turns this into DDPM, with no other line changing.
That is the design working as intended.

The model contract is deliberately minimal: any callable
`model(x, t, cond) -> raw output` with `x` of shape `(n_samples, ...)` and
per-sample `t`. Nothing in the subpackage wraps a network or knows more about
it than that — which is what lets the same machinery drive a toy MLP and a
SchNetPack `NeuralNetworkPotential` behind an adapter (where the sample axis
is atoms).


## The design argument

Six principles, each with the failure mode it prevents. These recur in every
axis file; here is the summary.

### 1. The axes stay orthogonal in both directions

Adding a parametrization never touches `processes.py`; adding a schedule
never touches `parametrizations.py`. Neither holds the other: a
parametrization is stateless field math taking the process as an argument,
and the consumers that need both — `Diffuse`, `MatchingLoss`, `Sampler`,
`ReverseProcess` — take the `(process, parametrization)` pair explicitly.

*Prevents:* the $N \times M$ explosion. Five predictions times five
schedules times two endpoint families is fifty classes if the axes are
entangled; here it is ten small ones plus composition.

### 2. Properties are judged from the configuration, not the class

Whether the one-sided Gaussian kernel
$p(x_t \mid x_0) = \mathcal{N}(a\,x_0, \sigma^2 I)$ holds — the assumption
behind the score/noise targets and the closed-form posterior — depends on the
*prior*, the *coupling* and the *bridge noise*, all constructor arguments.
So it is judged from them: `Process.has_gaussian_kernel` /
`gaussian_kernel_obstruction()`, not from a `GaussianDiffusion` subclass.

The decisive observation: Gaussianity is not aligned with the schedule axis.
`FlowMatching` under its default Gaussian prior *has* an exact Gaussian
kernel (score targets, DDIM and ancestral steps are all valid on it), while
`VE` under a shape prior — the GPFF configuration — does not. A class
hierarchy would either duplicate every schedule across a Gaussian and a
non-Gaussian branch or make false claims; a judged property is right in every
cell of that matrix, and names the failing condition when it refuses.

### 3. One owner per fact

The endpoint's scale lives on the prior (`prior.std`), declared once; the
schedule's $b$ is dimensionless and the process's noise level is the product
$\sigma(t) = b(t)\,\mathrm{std}$. The sampling start is not restated — it is
*derived* (`process.sampling_prior()` returns the training prior itself when
the coupling preserves the marginal). `perturb` owns every random draw of the
forward side, so the training target can see the same draws that built $x_t$.

*Prevents:* the silent-mismatch class of bug — a `sigma_max` set differently
in the schedule and the prior, a sampling start that drifts from what
training used, a bridge-noise draw the target never saw.

### 4. Validity settles at assembly

Every consumer constructor calls `parametrization.validate(process)`. A
score head on a shape-prior process fails when the `MatchingLoss` or
`Sampler` is *built*, with a message naming the obstruction — not mid-run,
not silently. Boolean declarations default to the safe side
(`Prior.gaussian = False`, `Coupling.preserves_marginal = False`): a wrong
`False` raises and asks for explicitness, a wrong `True` would train or
sample garbage silently.

### 5. The interpolant is the primitive; the SDE is derived

Training needs $x_t$ for a random $(x_0, t)$ in one shot, which the
interpolant gives and an SDE would make you integrate. The drift and
diffusion follow by differentiation ($f = \dot a/a$,
$g^2 = -\sigma^2\,\frac{\mathrm{d}}{\mathrm{d}t}\log \mathrm{SNR}$), whereas
the converse direction costs an ODE solve per schedule — and for flow
matching there is no intrinsic $g$ to start from, since its diffusion is a
sampler choice. See [flow_matching_sde.md](flow_matching_sde.md) for the
full derivation.

### 6. Numerical honesty over textbook forms

Formulas are implemented in the parametrization where they are conditioned:
$b = \sqrt{-\mathrm{expm1}(2\log a)}$ instead of $\sqrt{1-a^2}$, $g^2$ as
one log-derivative instead of a quotient of vanishing terms,
$\mathrm{sigmoid}(\log \mathrm{SNR})$ instead of
$\mathrm{SNR}/(1+\mathrm{SNR})$. Analytic derivative overrides are an
optimization over the autograd default, and their agreement is tested.


## The one obligation this leaves the caller

Training and sampling must name the **same** `(process, parametrization)`
pair. The design derives everything it can (the sampling start, the noise
level, the conversions), but it cannot know that the objects you hand the
`Sampler` are the ones the checkpoint was trained under — share the objects,
don't rebuild them. Everything else that can be checked, is.

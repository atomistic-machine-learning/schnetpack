# Axis 4 — Parametrizations: what the network predicts

*Module: `schnetpack.generative.parametrizations` · classes
`Parametrization`, `ScoreParametrization`, `EpsParametrization`,
`X0Parametrization`, `VelocityParametrization`,
`PseudoForceParametrization`.*

A parametrization is the **contract between a raw model output and the
canonical fields** of the generative model. It owns both directions:

- **Training** — `target(process, x0, x1, t, eps)` builds the regression
  target from an endpoint pair (consumed by [the losses](training.md));
- **Sampling** — `to_score`, `to_velocity`, `to_x0` convert a raw output
  into the fields the [reverse process](sampling.md) needs.

It is **stateless**: pure field math, holding nothing. Every method takes
the process it is applied to, and the consumers that need both — `Diffuse`,
`MatchingLoss`, `Sampler`, the reverse processes — take the
`(process, parametrization)` pair explicitly.


## 1. The theory: one model, five names

All heads regress conditional expectations of linear functions of the
endpoint pair, so an $L^2$-trained model converges to
$\mathbb{E}[\,\text{target} \mid x_t\,]$ — a field on $(x, t)$. The five
fields are linearly related wherever the
[Gaussian kernel](processes.md#5-the-gaussian-kernel-a-property-not-a-class)
$p(x_t \mid x_0) = \mathcal{N}(a x_0, \sigma^2 I)$ holds, via Tweedie's
formula $\mathbb{E}[x_0 \mid x_t] = (x_t + \sigma^2 s)/a$ and the schedule
coefficients. Two identities are load-bearing:

1. $f b^2 - b\dot b = -\tfrac12 g^2$ (immediate from the definition of
   $g^2$), hence

   $$
   v(x, t) = f(t)\,x - \tfrac12\,g^2(t)\,s(x, t)
   $$

   — the velocity *is* the probability-flow ODE drift, so flow matching and
   the PF-ODE never need separate code paths (derivation in
   [flow_matching_sde.md §5](flow_matching_sde.md)).
2. The Anderson reverse-time drift $f x - \tfrac12(1 + \eta^2) g^2 s$
   equals $v - \tfrac12 \eta^2 g^2 s$ — so the whole
   [reverse family](sampling.md) is one knob around the velocity, and at
   churn 0 a velocity-predicting model is used *directly*, never touching
   the singular velocity→score inversion.

Conversions route through the score: each head's `to_score` is its way *in*,
and the base-class `to_velocity`/`to_x0` are the shared ways *out*. A head
that *is* one of the fields overrides that field to return its output
untouched — which matters, because the generic route back can be singular
exactly where the direct one is exact (e.g. `to_x0` divides by $a$, which
is zero at flow matching's endpoint; the x0 and pseudo-force heads never
take that route).


## 2. The catalog

Throughout: $x_1$ is the endpoint draw, $\mathrm{std} = $ `process.std`,
$\sigma = b\cdot\mathrm{std}$.

| head | target | needs Gaussian kernel? | singular where | typical pairing |
| --- | --- | --- | --- | --- |
| `ScoreParametrization` | $-x_1 / (b\,\mathrm{std}^2)$ | **yes** | target $\to\infty$ as $b \to 0$ | NCSN/SMLD on `VE`, with $b^2$ weight |
| `EpsParametrization` | $x_1 / \mathrm{std}$ | **yes** | — (unit variance always) | DDPM on `VP` |
| `X0Parametrization` | $x_0$ | no | Tweedie `to_score` needs $\sigma > 0$ | EDM-style denoisers |
| `VelocityParametrization` | $\dot a\,x_0 + \dot b\,x_1$ | no | `to_score` divides by $g^2 \to 0$ at $t \to 0$ | flow matching, churn 0 |
| `PseudoForceParametrization` | $2\big((1-a)\,x_0 - b\,x_1\big)$ | no | — (recovery is division-free) | GPFF on `VE` |

Notes per head:

**Score.** The target is the score *of the Gaussian kernel*,
$-(x_t - a x_0)/\sigma^2 = -x_1/(b\,\mathrm{std}^2)$ — meaningless for any
other endpoint, hence the `validate` gate. It is the only target that
divides, so on a geometric `VE` schedule it spans orders of magnitude and an
unweighted $L^2$ sees only the low-noise end: pass
`weight=lambda t: process.b(t)**2`, which makes the objective identical to
noise matching up to the constant endpoint scale.

**Eps (DDPM convention).** Unit noise $x_1/\mathrm{std}$: unit variance at
every noise level and endpoint scale, which is the convention's whole
appeal. Score recovery is $s = -\epsilon/\sigma$.

**X0 (denoiser convention).** A plain conditional expectation, valid for
any process. `to_score` is Tweedie; `to_x0` returns the output directly —
the round trip through the score would divide by $a$.

**Velocity (flow-matching convention).** $\dot a x_0 + \dot b x_1$, valid
for every process — which is why flow, OT and bridge matching all regress
it. For `FlowMatching` the target is $x_1 - x_0$, constant along a pair —
the straightness that makes few-step sampling work. Its `to_score` inverts
identity 1 and degenerates as $g^2 \to 0$; reverse processes only ask for
the score at churn $> 0$, and their grids stop at `t_min`.

**Pseudo-force (GPFF convention).** $F = 2(x_0 - x_t)$, the negative
gradient of the pseudo-energy $\lVert x_t - x_0\rVert^2$: the head answers
"which way, and how far, back to a clean sample". Substituting the
interpolant gives the target without forming $x_t$. It is $x_0$ up to an
affine map — exact wherever the x0 head is, and recovery is division-free
($x_0 = x_t + F/2$), so nothing degenerates as $b \to 0$ and no $\sigma$ is
ever needed — which is what keeps GPFF's [direct denoising
sampler](sampling.md#5-directdenoisingsampler--gpffs-time-free-loop)
available under priors that declare no scalar scale.

What earns it a class of its own happens on a variance-*exploding* path
($a \equiv 1$), where the target collapses to $F = -2 b\,x_1$: its
magnitude carries the noise level $\sigma = b\,\mathrm{std}$, so a sampler
can estimate the noise level from the prediction alone and the head needs
**no time input at all**. That is the point of the method — and it is
VE-specific: on a VP-type path the $(1-a)\,x_0$ term mixes data back in and
the magnitude no longer reads as $\sigma$.

The same scaling is the cost: the target's scale runs with $b$ over the
orders of magnitude a geometric VE spans. Pass

```python
weight=lambda t: (1.0 / process.b(t) ** 2).clamp(max=1.0)
```

— the $1/b^2$ undoes the scaling exactly (making the objective noise
matching again), and the clamp is what keeps it distinct from an eps head:
it stops a handful of nearly-clean samples from dominating every gradient,
at the price of spending capacity where the correction is large.


## 3. Validity: `validate` and the Gaussian gate

`Parametrization.validate(process)` raises unless the head's target is
meaningful for the process, and **every consumer calls it in its
constructor** (`MatchingLoss`, `Diffuse`, `Sampler`, the reverse
processes). The score and eps heads override it to demand
`has_gaussian_kernel`; the resulting error names the obstruction and the
alternatives:

```
EpsParametrization regresses a target that is a statement about a Gaussian
kernel, which this process does not have: ShapePrior does not declare its
draws isotropic Gaussian (prior.gaussian is False). Fix the configuration,
or switch to a velocity, x0 or pseudo-force parametrization — those targets
are plain conditional expectations, valid for any process.
```

So an invalid assembly fails when it is *built*, not after an epoch of
training — the
[validity-at-assembly principle](README.md#4-validity-settles-at-assembly).

The sampling-side conversions have a gate of their own: `to_score` and the
generic `to_velocity`/`to_x0` routes are statements about the
[(f, g) chart](processes.md#3-derived-quantities-the-sde-chart), so each
parametrization declares `velocity_needs_chart` — `True` except for the
velocity head, whose conversion returns the output untouched. The reverse
processes read it at assembly: a chart-bound head on a chartless
configuration is refused when the `Sampler` is built, while a velocity head
at churn 0 rides the chart-free `ReverseODE`
([sampling.md](sampling.md#1-the-reverse-process-one-family-one-knob)).


## 4. Choosing a head

- **Normalized data on `VP`** → `EpsParametrization` (unit-variance target,
  no weighting needed) or `X0Parametrization`.
- **`VE` at data scale** → `EpsParametrization`, or `ScoreParametrization`
  with the $b^2$ weight; `PseudoForceParametrization` with the clamped
  weight when you want GPFF's time-free sampling and noise-level metering.
- **`FlowMatching`** → `VelocityParametrization`, churn 0.
- **Structured (non-Gaussian) priors, marginal-changing couplings** →
  velocity, x0 or pseudo-force only; the score/eps heads will refuse, and
  [flow_matching_sde.md §8](flow_matching_sde.md) explains what is and is
  not recoverable there.
- **Recovering $x_0$ near $t = 0$ matters** (relaxation, partial
  denoising) → x0 or pseudo-force: their recoveries are direct where the
  generic Tweedie route degenerates.


## 5. Writing a new parametrization

Subclass `Parametrization`, implement `target` and `to_score`, and override
`validate` if the target presumes structure:

```python
class NoisyX0Parametrization(Parametrization):
    """The head predicts x0 + c * x1 (illustrative)."""

    def target(self, process, x0, x1, t, eps=None):
        return x0 + self.c * x1

    def to_score(self, process, output, x_t, t):
        ...  # express the kernel score in terms of the head's output
```

Guidelines:

- `target` takes the endpoint pair, not $x_t$, on purpose: recovering $x_1$
  from $x_t$ means dividing by $b$, which is zero at $t = 0$. The process
  drew $x_1$; handing it over is free, and it leaves every target a
  multiply-add (bar the score's).
- Override `to_velocity`/`to_x0` whenever the direct expression exists —
  the generic route through the score is correct but can be singular where
  the direct one is exact.
- Read everything through the process interface — $a$, $b$, derivatives,
  `process.std`, `process.sigma(t)` — and never touch `process.prior`
  directly. The scale is declared once and read back, not mirrored.
- `eps` is the bridge-noise realization `perturb` drew (`None` for
  $\gamma \equiv 0$ schedules), reserved for the bridge targets that will
  need it.


## 6. Why this design

**Why the field math lives here and not on the process.** A target is the
*definition* of a parametrization, not a property of a noise schedule.
Adding a head must not mean editing `processes.py` — that would be the
prediction axis reaching into the schedule axis, exactly what the
separation exists to prevent. The process is the one interface everything
here reads; in return, a new schedule never touches this file.

**Why stateless, with the pair passed explicitly.** A bound
`(process, parametrization)` object would be convenient but would create a
second place where the pairing lives; consumers taking the pair explicitly
keeps a single assembly point per consumer, where `validate` runs. What the
split gives up is automatic consistency between training and sampling — the
[caller's one obligation](README.md#the-one-obligation-this-leaves-the-caller)
is to share the objects.

**Why everything routes through the score.** One canonical interchange
field means $5$ heads need $5$ conversions, not $5 \times 3$; the overrides
(direct velocity, direct x0) then reclaim exactness where the generic route
is singular. The score is the right hub because it is the field the
reverse-time SDE is written in — and the velocity identity makes the ODE
side equally reachable.

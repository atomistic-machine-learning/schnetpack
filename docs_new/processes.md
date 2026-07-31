# Axis 1 — Processes: the schedule

*Module: `schnetpack.generative.processes` · class `Process` and its
subclasses `VP`, `VE`, `VELinear`, `FlowMatching`, `VPISSNR`.*

A process is the **noising machine** of a generative model: it says how a
clean sample is turned into (a sample from) the prior, continuously in a
path time $t$. Everything else in the subpackage — training targets,
reverse processes, samplers — is derived from it.


## 1. The interpolant

The forward process is the stochastic interpolant

$$
x_t = a(t)\,x_0 + b(t)\,x_1 + \gamma(t)\,\epsilon,
\qquad t \in [t_{\min},\, t_{\max}],
$$

with $x_0$ the data at $t = 0$, $x_1$ the endpoint drawn from the
[prior](priors.md) at $t_{\max}$, the pair matched by the
[coupling](couplings.md), and $\epsilon$ an optional Gaussian *bridge noise*
whose coefficient $\gamma$ is identically zero for every process currently
shipped (`Process.gamma` returns `None`; the hook exists for Schrödinger
bridges and for score access under structured endpoints — see
[flow_matching_sde.md §8.4](flow_matching_sde.md)). So the familiar
two-term form is what actually runs.

Three conventions carry the design:

- **$t = 0$ is data, $t = t_{\max}$ is the prior.** The reverse process
  integrates from $t_{\max}$ down to $t_{\min}$.
- **$b$ is a normalized blending weight**: $0 \le b \le 1$ with
  $b(t_{\max}) = 1$ (exactly, or within the stated $t_{\max}$
  approximation). The schedule says *when* the endpoint takes over — never
  *how large* it is. The endpoint's physical scale belongs to the prior
  ([one owner per fact](README.md#3-one-owner-per-fact)); the process's
  noise level is the product

  $$\sigma(t) = b(t)\,\cdot\,\texttt{prior.std}.$$

  A schedule that put a physical scale into $b$ would be smuggling the
  endpoint's law into the geometry.
- **Times may be per-sample** (shape `(n_samples,)`): samples in one batch
  can sit at different times, which per-sample training and adaptive
  integrators rely on. `expand_t` right-pads a time tensor for broadcasting
  against sample data.

`t_min` and `t_max` bound the usable range because the endpoints are where
schedules misbehave: the score diverges as $b \to 0$, and some diffusion
coefficients blow up at $t_{\max}$ (flow matching's does — see §5).


## 2. Two ways to define a schedule

A subclass fixes the schedule by defining **either** of two pairs; whichever
is given, the other is derived, so both are always available.

**Route 1 — the coefficients $(a, b)$ directly.** The classic vocabulary:
`VP`, `VE`, `FlowMatching` are all written this way, in the literature's own
symbols.

**Route 2 — total variance and log-SNR** (`tv`, `log_snr`), the
disentangled reparametrization of Kahouli et al. (arXiv:2502.08598):

$$
\mathrm{TV}(t) = a^2 + b^2,
\qquad
\log \mathrm{SNR}(t) = \log \frac{a^2}{b^2} .
$$

The inversion goes through the sigmoid,

$$
a^2 = \mathrm{TV}\cdot\mathrm{sigmoid}(\log\mathrm{SNR}),
\qquad
b^2 = \mathrm{TV}\cdot\mathrm{sigmoid}(-\log\mathrm{SNR}),
$$

which is not cosmetic: SNR itself spans the whole positive axis and
overflows a float at the ends of a schedule, where the sigmoid simply
saturates. Note the convention — SNR is the *squared* ratio $a^2/b^2$,
matching Kingma's log-SNR and the TV/SNR reference implementation.

Why the second route exists: the two knobs are **independent**. TV fixes how
large $x_t$ is, SNR fixes how much of it is signal, and neither constrains
the other. In $(a, b)$ coordinates those choices are tangled — changing the
noise level moves the total variance too — which is why historically good
schedules are folklore. A variance-preserving schedule is *exactly* one with
$\mathrm{TV} \equiv 1$. `VPISSNR` is the demonstration that this route is a
real one: it defines `tv` and `log_snr` and nothing else.

Defining neither pair is caught at class-definition time
(`__init_subclass__` raises a `TypeError` naming the class) — the two routes
are mutually recursive, and without the guard the failure would be a
`RecursionError` at the first call, far from its cause.

### Derivatives

`a_dot`, `b_dot`, `log_a_dot`, `log_b_dot`, `log_snr_dot` default to
**autograd through the schedule**, so a subclass need not differentiate by
hand. Overriding them with analytic forms is a speed and precision
optimization, not a requirement; every shipped schedule does, and agreement
with autograd is tested. Two subtleties worth knowing:

- The autograd default returns plain tensors (no graph), so targets built in
  dataloader workers can be pickled. The cost: a schedule with *learnable*
  parameters gets no gradient through the default and must override
  analytically.
- `log_a_dot` is the hook to override whenever $\log a$ is known in closed
  form, because that form usually has no division in it — `VP`'s is simply
  $-\beta/2$, finite even where $a$ underflows and the quotient
  $\dot a / a$ becomes $0/0$.


## 3. Derived quantities: the SDE coefficients

When the endpoint is an independent isotropic Gaussian, the interpolant's
conditional marginals are those of the linear SDE
$\mathrm{d}x = f x\,\mathrm{d}t + g\,\mathrm{d}w$ with

$$
f(t) = \frac{\mathrm{d}}{\mathrm{d}t}\log a(t),
\qquad
g^2(t) = -\,\sigma^2(t)\,\frac{\mathrm{d}}{\mathrm{d}t}\log\mathrm{SNR}(t).
$$

(Derivation by moment matching in
[flow_matching_sde.md §2](flow_matching_sde.md).) The $g^2$ form is chosen
for conditioning: the textbook $2 b\dot b - 2 f b^2$ is a difference of
terms that vanish together, while one log-derivative of one schedule
quantity degenerates nowhere and puts the sign where it can be read —
$g^2 \ge 0$ precisely because signal only ever turns into noise. A constant
endpoint scale multiplies $\sigma^2$ and leaves $\log\mathrm{SNR}$
unchanged, which is why the scale enters as a plain factor.

Two things $g^2$ is *not*: it is not defined for priors that declare no
scalar scale (the `Process.std` property raises with a diagnosis), and it is
not an intrinsic property of the interpolant — it is the canonical choice
that makes the forward drift linear. Samplers decide how much of it to use
via their churn knob; flow matching at churn 0 never evaluates it.


## 4. The forward move: `perturb`

`Process.perturb` composes the three axes, one owner per line:

```python
x1 = prior.sample_like(x0, context)   # what x1 is        (the prior)
x0, x1 = coupling.pair(x0, x1)        # how paired        (the coupling)
x_t = self.interpolate(x0, x1, t)     # when it takes over (the schedule)
```

and returns `(x_t, x0, x1, t, eps)` — the perturbed batch, the (possibly
re-paired) endpoints, the times, and the bridge noise. It **owns every
random draw of the forward side**: for bridge schedules the training target
must see the *same* $\epsilon$ that entered the interpolant, so `perturb`
draws it and hands it back rather than hiding it inside `interpolate`.

Training times come from `sample_t` (uniform on
$[t_{\min}, t_{\max}]$) unless a `t_sampler` hook is supplied to
[`MatchingLoss` or `Diffuse`](training.md) — the same hook that admits the
EDM/GPFF log-normal-$\sigma$ density.

The optional `groups` argument restricts which rows a re-pairing coupling
may exchange endpoints between (for a collated batch of molecules:
`(idx_m, Z)`) — see [couplings.md](couplings.md).


## 5. The Gaussian kernel: a property, not a class

The score and noise training targets, and the closed forms below, are
statements about the one-sided kernel

$$
p(x_t \mid x_0) = \mathcal{N}\!\big(a(t)\,x_0,\ \sigma(t)^2 I\big),
$$

which holds exactly when three configuration facts do:

1. the prior declares its draws isotropic Gaussian (`prior.gaussian`),
2. with a scalar scale (`prior.std` is not `None`),
3. the coupling preserves $x_1$'s marginal, and the schedule carries no
   bridge noise ($\gamma \equiv 0$).

`Process.gaussian_kernel_obstruction()` checks these against the *actual
configuration* and names the first failure as a readable sentence;
`has_gaussian_kernel` is the boolean. Score/noise parametrizations demand it
in their `validate` (called by every consumer constructor), and the
Gaussian-only closed forms call it before answering.

Judging the configuration rather than the class is what lets one schedule
serve both modes: the same `VE` is a Gaussian diffusion under a
`GaussianPrior` and a general stochastic interpolant (Albergo et al.,
arXiv:2303.08797) under a shape prior — with no second class hierarchy, and
no false type-level claims in either mode. See the
[design argument](README.md#2-properties-are-judged-from-the-configuration-not-the-class).

### The closed forms

**`kernel(t)`** returns $(a(t), \sigma(t))$ of the perturbation kernel.

**`posterior(x_t, x0, t, s)`** returns mean and std of the exact Gaussian
posterior $p(x_s \mid x_t, x_0)$ for $s < t$. The linear-Gaussian Markov
structure gives, with $r = a_t/a_s$ and
$\mathrm{var}_{ts} = \sigma_t^2 - r^2 \sigma_s^2$:

$$
\text{mean} = \frac{r\,\sigma_s^2}{\sigma_t^2}\,x_t
            + \frac{a_s\,\mathrm{var}_{ts}}{\sigma_t^2}\,x_0,
\qquad
\text{var} = \frac{\sigma_s^2\,\mathrm{var}_{ts}}{\sigma_t^2} .
$$

For `VP` at the DDPM discretization this is the textbook ancestral
($\tilde\beta$) posterior; for `VE` ($a \equiv 1$) it reduces to
$\text{mean} = (\sigma_s^2/\sigma_t^2) x_t + (1 - \sigma_s^2/\sigma_t^2) x_0$.
This is what [ancestral sampling and DDIM](sampling.md) discretize — with a
model's $x_0$-estimate in place of $x_0$, the step is exact, no score
reconstruction involved.


## 6. The catalog

### `VP` — variance-preserving (continuous-time DDPM)

$$
a(t) = \exp\Big(\!-\tfrac12 \int_0^t \beta\Big),
\quad
b = \sqrt{1 - a^2},
\quad
\beta(t) = \beta_{\min} + \tfrac{t}{t_{\max}}(\beta_{\max} - \beta_{\min}),
$$

giving $f = -\tfrac12\beta$ and (at unit scale) $g^2 = \beta$. Unit-variance
data keeps unit variance for all $t$; the default unit-Gaussian endpoint is
right for normalized data — pass `scale` (the data std) otherwise.
Implementation note: $b$ is computed as
$\sqrt{-\mathrm{expm1}(2\log a)}$, because at small $t$ the naive
$\sqrt{1-a^2}$ loses every digit.

```python
VP(beta_min=0.1, beta_max=20.0)          # the DDPM-linear defaults
```

### `VE` — variance-exploding, geometric noise (score matching / SMLD)

$a \equiv 1$ and $b(t) = b_{\min}^{\,1 - t/t_{\max}}$: the mean never moves
and $b$ climbs geometrically to exactly 1 at $t_{\max}$. The classic
$(\sigma_{\min}, \sigma_{\max})$ schedule is this process with the scale on
the prior, a split the constructor performs once:

```python
VE(sigma_min=0.3, sigma_max=30.0)
# == b_min = sigma_min/sigma_max,  prior = GaussianPrior(std=sigma_max)
# => sigma(t) = sigma_min^(1-t) * sigma_max^t, exactly
```

For a structured endpoint (a GPFF shape prior, a scaffold), pass the
dimensionless route instead: `VE(b_min=1e-2, prior=my_shape_prior)`. Since
$b(0) = b_{\min} > 0$, nothing is singular at $t = 0$ and `t_min` may stay
there.

> **The VE footgun.** Unlike VP, the process is not scale-free:
> $\sigma_{\max}$ (the prior's std) must match your data — the rule of thumb
> is the largest pairwise distance in the dataset. Overshooting does not
> fail loudly: the loss looks fine and the samples are garbage. If sampling
> misbehaves, check $\sigma_{\max}$ first. And a score head on VE wants
> `weight=lambda t: process.b(t)**2` in the loss — see
> [parametrizations.md](parametrizations.md).

### `VELinear` — variance-exploding, straight ramp

$a \equiv 1$, $b = t/t_{\max}$: the Karras et al. (2022) geometry in the
shared normalized convention, with their $\sigma_{\max}$ as the `scale`.
Prefer `VE` for the geometric ramp of classic score matching.

### `FlowMatching` — linear interpolant (rectified flow / OT-FM)

$a = 1 - t$, $b = t$. The velocity target is constant along a pair,
$x_1 - x_0$, which is what makes the learned field straight and few-step
sampling work. Pair with `VelocityParametrization` and churn 0. The prior at
$t = 1$ is exact ($a(1) = 0$), but $g^2 = 2t/(1-t)$ (unit scale) diverges
there, so `t_max` defaults to $1 - 10^{-3}$; pure-ODE users may pass
`t_max=1.0`. The full SDE story is in
[flow_matching_sde.md](flow_matching_sde.md).

### `VPISSNR` — variance-preserving, inverse-sigmoid log-SNR

The headline schedule of the TV/SNR paper, and the one process defined the
second way:

$$
\mathrm{TV}(t) = 1,
\qquad
\log\mathrm{SNR}(t) = \eta\,\log(1/t - 1) + \kappa .
$$

$a, b$ and all derivatives follow from the base class — the subclass is the
whole schedule. $\eta$ sets how fast signal turns into noise, $\kappa$
shifts where the schedule sits, and $\mathrm{TV} = 1$ pins the total
variance regardless. With the paper's companion TV schedule, $\eta = 2$,
$\kappa = 0$ is exactly optimal-transport flow matching; this class is its
variance-preserving sibling. (The paper quotes $\eta = 1$ because its text
defines SNR unsquared; the factor of two lands on $\eta$.)


## 7. Writing a new schedule

```python
class Cosine(Process):
    """a = cos(pi t / 2), b = sin(pi t / 2) — the Nichol-Dhariwal shape."""

    def a(self, t):
        return torch.cos(0.5 * torch.pi * t / self.t_max)

    def b(self, t):
        return torch.sin(0.5 * torch.pi * t / self.t_max)
```

That is a complete, working process: derivatives come from autograd, $f$ and
$g^2$ from the identities, the closed forms from the Gaussian-kernel
machinery, the endpoint and pairing from the constructor arguments it
inherits. Checklist for going further:

- Define **both members of one pair** — `(a, b)` or `(tv, log_snr)`. One
  member alone leaves the other pair's derivation recursive (the class-level
  guard will tell you).
- Keep $b$ **normalized** ($b(t_{\max}) = 1$, dimensionless). If your
  schedule has a scale knob, it belongs on the prior — accept `scale`/
  `prior` in your constructor and pass them through, as every shipped
  schedule does.
- Override derivatives analytically **if** the schedule is hot or has
  learnable parameters; test against the autograd default.
- Choose `t_min`/`t_max` defaults away from any singularity of *your*
  coefficients, and say in the docstring which consumer the guard protects.


## 8. Why this design

**Why the schedule is the subclass, and the endpoint/pairing are
arguments.** A schedule *is* the mathematical identity of the process — `VE`
means "geometric noise ramp" in every paper — while endpoint and pairing
vary independently of it (the same `VE` runs Gaussian or shape-prior). Making
the varying parts constructor arguments avoids a class per combination; the
subclass carries only what defines it, spelled in the literature's
vocabulary (`VE(sigma_min=0.3, sigma_max=30.0)` reads like the paper).

**Why the interpolant is the primitive rather than $(f, g)$.** Training
needs $x_t$ at a random $(x_0, t)$ in one shot, which the interpolant gives
and the SDE would make you integrate; $(f, g)$ derive from the schedule by
differentiation, whereas the converse costs an ODE solve per schedule — and
for flow matching there is no intrinsic $g$ to start from.

**Why the Gaussian kernel is judged, not subclassed.** Because it is decided
by constructor arguments, a type could not check it; and because it cuts
*across* the schedule axis, a hierarchy would duplicate every schedule into
Gaussian and non-Gaussian variants. See the
[README](README.md#the-design-argument) for the full argument, and
[flow_matching_sde.md](flow_matching_sde.md) for what mathematically
survives on the far side of the boundary.

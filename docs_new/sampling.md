# Sampling: reverse processes, integrators, grids, samplers

*Modules: `schnetpack.generative.sde`, `.reverse`, `.integrators`,
`.grids`, `.sampler` · classes `ReverseSDE`, `ReverseODE` (assembled by
`reverse()`), `Integrator`, `EulerMaruyama`, `Heun`, `Ancestral`,
`AncestralDDPM`, `TimeGrid`, `UniformGrid`, `Sampler`,
`DirectDenoisingSampler`.*

Generation runs the forward process backwards. The machinery factors the
same way as the forward side: **what** is integrated (the reverse process,
derived — never implemented per schedule), **how** each step is computed
(the integrator), **where** the steps go (the grid), and the thin
composition that binds them (the sampler).

```
prior.sample() ──> x at t_max ──[ integrator steps on the reverse process ]──> x at t_min
                                     │ grid: which times      │
                                     │ churn: how stochastic  │
```


## 1. The reverse process: one family, one knob

For a forward process with drift $f x$ and diffusion $g$
([the SDE chart](processes.md#3-derived-quantities-the-sde-chart)),
the time reversal (Anderson 1982) and the probability-flow ODE are the two
ends of a one-parameter family, written around the **velocity**:

$$
\mathrm{d}x
= \Big[\, v(x, t) \;-\; \tfrac12\,\chi\, g^2(t)\, s(x, t) \,\Big]\mathrm{d}t
\;+\; \sqrt{\chi}\; g(t)\,\mathrm{d}\bar w,
\qquad \chi \in [0, 1],
$$

integrated from $t_{\max}$ down to $t_{\min}$ (integrators receive
$\mathrm{d}t < 0$). Every member shares the forward marginals — the extra
drift and the extra noise cancel in Fokker–Planck
([flow_matching_sde.md §6](flow_matching_sde.md)) — so `churn` $= \chi$
moves the *path measure*, never the distribution being sampled. $\chi = 1$
is the reverse-time SDE, $\chi = 0$ the PF-ODE, and the knob maps onto the
usual $\eta$ by $\chi = \eta^2$.

Writing the family around the velocity rather than the score is a
deliberate choice: the route from a velocity back to a score divides by
$g^2$ and blows up as $t \to 0$, and on the ODE path that route is **never
taken**. Flow matching therefore costs nothing it shouldn't. At $\chi > 0$
the score is a *second conversion of the same raw output* — never a second
model call.

Reverse processes are never implemented per schedule; they split by
**capability** instead, and the factory
`reverse(process, parametrization, model, churn, cond, require_sde)` picks:

- **`ReverseSDE(process.sde(), ...)`** — everything that crosses the
  [(f, g) chart](processes.md#3-derived-quantities-the-sde-chart):
  churn $> 0$ outright, and churn $= 0$ for any non-velocity head (the
  PF-ODE drift converts through $f$ and $g^2$). Taking the `SDE` in its
  constructor means it *cannot exist* for a configuration without the
  Gaussian kernel — the refusal happens at assembly and names the
  obstruction. Exposes `drift(x, t)`/`diffusion(t)` for the generic
  integrators, `score(x, t)`/`x0(x, t)` for the ancestral ones, and
  `g2(t)`/`sde` read through to the chart.
- **`ReverseODE(process, ...)`** — continuity-equation transport
  $\mathrm{d}x = v\,\mathrm{d}t$, valid for **any** endpoint law
  ([flow_matching_sde.md §8.2](flow_matching_sde.md)); accepts only
  chart-free velocity heads. This is vanilla flow matching's path, and the
  reason shape-prior velocity sampling never touches the chart.

Method-specific behavior belongs in the composed parts.


## 2. Integrators

An integrator advances the state one step and is agnostic to what it
integrates — it consumes only `drift` and `diffusion` (a reverse SDE, a
PF-ODE, or even a forward process). The base class supplies the
`integrate(process, x, ts)` loop over a monotone time grid.

### `EulerMaruyama` — first order

$x \leftarrow x + \text{drift}\cdot\mathrm{d}t + g\sqrt{|\mathrm{d}t|}\,z$.
For $g = 0$ this is plain Euler. The robust default; needs relatively many
steps.

### `Heun` — second order

A Heun (trapezoidal) step on the drift, with the diffusion contribution
added as an Euler–Maruyama increment. On the PF-ODE ($\chi = 0$) this is
the deterministic second-order sampler popularized by EDM (Karras et al.
2022): comparable quality at far fewer model evaluations than first order.
Costs two drift evaluations (= two model calls) per step.

### `Ancestral` — the exact-posterior step

The generic ancestral update: estimate $x_0$ from the model, then draw from
the closed-form Gaussian posterior the process already knows,

$$
\hat x_0 = \texttt{reverse.x0}(x, t),
\qquad
x_s \sim p(x_s \mid x_t, \hat x_0)
\quad\text{via } \texttt{SDE.posterior}.
$$

Schedule logic lives entirely in
[the closed form](processes.md#the-closed-forms), so one class covers every
process with a Gaussian kernel: on `VP` it is the textbook DDPM ancestral
step with the exact ($\tilde\beta$) posterior variance; on `VE` it reduces
to the familiar NCSN/GPFF ancestral update. Declares `requires_sde`, so
the `Sampler` assembles a `ReverseSDE` even at churn 0 and a configuration
without the Gaussian kernel is refused at assembly. Works with any head
that can produce an $x_0$-estimate, and — being intrinsically stochastic —
**ignores `churn`**. Note this also means ancestral sampling is perfectly
valid for a flow-matching model under its default Gaussian prior.

### `AncestralDDPM` — the score-form DDPM step

The exact discrete-time DDPM update written in terms of the raw score,

$$
x_{k-1} = \frac{x_k + \beta_k\, s}{\sqrt{1 - \beta_k}} + \sqrt{\beta_k}\,z,
\qquad \beta_k = g^2(t)\,\lvert\mathrm{d}t\rvert,
$$

with the DDPM $\sigma_t^2 = \beta_t$ variance choice. A deliberate
exception to the drift/diffusion rule: the step *is* a statement about the
score, and rewriting it through the drift would only obscure it. Also
intrinsically stochastic — ignores `churn`.

> **Warning:** the update's algebra assumes a **unit-scale VP path** — the
> single number $\beta_k$ serves as both the variance increment and the
> mean contraction, which requires $g^2(t) = -2 f(t)$ (an identity for any
> unit-scale variance-preserving schedule, false elsewhere). On `VE`,
> `FlowMatching`, or a `VP(scale≠1)` it runs without error and produces a
> subtly wrong discretization. Pairing it correctly is currently the
> caller's responsibility; `Ancestral` subsumes it (on unit VP the exact
> posterior *is* the $\tilde\beta$ DDPM step) and is safe everywhere it
> constructs.

### Choosing

| situation | integrator, churn |
| --- | --- |
| flow matching, few steps | `Heun`, churn 0 (or `EulerMaruyama` for very few, cheap steps) |
| classic DDPM behavior | `Ancestral` (exact posterior), or `EulerMaruyama` at churn 1 |
| NCSN/GPFF-style annealed Langevin flavor | `Ancestral` on `VE` |
| error-tolerant long runs, many steps | `EulerMaruyama`, churn $\in (0, 1]$ — stochasticity re-contracts accumulated error |
| quality per model call is the metric | `Heun`, churn 0, on a warped grid |


## 3. Time grids

`TimeGrid` builds the sequence of times a sampler steps through —
`grid(t_start, t_end, n_steps)` returning a monotone `(n_steps + 1,)`
tensor, decreasing when denoising. Separating the grid from the solver means
*how many steps and where* is independent of *what each step computes*: a
uniform grid wastes steps at high noise (where the reverse process barely
moves) and starves the low-noise end (where the detail appears); a warped
grid just moves them without touching the integrator.

`UniformGrid` is the shipped default and the right choice for VP-type
paths. EDM-style warped grids (the $\rho$-schedule) are a `TimeGrid`
subclass away — the seam exists precisely so they never touch a solver.


## 4. `Sampler` — the composition

A thin wrapper: prior → reverse process → integrator.

```python
sampler = Sampler(
    process,             # the SAME process the model trained under
    parametrization,     # the SAME head contract
    integrator=Heun(),
    grid=None,           # default UniformGrid()
    prior=None,          # default: derived from the process (see below)
    churn=0.0,
    t_min=None, t_max=None,   # default: the process's own bounds
)

samples = sampler.sample(model, shape=(n, ...), n_steps=50,
                         cond=None, context=None)
```

Design points worth knowing:

- **The starting distribution is derived, not asked for.**
  `process.sampling_prior()` returns the training prior itself whenever the
  coupling preserves $x_1$'s marginal — the same object, not a restatement,
  so train/sample starts cannot drift apart. An explicit `prior=` overrides
  it, and is *required* when the coupling changes the marginal (the process
  refuses to guess). See [priors.md](priors.md) and
  [couplings.md](couplings.md).
- **`context` flows to the prior** exactly as during training — a
  `CenteredGaussianPrior`-style endpoint reads the molecule layout
  (`idx_m`) out of it, so per-molecule centering works on both sides.
- **The assembly is validated at construction**: the pairing via
  `parametrization.validate(process)`, and — whenever churn $> 0$, the head
  is not a chart-free velocity, or the integrator declares `requires_sde` —
  the chart via `process.sde()`. An invalid assembly fails when built, with
  the obstruction named; nothing is left to fail mid-run or, worse, to
  return plausible wrong numbers.
- **`denoise(model, x_t, t_start, n_steps)`** is the partial-denoising
  entry point: relaxation of given structures, scaffolded generation, and
  structured priors that start below $t_{\max}$ all enter here — `sample`
  is just `denoise` from a prior draw at $t_{\max}$.
- If you find yourself subclassing `Sampler`, the logic probably belongs in
  a process, parametrization, integrator or grid — that is what the axes
  are for.

`generate()` — the "model in, ASE Atoms out" convenience on top of this —
lands with the M1.3 milestone, together with the `spkgenerate` CLI.


## 5. `DirectDenoisingSampler` — GPFF's time-free loop

GPFF's direct denoising is not an integrator on a time grid, which is why it
is a *sibling* of `Sampler` rather than a part of one. Each of `n_steps`
iterations does

$$
x \leftarrow x + \lambda\,\big(1 - k/N\big)\,z, \quad z \sim \mathcal{N}(0, I)
\qquad\text{(decaying injection)},
$$
$$
x \leftarrow \hat x_0(x)
\qquad\text{(jump to the model's } x_0\text{-estimate)}.
$$

There is no time grid, no reverse SDE/ODE, no noise schedule at sampling
time; the only ingredients are `parametrization.to_x0` and the injection.
`stochastic_lambda` is in **data units** (Å for positions); 0 disables the
injection (plain direct denoising), positive values buy sample diversity.

The model is evaluated at $t = 0$ throughout — the sampler never knows the
noise level of its iterate. It therefore presumes the **time-free
contract** that makes GPFF possible: a model that ignores its $t$ input,
under a head whose `to_x0` never reads $t$ either. The pseudo-force and x0
heads satisfy that (their recoveries are
[division- and $\sigma$-free](parametrizations.md#2-the-catalog)); a
score-type head divides by $\sigma(t)$ and would read the lie.
Time-conditioned models belong in `Sampler` — including
`Sampler.denoise` for relaxing structures whose noise level you *do* know.

The flip side of time-freeness: on a `VE` path the pseudo-force magnitude
itself carries $\sigma$, so the model meters its own noise level from the
input — which is exactly why this loop can denoise structures of *unknown*
noisiness, its home turf.


## 6. How the pieces meet: a worked example

DDPM, exactly, from parts — then two one-line pivots:

```python
process = VP()                             # beta-linear, unit Gaussian endpoint
param   = EpsParametrization()

sampler = Sampler(process, param, Ancestral())          # textbook DDPM
samples = sampler.sample(model, (64, 3), n_steps=1000)

# pivot 1: deterministic few-step sampling of the SAME model
sampler = Sampler(process, param, Heun(), churn=0.0)    # PF-ODE
samples = sampler.sample(model, (64, 3), n_steps=30)

# pivot 2: interpolate stochasticity
sampler = Sampler(process, param, EulerMaruyama(), churn=0.3)
```

The same trained checkpoint serves all three — the sampler family shares
the marginals the model learned, and the churn knob, the integrator and the
grid are pure inference-time choices. That is the practical payoff of
[deriving the reverse process](README.md#5-the-interpolant-is-the-primitive-the-sde-is-derived)
instead of implementing it per method.

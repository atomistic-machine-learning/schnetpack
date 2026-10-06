# Sampling: reverse processes, integrators, grids, samplers

*Modules: `schnetpack.generative.differential_equations`,
`schnetpack.dynamics` (`.sample`, `.optimize`, `.calculator`, `.hooks`,
`.guidance`) · classes `SDE`, `ReverseSDE`, `ReverseODE`, `Sampler`,
`EulerMaruyama`, `Heun`, `Ancestral`, `TimeGrid`, `UniformGrid`, `Dynamics`,
`DirectDenoising`, `Calculator`, `GenerativeCalculator`, `ForceCalculator`,
`Hook`, `FreezeScaffold`, `Guidance`.*

Generation runs the forward process backwards. The machinery factors the
same way as the forward side: **what** is integrated (the reverse process,
derived — never implemented per schedule), **how** each step is computed
(the integrator), **where** the steps go (the grid), and the thin
composition that binds them (the sampler).

```
prior.sample(n) ──> x at t_max ──[ integrator steps on the reverse process ]──> x at t_min
                                     │ grid: which times      │
                                     │ eta2: how stochastic  │
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
([flow_matching_sde.md §6](flow_matching_sde.md)) — so `eta2` $= \chi$
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
**capability** instead:

- **`ReverseSDE(process.sde(), eta2)`** — everything that crosses the
  [(f, g) chart](processes.md#3-derived-quantities-the-sde-chart). Taking
  the `SDE` in its constructor means it *cannot exist* for a configuration
  without the Gaussian kernel — the refusal happens at assembly and names
  the obstruction. The `Sampler` builds one at construction and steps on the
  calculator's score: `drift(x, t, score)`/`diffusion(t)` for the generic
  integrators, the chart's `x0_from_score`/`posterior` for the ancestral
  one.
- **`ReverseODE(process, ...)`** — continuity-equation transport
  $\mathrm{d}x = v\,\mathrm{d}t$, valid for **any** endpoint law
  ([flow_matching_sde.md §8.2](flow_matching_sde.md)); accepts only
  chart-free velocity heads. This is vanilla flow matching's path, and the
  reason shape-prior velocity sampling never touches the chart. No shipped
  sampler steps on it yet.

Method-specific behavior belongs in the composed parts.


## 2. Integrators

An integrator is a `Sampler` subclass: its `step(batch, x, t, dt)` advances
the state one grid interval on the reverse process's `drift` and
`diffusion` and the calculator's score. It has no loop of its own: the
`Sampler` base calls `step` once per grid interval inside its `run` loop,
which is where hooks run (§7).

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
to the familiar NCSN/GPFF ancestral update. Like every sampler it assembles
a `ReverseSDE`, so a configuration without the Gaussian kernel is refused
at assembly. Works with any head
that can produce an $x_0$-estimate, and — being intrinsically stochastic —
**ignores `eta2`**. Note this also means ancestral sampling is perfectly
valid for a flow-matching model under its default Gaussian prior.

### Choosing

| situation | integrator, eta2 |
| --- | --- |
| flow matching, few steps | `Heun`, eta2 0 (or `EulerMaruyama` for very few, cheap steps) |
| classic DDPM behavior | `Ancestral` (exact posterior), or `EulerMaruyama` at eta2 1 |
| NCSN/GPFF-style annealed Langevin flavor | `Ancestral` on `VE` |
| error-tolerant long runs, many steps | `EulerMaruyama`, eta2 $\in (0, 1]$ — stochasticity re-contracts accumulated error |
| quality per model call is the metric | `Heun`, eta2 0, on a warped grid |


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
# inference: device, dtype, neighbor list, grad policy (§7), and the pair
calculator = GenerativeCalculator(
    model,               # batch -> outputs, or a path to a saved model
    process,             # the SAME process the model trained under
    parametrization,     # the SAME head contract
    output_key="eps_pred",    # model output holding the raw head
    device="cuda",
)

sampler = Heun(          # the integrator is the Sampler subclass
    calculator,
    grid=None,           # default UniformGrid()
    prior=None,          # default: derived from the process (see below)
    eta2=0.0,
)

# 64 structures from the prior's structures (e.g.
# GaussianPrior(structures=StatisticsStructures.from_dataset(train)), given
# to the process or the sampler), positions drawn
out = sampler.sample(64, n_steps=50)

# or redraw the positions of a batch of your own (a test-set batch, a template
# with Z, n_atoms, idx_m and any conditioning keys)
out = sampler.run(sampler.prior.sample_from_batch(batch), n_steps=50)
positions = out[properties.R]
```

Design points worth knowing:

- **The starting distribution is derived, not asked for.**
  `process.sampling_prior()` returns the training prior itself whenever the
  coupling preserves $x_1$'s marginal — the same object, not a restatement,
  so train/sample starts cannot drift apart. An explicit `prior=` overrides
  it, and is *required* when the coupling changes the marginal (the process
  refuses to guess). See [priors.md](priors.md) and
  [couplings.md](couplings.md).
- **The batch dict is the whole interface** (§7). The model is
  `batch -> outputs`, like any `NeuralNetworkPotential`, reached through a
  `GenerativeCalculator`: the time arrives
  under `properties.t` (the key `Diffuse` writes in training), conditioning
  keys simply stay in the batch, and the raw head is read from
  `outputs[output_key]`. The prior receives the batch as its context, so a
  `GaussianPrior` reads the molecule layout (`idx_m`) out of it and
  per-molecule centering works exactly as during training.
- **The assembly is validated at construction**: the pairing via
  `parametrization.validate(process)` in the calculator, and the chart via
  `process.sde()` in the sampler, which always steps on the score. An invalid assembly fails when built, with
  the obstruction named; nothing is left to fail mid-run or, worse, to
  return plausible wrong numbers.
- **`run(batch, n_steps, t_start=None)`** is the partial-denoising
  entry point: relaxation of given structures, scaffolded generation, and
  structured priors that start below $t_{\max}$ all enter here — `sample`
  is just `run` from a prior draw at $t_{\max}$.
- If you find yourself subclassing `Sampler`, the logic probably belongs in
  a process, parametrization, step rule or grid — that is what the axes
  are for.

`generate()` — the "model in, ASE Atoms out" convenience on top of this —
lands with the M1.3 milestone, together with the `spkgenerate` CLI.


## 5. `DirectDenoising` — GPFF's time-free loop

GPFF's direct denoising is not an integrator on a time grid, which is why it
is an `Optimizer` rather than a `Sampler`. Each of `n_steps` iterations
jumps to the model's $x_0$-estimate,

$$
x \leftarrow x + \tfrac12 F(x) = \hat x_0(x),
\qquad F = 2\,(\hat x_0 - x)
\quad\text{(the pseudo-force)}.
$$

There is no time grid, no reverse SDE/ODE, no noise schedule at sampling
time; the only ingredient is the pseudo-force, which a
`ForceCalculator(model, kind="pseudo")` returns in Å. Being an `Optimizer`,
it stops early once every structure's largest pseudo-force is below `fmax`
(Å), and holds fixed atoms. It injects no noise.

The model is evaluated at $t = 0$ throughout — the sampler never knows the
noise level of its iterate. It therefore presumes the **time-free
contract** that makes GPFF possible: a model that ignores its $t$ input,
under a head whose `to_x0` never reads $t$ either. The pseudo-force and x0
heads satisfy that (their recoveries are
[division- and $\sigma$-free](parametrizations.md#2-the-catalog)); a
score-type head divides by $\sigma(t)$ and would read the lie.
Time-conditioned models belong in `Sampler` — including
`Sampler.run` for relaxing structures whose noise level you *do* know.

The flip side of time-freeness: on a `VE` path the pseudo-force magnitude
itself carries $\sigma$, so the model meters its own noise level from the
input — which is exactly why this loop can denoise structures of *unknown*
noisiness, its home turf.


## 6. How the pieces meet: a worked example

DDPM, exactly, from parts — then two one-line pivots:

```python
stats   = StatisticsStructures.from_dataset(train)            # compositions to generate
process = VP(prior=GaussianPrior(structures=stats))          # unit Gaussian endpoint
param   = EpsParametrization()
calc    = GenerativeCalculator(model, process, param)        # the trained checkpoint

sampler = Ancestral(calc)                                     # textbook DDPM
samples = sampler.sample(64, n_steps=1000)

# pivot 1: deterministic few-step sampling of the SAME model
sampler = Heun(calc, eta2=0.0)                                # PF-ODE
samples = sampler.sample(64, n_steps=30)

# pivot 2: interpolate stochasticity
sampler = EulerMaruyama(calc, eta2=0.3)
```

The same trained checkpoint serves all three — the sampler family shares
the marginals the model learned, and the eta2 knob, the integrator and the
grid are pure inference-time choices. That is the practical payoff of
[deriving the reverse process](README.md#5-the-interpolant-is-the-primitive-the-sde-is-derived)
instead of implementing it per method.


## 7. The loop, the batch, the calculator and hooks

`Sampler` and the optimizers (`DirectDenoising`, `LBFGS`, ...) are all
`Dynamics`. `Dynamics` holds only the calculator, an optional prior, the
moved key and the hooks, so non-generative drivers subclass it without a
process, parametrization or time key. The generative pair, `output_key` and
`time_key` live on the `GenerativeCalculator`, and the sampler defaults the
prior to the process's sampling prior. The structure they move is the batch
dict — the one datasets, transforms and models use — and nothing else. Each
driver holds its model as `self.calculator`, and each family's base writes
the loop out in `run(batch, n_steps)` (the sampler adds `t_start=None`),

```
self.calculator.reset()
batch = self.calculator.prepare(batch)       # to the run's device/dtype, once
batch = {**batch, time_key: t_0}             # sampler only
for i in range(n_steps):
    batch = self.before_step(batch, i, n_steps)       # hooks, in order
    batch = <one step>                                # self.step(...)
    batch = self.after_step(batch, i + 1, n_steps)    # hooks, in order
return batch
```

and `sample(n_samples, n_steps)`, shared by every driver, draws
`n_samples` starting structures from the prior first
(`prior.sample(n_samples)`). To start from a batch of your own, redraw its
positions with `prior.sample_from_batch(batch)` and call `run`; a
relaxation from stored structures passes `prior=DatasetPrior(dataset)`. Per-run data (the sampler's time grid, the optimizer's step-rule history) are locals of that loop.

The first constructor argument of every driver is its calculator; the
batch contract is the keyword `key` of `Dynamics` (the sampler takes it
from its calculator) and the keywords of the `GenerativeCalculator`:

| keyword | default | meaning |
|---|---|---|
| `key` | `properties.R` | batch key the driver moves. Process, parametrization and step rule stay pure tensor math on it; the driver reads it and writes it back. Joint iterates (positions + cell + types) need per-key processes and are not built yet. |
| `output_key` | `"prediction"` | model output holding the raw head |
| `time_key` | `properties.t` | where the path time goes, one value per row of the moved key (grid time on the sampler, zeros on a pseudo-force calculator) |

**Inference goes through a `Calculator`**
(`Calculator(model, neighbor_list=None, device=None, dtype=None, enable_grad=False, cache_last=False, guidance=())`):
it moves the batch to the run's device/dtype once at loop entry
(`prepare`), rebuilds the neighbor list on every call, sets the gradient
policy (off for generative heads, on for models that differentiate an
energy) and calls the model. It works on a shallow copy, so neighbor lists,
`Rij` and outputs never land in the driver's batch — which is why nothing
has to invalidate them when the structure moves. Keys the driver does not
move are carried along as given: a static, fully connected neighbor list in
the template simply stays valid, while a cutoff list needs `neighbor_list=`.
Output caching is off by default (Heun evaluates two structures per step);
the optimizers turn on `cache_last`, since the stop test and the step ask
about the same positions. Its per-run state is cleared by `reset()` at the
start of every run.

**Guidance lives on the calculator.** A `Guidance` term
(`HarmonicRestraint`, or kT ∇ log p of a classifier) returns one force-like
term in eV/Å; the calculator adds the weighted sum to the field it returns
— to the forces of a `ForceCalculator`, to the score (and through it to x0
and the velocity) of a `GenerativeCalculator`. The drivers never see it as
a separate part: the field they step on is already guided. Passing a
guidance term as a hook raises a `TypeError`.

A `Hook` edits the batch **before** a step (what the model sees)
or **after** it (what the step produced), with
`before_step(batch, step, n_steps)` / `after_step(...)` that return
a new dict. `step` counts completed steps.
Hooks run between full steps only, never between the stages of Heun;
they run in list order, so a hook that overwrites rows belongs after
one that perturbs them.

- **`FreezeScaffold(key=properties.R, mask_key=properties.fixed_atoms, reference_key=properties.R_reference)`**
  — holds the atoms flagged in the per-atom mask at the reference positions.
  Both are batch keys, so they collate with the structures and each molecule
  may carry its own scaffold. The scaffold rows of `key` are overwritten
  with the reference before and after every step, on any driver: the model
  always sees the clean scaffold, and the run ends on it.

```python
template[properties.fixed_atoms] = mask            # bool, one per atom
template[properties.R_reference] = reference       # (n_atoms, 3); masked rows read
gpff = DirectDenoising(ForceCalculator(model, kind="pseudo"),
                       prior=process.sampling_prior(),
                       hooks=[FreezeScaffold()])
out = gpff.run(gpff.prior.sample_from_batch(template), n_steps=100)
```

On a time-aware sampler a clean scaffold sits off the noise manifold at
$t$, so the model sees inputs it never saw in training; that is the price of
overwriting rather than re-noising. On a time-free loop any edit is fine.
Overwriting rows also breaks the zero center of geometry a centered
`GaussianPrior` guarantees: give the reference in the frame the model
expects.

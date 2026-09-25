# Sampling overview: what goes where

*A one-page map of the sampling-side objects and their connections, after
the field-binding refactor (2026-08-01). Detailed treatment:
[sampling.md](sampling.md); math grounding:
[flow_matching_sde.md](flow_matching_sde.md).*

## The pieces, one line each

| Object | Home | Owns exactly | Knows nothing about |
|---|---|---|---|
| `Process` (VE, VP, FM, …) | `processes.py` | the interpolant chart: schedules a/b, `sigma`, `perturb`, `sampling_prior`, the kernel check | models, sampling, integrators |
| `SDE` | `differential_equations.py` | the (f, g) chart: `f`, `g2`, `kernel`, `posterior`, `x0_from_score`; constructing it **is** the Gaussian-kernel check | models, parametrizations |
| `ReverseSDE` | `differential_equations.py` | Anderson family on the chart: `drift = f x − ½(1+churn) g² s`, `diffusion`, `score`, `g2` | how the score was produced |
| `ReverseODE` | `differential_equations.py` | transport `drift = v`, `diffusion = 0` | everything else — chart-free by design |
| `Parametrization` | `parametrizations.py` | head contract: training `target`, conversions `to_score` / `to_velocity` / `to_x0` | integrators, samplers |
| `Integrator` (Euler, Heun) | `dynamics/integrators/` | one numerical step on `dynamics.drift` / `dynamics.diffusion` | what the dynamics is made of |
| `Ancestral`, `AncestralDDPM` | `dynamics/integrators/` | exact-posterior / DDPM steps through the chart's closed forms (`requires_sde = True`) | parametrizations (they see only `score` and the chart) |
| `TimeGrid` | `dynamics/sampling/grids.py` | where the steps land | everything else |
| `Prior` | `priors.py` | the x1 endpoint law; the start state | everything else |
| `Dynamics` | `dynamics/base.py` | the calculator, the optional prior and its draw in `sample`, the moved `key`, the constraint hooks every driver's own `for` loop (abstract `denoise`) calls around each step | generative models, what one step is |
| `Calculator` | `dynamics/calculator.py` | inference: device/dtype (`prepare`, once per run), neighbor list (every call, on a copy), grad policy, the model call | processes, steps |
| `StateConstraint` (`AnnealedNoise`, `Scaffold`) | `dynamics/constraints/state.py` | edits of the batch before and/or after a step | models, integrators |
| `FieldConstraint` | `dynamics/constraints/field.py` | changes to the field a step follows (restraints, guidance); base class only for now | the state between steps |
| `Sampler` | `dynamics/sampling/sampler.py` | **the assembly** (a `Dynamics`): decides; holds the pair, `output_key`, `time_key`; defaults the prior to the process's `needs_chart`, binds the field, picks the reverse class; one step = one integrator step on the grid | numerics (delegated), field math (delegated) |
| `DirectDenoising` | `dynamics/relax/direct_denoising.py` | GPFF's jump-to-`to_x0` step (a time-free `Dynamics`); its noise injection is an `AnnealedNoise` constraint | the entire reverse machinery — deliberately outside it |

## The chart of the connections

```
CONFIGURATION (held by Sampler, checked at construction)

  Process ──── prior, coupling, schedule
     │
     │ .sde()  — succeeds iff the Gaussian kernel holds;
     │           refusal names the obstruction
     ▼
  SDE chart (f, g², kernel, posterior, x0_from_score)


ASSEMBLY (Sampler.reverse, once per step — calculator and batch arrive here)

  calculator ──────┐  calculator({**batch, R: x, t: t}) → neighbor list → model
  parametrization ─┤  bound into ONE field callable (x, t) → raw
  output_key ──────┘
        │
        ├─ needs_chart?  churn > 0  OR  velocity_needs_chart  OR  integrator.requires_sde
        │
        ├─ yes:  score_fn(x,t) = to_score(process, model(...), x, t)
        │        dynamics = ReverseSDE(process.sde(), score_fn, churn)
        │
        └─ no:   velocity_fn(x,t) = to_velocity(process, model(...), x, t)
                 dynamics = ReverseODE(velocity_fn)          ← never touches the chart


INTEGRATION (Sampler.denoise: a for loop, one integrator step per iteration)

  prior.sample ──> x(t_max)          grid ──> ts (t_max → t_min)
        │                                        │
        ▼                                        ▼
  for i:  batch = before_step(batch, i, n)              batch[t] = ts[i]
         x = integrator.step(dynamics, x, t, dt)         dt < 0
         batch = after_step(batch, i + 1, n)             batch[t] = ts[i+1]
             │
             ├─ Euler/Heun:      dynamics.drift, dynamics.diffusion
             ├─ Ancestral:       s = dynamics.score
             │                   x0̂ = dynamics.sde.x0_from_score(x, s, t)
             │                   mean, std = dynamics.sde.posterior(x, x0̂, t, t+dt)
             └─ AncestralDDPM:   dynamics.g2, dynamics.score
```

The chart-free lane, entirely apart:

```
  DirectDenoising:  loop { AnnealedNoise (before-step constraint);
                           x = to_x0(process, model(x,0), x, 0) }
  — no grid, no reverse object, no chart; the GPFF path. Same Dynamics
    constraint hooks, so the same constraints (Scaffold) apply.
```

## The three load-bearing rules

1. **Reverse objects take bound fields, not (model, parametrization, batch).**
   `ReverseSDE` is pure chart math over a `score_fn(x, t)`; `ReverseODE` is
   pure transport over a `velocity_fn(x, t)`. The composition of head +
   conversion + batch happens once per step, in `Sampler.reverse` — or by
   hand, if you hold a ready field:
   `ReverseSDE(process.sde(), score_fn, churn=1.0)`.

2. **The chart is acquired where it is needed, and acquisition is the
   check.** One condition — `churn > 0 or velocity_needs_chart or
   integrator.requires_sde` — decides both the eager check in
   `Sampler.__init__` (fail at assembly, obstruction named) and the class
   picked in `reverse()`. Configurations without the kernel (shape prior,
   value-dependent coupling) keep the chart-free lanes: velocity head at
   churn = 0, and direct denoising.

3. **Integrators see only the dynamics.** The generic solvers consume
   `drift`/`diffusion` and work on either reverse class. The ancestral pair
   declares `requires_sde` and additionally reads `score` plus the chart's
   closed forms — the one sanctioned widening of the contract.

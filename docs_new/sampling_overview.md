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
| `Integrator` (Euler, Heun) | `integrators/` | one numerical step on `dynamics.drift` / `dynamics.diffusion` | what the dynamics is made of |
| `Ancestral`, `AncestralDDPM` | `integrators/` | exact-posterior / DDPM steps through the chart's closed forms (`requires_sde = True`) | parametrizations (they see only `score` and the chart) |
| `TimeGrid` | `grids.py` | where the steps land | everything else |
| `Prior` | `priors.py` | the x1 endpoint law; the start state | everything else |
| `Sampler` | `sampler.py` | **the assembly**: validates the pair, decides `needs_chart`, binds the field, picks the reverse class, runs the integrator | numerics (delegated), field math (delegated) |
| `DirectDenoisingSampler` | `sampler.py` | GPFF's inject-noise / jump-to-`to_x0` loop | the entire reverse machinery — deliberately outside it |

## The chart of the connections

```
CONFIGURATION (held by Sampler, checked at construction)

  Process ──── prior, coupling, schedule
     │
     │ .sde()  — succeeds iff the Gaussian kernel holds;
     │           refusal names the obstruction
     ▼
  SDE chart (f, g², kernel, posterior, x0_from_score)


ASSEMBLY (Sampler.denoise, once per call — model arrives here)

  model(x,t,cond) ─┐
  parametrization ─┤  bound into ONE field callable
  cond ────────────┘
        │
        ├─ needs_chart?  churn > 0  OR  velocity_needs_chart  OR  integrator.requires_sde
        │
        ├─ yes:  score_fn(x,t) = to_score(process, model(...), x, t)
        │        dynamics = ReverseSDE(process.sde(), score_fn, churn)
        │
        └─ no:   velocity_fn(x,t) = to_velocity(process, model(...), x, t)
                 dynamics = ReverseODE(velocity_fn)          ← never touches the chart


INTEGRATION (integrator.integrate(dynamics, x, ts))

  prior.sample ──> x(t_max)          grid ──> ts (t_max → t_min)
        │                                        │
        ▼                                        ▼
  loop:  x = integrator.step(dynamics, x, t, dt)         dt < 0
             │
             ├─ Euler/Heun:      dynamics.drift, dynamics.diffusion
             ├─ Ancestral:       s = dynamics.score
             │                   x0̂ = dynamics.sde.x0_from_score(x, s, t)
             │                   mean, std = dynamics.sde.posterior(x, x0̂, t, t+dt)
             └─ AncestralDDPM:   dynamics.g2, dynamics.score
```

The chart-free lane, entirely apart:

```
  DirectDenoisingSampler:  loop { inject noise; x = to_x0(process, model(x,0), x, 0) }
  — no grid, no reverse object, no chart; the GPFF path.
```

## The three load-bearing rules

1. **Reverse objects take bound fields, not (model, parametrization, cond).**
   `ReverseSDE` is pure chart math over a `score_fn(x, t)`; `ReverseODE` is
   pure transport over a `velocity_fn(x, t)`. The composition of head +
   conversion + conditioning happens once, in `Sampler.denoise` — or by
   hand, if you hold a ready field:
   `ReverseSDE(process.sde(), score_fn, churn=1.0)`.

2. **The chart is acquired where it is needed, and acquisition is the
   check.** One condition — `churn > 0 or velocity_needs_chart or
   integrator.requires_sde` — decides both the eager check in
   `Sampler.__init__` (fail at assembly, obstruction named) and the class
   picked in `denoise()`. Configurations without the kernel (shape prior,
   value-dependent coupling) keep the chart-free lanes: velocity head at
   churn = 0, and direct denoising.

3. **Integrators see only the dynamics.** The generic solvers consume
   `drift`/`diffusion` and work on either reverse class. The ancestral pair
   declares `requires_sde` and additionally reads `score` plus the chart's
   closed forms — the one sanctioned widening of the contract.

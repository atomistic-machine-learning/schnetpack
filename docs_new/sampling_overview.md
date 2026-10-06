# Sampling overview: what goes where

*A one-page map of the sampling-side objects and their connections, after
the split of `schnetpack.dynamics` into the sample and optimize families.
Detailed treatment: [sampling.md](sampling.md); math grounding:
[flow_matching_sde.md](flow_matching_sde.md).*

## The pieces, one line each

| Object | Home | Owns exactly | Knows nothing about |
|---|---|---|---|
| `Process` (VE, VP, FM, …) | `generative/processes.py` | the interpolant chart: schedules a/b, `sigma`, `perturb`, `sampling_prior`, the kernel check | models, sampling, step rules |
| `SDE` | `generative/differential_equations.py` | the (f, g) chart: `f`, `g2`, `kernel`, `posterior`, `x0_from_score`; constructing it **is** the Gaussian-kernel check | models, parametrizations |
| `ReverseSDE` | `generative/differential_equations.py` | Anderson family on the chart: `drift(x, t, score) = f x − ½(1+eta2) g² score`, `diffusion` | how the score was produced |
| `ReverseODE` | `generative/differential_equations.py` | transport `drift = v`, `diffusion = 0` | everything else — chart-free by design |
| `Parametrization` | `generative/parametrizations.py` | head contract: training `target`, conversions `to_score` / `to_velocity` / `to_x0` | step rules, samplers |
| `Prior` | `generative/priors.py` | the x1 endpoint law; the start state | everything else |
| `Calculator` | `dynamics/calculator.py` | inference: device/dtype (`prepare`, once per run), neighbor list (every call, on a copy), grad policy, the model call, the `guidance` terms | processes, steps |
| `GenerativeCalculator` | `dynamics/calculator.py` | the model with its process, parametrization, `key`, `output_key`, `time_key`; the guided `score` / `x0` / `velocity` at (x, t) | grids, step rules |
| `ForceCalculator` | `dynamics/calculator.py` | the guided force of a physical (eV/Å) or pseudo-force (Å) model, with the unit conversion | grids, step rules |
| `Guidance` (`HarmonicRestraint`) | `dynamics/guidance.py` | one weighted force-like term the calculator adds to the field it returns | drivers, the state between steps |
| `Dynamics` | `dynamics/base.py` | the calculator, the optional prior and its draw in `sample`, the moved `key`, the hooks run around each step (abstract `run`) | generative models, what one step is |
| `Hook` (`FreezeScaffold`) | `dynamics/hooks.py` | edits of the batch before and/or after a step | models, the field, step rules |
| `TimeGrid` | `dynamics/sample/grids.py` | where the steps land | everything else |
| `Sampler` | `dynamics/sample/sampler.py` | the time-indexed loop (a `Dynamics`): defaults the prior to the process's, acquires the chart and builds the `ReverseSDE` at construction, steps along the grid | numerics of one step (the subclass), field math (the calculator) |
| `EulerMaruyama`, `Heun`, `Ancestral` | `dynamics/sample/sampler.py` | one step rule each, on `reverse.drift` / `reverse.diffusion` and the calculator's score (`Ancestral`: the chart's closed forms) | what produced the score |
| `Optimizer` | `dynamics/optimize/optimizer.py` | the time-free loop (a `Dynamics`): the force, the `fmax` stop test, holding converged structures and fixed atoms | processes, grids |
| `DirectDenoising` | `dynamics/optimize/direct_denoising.py` | GPFF's jump-to-x0 step, x ← x + F/2 on a pseudo-force | the entire reverse machinery — deliberately outside it |

## The chart of the connections

```
CONFIGURATION (checked at construction)

  GenerativeCalculator(model, process, parametrization)
     │  parametrization.validate(process)
     ▼
  Sampler(calculator, grid, prior, eta2, hooks)
     │  process.sde()  — succeeds iff the Gaussian kernel holds;
     │                   refusal names the obstruction
     ▼
  reverse = ReverseSDE(sde, eta2)


FIELD (the calculator, every evaluation)

  calculator.score(batch, x, t)
     = to_score(process, model({**batch, R: x, t: t})[output_key], x, t)
       + Σ wᵢ Fᵢ                                    ← guidance, if any


INTEGRATION (Sampler.run: a for loop, one step per grid interval)

  prior.sample ──> x(t_max)          grid ──> ts (t_max → t_min)
        │                                        │
        ▼                                        ▼
  for i:  batch = before_step(batch, i, n)              batch[t] = ts[i]
         x = self.step(batch, x, t, dt)                  dt < 0
         batch = after_step(batch, i + 1, n)             batch[t] = ts[i+1]
             │
             ├─ EulerMaruyama/Heun:  reverse.drift(x, t, score), reverse.diffusion(t)
             └─ Ancestral:           x0̂ = sde.x0_from_score(x, score, t)
                                     mean, std = sde.posterior(x, x0̂, t, t+dt)
```

The chart-free lane, entirely apart:

```
  DirectDenoising (an Optimizer on ForceCalculator(model, kind="pseudo")):
      loop { stop if max |F| < fmax;  x = x + F(x)/2 }      F = 2 (x0̂ − x)
  — no grid, no reverse object, no chart; the GPFF path. Same Dynamics
    hooks, so the same hooks (FreezeScaffold) apply.
```

## The three load-bearing rules

1. **Reverse objects take the score, not (model, parametrization, batch).**
   `ReverseSDE` is pure chart math: `drift(x, t, score)`. The composition of
   head + conversion + batch + guidance happens in the calculator
   (`GenerativeCalculator.score`), once per evaluation.

2. **The chart is acquired where it is needed, and acquisition is the
   check.** The `Sampler` always steps on the score and acquires the chart
   in `__init__` (fail at assembly, obstruction named); a
   `GenerativeCalculator` with guidance acquires it too, since guidance
   reaches x0 and the velocity through it. Configurations without the
   kernel (shape prior, value-dependent coupling) keep the chart-free lane:
   direct denoising.

3. **Guidance belongs to the calculator, hooks to the driver.** Anything
   that changes the field — a restraint, classifier guidance — is a
   `Guidance` term on the calculator, so every evaluation of the field
   (both stages of Heun included) sees it. Anything that edits the batch
   between full steps is a `Hook` on the driver. A driver refuses guidance
   passed as a hook.

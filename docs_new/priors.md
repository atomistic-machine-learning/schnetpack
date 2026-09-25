# Axis 2 — Priors: the endpoint

*Module: `schnetpack.generative.priors` · classes `Prior`, `GaussianPrior`,
`DatasetPrior`, `DatasetStructures`, `StatisticsStructures`.*

A prior answers one question — **what $x_1$ is** — and it is asked twice:

- **Training.** The forward process needs an endpoint for every data sample:
  `Prior.sample_positions(batch)` draws $x_1$ positions for the batch being
  diffused, and the [coupling](couplings.md) then decides how the two are
  paired. Under the default `IdentityCoupling` this is the familiar "fresh
  noise per sample". Training only ever draws positions.
- **Sampling.** With $b(t_{\max}) = 1$, the state the reverse process starts
  from *is* the $x_1$ endpoint, so the correct start distribution is $x_1$'s
  marginal. When the coupling only re-pairs, that marginal is this prior
  itself, and `Process.sampling_prior()` hands the **very same object** to
  the `Sampler`. Sampling starts from a full batch, as the dataloader would
  give it — see [§4](#4-sampling-full-batches).

One object serving both sides is the point: train-time and sample-time $x_1$
cannot drift apart, because there is nothing to restate. This is the
[one-owner-per-fact principle](README.md#3-one-owner-per-fact) applied to
the endpoint's law.


## 1. The two declarations

A prior also declares what its draws *are*, and both declarations default to
the safe side:

**`Prior.gaussian`** (default `False`) — whether draws are independent
isotropic Gaussian. This gates the score/noise parametrizations and the
process's Gaussian-only closed forms, via
[`gaussian_kernel_obstruction`](processes.md#5-the-gaussian-kernel-a-property-not-a-class).
It defaults to `False` because the two failure directions are not
symmetric: a wrong `True` trains to garbage *silently* (the score target is
regressed against a meaningless quantity), while a wrong `False` merely
raises at assembly and asks you to opt in.

**`Prior.std`** (default `None`) — the endpoint's scale, or `None` when it
has no single scalar value (a shape prior with per-molecule covariance). The
process reads its noise level from this,

$$\sigma(t) = b(t)\cdot\texttt{prior.std},$$

so everything that needs $\sigma$ — score conversions, the SDE diffusion
$g^2$, churn $> 0$ sampling — needs a declared `std`, and fails with a
diagnosis (the `Process.std` property) rather than a `NoneType` error when
it is absent. The routes that never form $\sigma$ — churn 0 velocity
sampling, the pseudo-force/x0 recoveries — work without it.

### Scale ownership, concretely

For a `VE` process, `std` **is** $\sigma_{\max}$ — the knob that must match
the data scale (largest-pairwise-distance rule; see the
[VE footgun](processes.md#ve--variance-exploding-geometric-noise-score-matching--smld)).
The constructor sugar keeps the declaration single:

```python
VE(sigma_min=0.3, sigma_max=30.0)   # builds GaussianPrior(std=30.0) internally
VP(scale=2.5)                        # sugar for prior=GaussianPrior(std=2.5)
VE(b_min=1e-2, prior=my_shape_prior) # structured endpoint owns its own scale
```

Passing both `scale` and `prior` is a `TypeError` — the scale would then be
declared twice, which is exactly the mismatch the design exists to prevent.


## 2. What a draw may read from the batch — and what not

`sample_positions(batch)` receives the batch it draws for: atom types, atom
count, scaffold indices, the layout (`idx_m`) a draw must respect. Its
positions, when present, give the draw its shape, dtype and device; their
values are never read. The boundary is principled:

> **Structure is not values.** A prior may read *structure* out of the
> batch (which rows form a molecule, how many atoms). A distribution
> shaped by the **data values** themselves is not a prior but a
> [coupling](couplings.md) — see `PCVarianceCoupling`, which reshapes drawn
> noise against the actual $x_0$.

The distinction matters because the prior must be drawable at generation
time, when there is no $x_0$. Everything a prior reads from `context` is
available then too.


## 3. `GaussianPrior` and per-molecule centering

`GaussianPrior(std, centered=True)` is the endpoint of every plain diffusion
and flow-matching process: $\mathcal{N}(0, \mathrm{std}^2 I)$, **centered
per molecule** by default — each segment's mean is subtracted, putting $x_1$
in the same zero-center-of-geometry subspace that
`SubtractCenterOfGeometry` puts $x_0$ in.

Why centering is what molecules need: a translation-invariant network can
never predict a displacement of a whole structure, so an off-subspace
endpoint component is unlearnable noise in every training target, and an
offset nothing removes in every sampling start. Uncentered, a draw carries a
center of geometry of scale $\mathrm{std}\cdot\sqrt{d/n}$ per molecule —
4.6 Å for a 12-atom molecule at $\mathrm{std} = 10$, larger than the
molecule itself.

Two facts make this safe and one makes it load-bearing:

- **Centering does not cost the Gaussian kernel.** Projecting a standard
  normal onto a subspace gives a standard normal *on that subspace* with the
  same per-direction variance, so `gaussian` stays `True` and the
  score/noise parametrizations stay exact. Only the space changes.
- Which rows share a mean is read from the batch's `idx_m` (a batch
  without it is one group), so centering is per molecule, not per batch.
  `Diffuse` and the sampling entries all hand the prior the batch with its
  layout.
- **$x_0$ must be centered too** — compose `SubtractCenterOfGeometry`
  before [`Diffuse`](training.md). Centered noise on uncentered data leaves
  $x_t$'s mean drifting with $a(t)$, and the kernel is no longer the one the
  targets assume.

Set `centered=False` for data with no translation symmetry to quotient out,
or when the leading axis is independent samples rather than the atoms of one
structure — centering couples the rows it spans.


## 4. Sampling full batches

A sampling start is a full batch — atom types, positions, `n_atoms`,
`idx_m`, as the dataloader would give it. Three entries, one per use:

| Entry | Returns | Used by |
|---|---|---|
| `sample_positions(batch)` | positions tensor | training (`Process.perturb`), `Scaffold` re-noising |
| `sample_from_batch(batch)` | the batch with its positions redrawn | sampling from a test set |
| `sample(n_samples)` | `n_samples` structures from `prior.structures`, positions redrawn | `Dynamics.sample` |

`structures` is the prior's optional source of structures, any object with
`sample(n_samples) -> batch`:

- `DatasetStructures(dataset, shuffle=True)` — structures from a dataset,
  collated as the dataloader would. Successive calls walk the dataset without
  repeating a structure until a pass is complete, reshuffled per pass.
- `StatisticsStructures(n_atoms, atom_types)` — compositions from scratch:
  each structure draws its atom count from the `n_atoms` histogram (entry
  $k$ weighs $k$ atoms), each atom its type from the `atom_types` weights
  (entry $z$ weighs atomic number $z$). `StatisticsStructures.from_dataset`
  counts both over a dataset. Atom types are drawn independently, so a
  composition need not occur in the dataset.

```python
stats = StatisticsStructures.from_dataset(train)
process = VP(prior=GaussianPrior(structures=stats))   # training ignores structures
samples = Sampler(calc, process, param, Heun()).sample(64, n_steps=50)

for batch in test_loader:                              # or: from a test set
    out = sampler.denoise(sampler.prior.sample_from_batch(batch), n_steps=50)
```

**`DatasetPrior(dataset, shuffle=True)`** returns stored structures
unchanged — non-equilibrium structures to relax, say — so it is a sampling
start only: `DirectDenoising(calc, process, param, prior=DatasetPrior(noneq))
.sample(n, n_steps)`. It has no positions law, and `sample_positions` raises
rather than silently training on $x_1 = x_0$.


## 5. Writing a custom prior

```python
class ConformerPrior(Prior):
    """Endpoints drawn from a physically-motivated conformer ensemble."""

    gaussian = False       # be honest: these draws are not isotropic Gaussian
    std = None             # and they have no single scalar scale

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def sample_positions(self, batch):
        return self.ensemble.draw(batch[properties.R].shape)
```

Guidelines:

- Define the law **once**, in `sample_positions`; training,
  `sample_from_batch` and `sample(n_samples)` all go through it.
- A prior that draws whole structures rather than positions overrides
  `sample` / `sample_from_batch` instead, as `DatasetPrior` does.
- Declare `gaussian` and `std` **honestly**. The declarations are trusted:
  a false `gaussian = True` makes score/eps training silently meaningless;
  a false `std` corrupts every $\sigma$-consuming conversion. When in doubt,
  leave the conservative defaults — the machinery will tell you exactly
  which routes that closes (velocity/x0/pseudo-force training and churn 0
  sampling all remain open; see
  [flow_matching_sde.md §8](flow_matching_sde.md)).
- A structured prior usually pairs with `Sampler.denoise` (starting below
  $t_{\max}$ from a structured state) or with `DirectDenoising` —
  see [sampling.md](sampling.md).


## 6. Why this design

**Why a prior class at all, rather than `torch.randn` in the process?**
Because the endpoint is asked for twice — training and sampling — and the
two call sites live in different objects. Making the endpoint a value-object
the process holds means `sampling_prior()` can *derive* the sampling start
instead of asking the user to restate it, eliminating the classic
train/sample mismatch bug by construction.

**Why the scale lives here.** $\sigma_{\max}$-type knobs describe the
endpoint's law, not the schedule's geometry; putting them on the prior keeps
the schedule dimensionless and reusable, and makes "declared once" literal —
`VE(sigma_min, sigma_max)` mentions the scale in exactly one place, and
every consumer reads it back through `process.std`.

**Why declarations instead of introspection.** Whether draws are Gaussian is
a semantic fact no amount of tensor inspection can establish; the design
makes it an explicit, safe-defaulted contract and then *judges* everything
downstream from it
([configuration, not class](README.md#2-properties-are-judged-from-the-configuration-not-the-class)).

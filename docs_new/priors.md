# Axis 2 — Priors: the endpoint

*Module: `schnetpack.generative.priors` · classes `Prior`, `GaussianPrior`.*

A prior answers one question — **what $x_1$ is** — and it is asked twice:

- **Training.** The forward process needs an endpoint for every data sample:
  `Prior.sample_like(x0, context)` draws $x_1$ shaped like $x_0$, and the
  [coupling](couplings.md) then decides how the two batches are paired.
  Under the default `IdentityCoupling` this is the familiar "fresh noise per
  sample".
- **Sampling.** With $b(t_{\max}) = 1$, the state the reverse process starts
  from *is* the $x_1$ endpoint, so the correct start distribution is $x_1$'s
  marginal. When the coupling only re-pairs, that marginal is this prior
  itself, and `Process.sampling_prior()` hands the **very same object** to
  the `Sampler`.

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


## 2. What `context` is — and is not

`sample` and `sample_like` accept a `context`: generation-time conditioning
the endpoint may legitimately depend on — composition, atom count, scaffold
indices, the batch layout a draw must respect. The boundary is principled:

> **Structure is not values.** A prior may read *structure* out of the
> context (which rows form a molecule, how many atoms). A distribution
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
- Which rows share a mean comes from the `context` (a batch dict holding
  `idx_m`, a raw segment-id tensor, or `None` for one group), so centering
  is per molecule, not per batch. `Diffuse` and `Sampler.sample` both pass
  the layout through on their own.
- **$x_0$ must be centered too** — compose `SubtractCenterOfGeometry`
  before [`Diffuse`](training.md). Centered noise on uncentered data leaves
  $x_t$'s mean drifting with $a(t)$, and the kernel is no longer the one the
  targets assume.

Set `centered=False` for data with no translation symmetry to quotient out,
or when the leading axis is independent samples rather than the atoms of one
structure — centering couples the rows it spans.


## 4. Writing a custom prior

```python
class ConformerPrior(Prior):
    """Endpoints drawn from a physically-motivated conformer ensemble."""

    gaussian = False       # be honest: these draws are not isotropic Gaussian
    std = None             # and they have no single scalar scale

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def sample(self, shape, dtype=None, device=None, context=None):
        return self.ensemble.draw(shape, dtype=dtype, device=device)
```

Guidelines:

- Define the law **once**, in `sample`; `sample_like` defaults to calling it
  with $x_0$'s shape/dtype/device. Override `sample_like` only when the
  training draw needs more than the shape (e.g. a scaffold prior copying
  fixed atoms out of the context).
- Declare `gaussian` and `std` **honestly**. The declarations are trusted:
  a false `gaussian = True` makes score/eps training silently meaningless;
  a false `std` corrupts every $\sigma$-consuming conversion. When in doubt,
  leave the conservative defaults — the machinery will tell you exactly
  which routes that closes (velocity/x0/pseudo-force training and churn 0
  sampling all remain open; see
  [flow_matching_sde.md §8](flow_matching_sde.md)).
- A structured prior usually pairs with `Sampler.denoise` (starting below
  $t_{\max}$ from a structured state) or with `DirectDenoisingSampler` —
  see [sampling.md](sampling.md).


## 5. Why this design

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

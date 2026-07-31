# Axis 3 — Couplings: the pairing

*Module: `schnetpack.generative.couplings` · classes `Coupling`,
`IdentityCoupling`, `PermutationCoupling`, `PCVarianceCoupling`,
`OTCoupling`.*

A coupling answers the third independent question about the forward process:
given a data batch and a batch of endpoint draws, **how are the two
paired?** Formally it chooses the joint law $\pi(x_0, x_1)$ whose marginals
the [prior](priors.md) and the data have already fixed.

The interface is deliberately narrow: a coupling **re-pairs batches that are
already drawn, and never draws**. Drawing is the prior's job; keeping the
two roles apart is what lets `sampling_prior()` reason about the sampling
start (below) and keeps each axis testable alone.

```python
x0, x1 = coupling.pair(x0, x1, groups)   # returns both, re-paired
```


## 1. Why the pairing is an axis at all

For the classic methods it is invisible: VE, VP and plain flow matching use
the **product coupling** $\pi = p_0 \times p_1$ — every sample keeps the
fresh, independent draw the prior handed it (`IdentityCoupling`). But the
pairing is a genuine degree of freedom, and the literature uses it:

- **Straightening transport.** The flow-matching training paths run
  $x_0 \to x_1$ in straight lines. Under independent pairing those lines
  criss-cross, and the marginal velocity field the model learns is curved
  where they do. Re-pairing each $x_0$ with a *nearby* $x_1$ (minibatch
  optimal transport — rectified flow, OT-CFM) shortens and de-crosses the
  paths, which straightens the learned field and cuts the number of sampling
  steps needed.
- **Shaping the endpoint against the data.** GPFF-style structured noising
  can go further and reshape the drawn endpoints from the data's values
  (e.g. matching the variance ellipsoid — `PCVarianceCoupling`), trading
  transport distance for a changed endpoint law.

Both are one knob on the same object; nothing else in the stack changes.


## 2. The one declared property: `preserves_marginal`

`Coupling.preserves_marginal` (default `False`) states whether `pair` leaves
$x_1$'s **marginal law** untouched. It is `True` for pure re-orderings
(identity, permutation, OT) and `False` for anything that reshapes values
from the data. Two consumers read it:

- **The sampling start.** With $b(t_{\max}) = 1$, sampling must start from
  $x_1$'s marginal. If the coupling only re-pairs, that marginal is the
  training prior itself, and `Process.sampling_prior()` returns the very
  same object. A marginal-*changing* coupling has no data-free start
  distribution, and the `Sampler` demands an explicit `Prior` matching the
  statistics trained under — refusing to guess.
- **The Gaussian-kernel judgment**
  ([`gaussian_kernel_obstruction`](processes.md#5-the-gaussian-kernel-a-property-not-a-class)),
  which gates the score/noise parametrizations and the closed forms.

The default is `False` for the usual asymmetry: a wrong `True` silently
samples from the wrong start; a wrong `False` merely asks for an explicit
prior.

> **A subtlety worth internalizing: marginal ≠ conditional.** Re-pairing
> preserves $x_1$'s *marginal* — which is exactly what the sampling start
> needs — but any pairing chosen by looking at the data changes the
> *conditional* law $p(x_1 \mid x_0)$. The one-sided kernel
> $p(x_t \mid x_0) = \mathcal{N}(a x_0, \sigma^2 I)$ behind the score/noise
> targets is a statement about that conditional: it needs $x_1$ independent
> of $x_0$, which only the identity coupling gives exactly. With a
> data-dependent re-pairing, prefer the targets that are plain conditional
> expectations — velocity, $x_0$, pseudo-force — which are valid under
> *any* joint law (they regress $\mathbb{E}[\,\cdot \mid x_t]$ for whatever
> path law the coupling induces). This is also the standard practice in the
> OT-flow-matching literature, which trains velocities.


## 3. The `groups` mechanism: interchangeable rows

The tensor core sees one anonymous sample axis. For a collated batch of
molecules, that axis is *atoms*, and an unrestricted re-pairing would trade
endpoints across molecules and across elements — meaningless for the
physics. `pair` therefore accepts an optional `groups` tensor labelling the
rows; re-pairing stays **within** rows that share all labels:

```python
# label by (molecule index, atomic number): an atom trades endpoints only
# with atoms of its own element in its own molecule
process.perturb(x0, groups=torch.stack([idx_m, Z], dim=-1))
```

`row_blocks` partitions the axis into blocks of interchangeable rows;
couplings solve per block. This is cheaper as well as more correct: for the
assignment solve, a sum of small cubes beats the cube of the sum. The
atomistic edge ([`Diffuse`](training.md)) reads the labels off the batch
(`group_keys`, default `(idx_m, Z)`) and passes them down — the core itself
never learns what an atom is.

`perturb` only forwards `groups` when given one, so couplings written
against the two-argument `pair` keep working.


## 4. The catalog

### `IdentityCoupling` — the product coupling

Leaves the pairing exactly as drawn: $\pi = p_0 \times p_1$. What VE, VP and
plain flow matching use, and the reason their score/noise targets are valid:
$x_1$ stays the independent noise realization the kernel math assumes.
`preserves_marginal = True` trivially.

### `PermutationCoupling` — single-batch optimal assignment

Reorders the drawn endpoints by solving the linear assignment problem
against the data under a `cost_power` distance cost (2.0 = the squared-cost
OT special case where the plan is a permutation). Each $x_0$ row keeps
exactly one $x_1$ partner; with `groups`, one solve per block. Marginal
preserved (a re-ordering); the joint with $x_0$ — and hence the path
geometry — is what changes. Needs SciPy
(`scipy.optimize.linear_sum_assignment`).

### `PCVarianceCoupling` — match the data's variance ellipsoid

Computes the principal axes of the drawn $x_1$ cloud and rescales each
principal component so its variance matches the data's corresponding one
(both sorted descending): the prior keeps its own random orientation but
takes on the data's *shape* — a long molecule is met by an elongated noise
cloud, making the transport closer to a rotation than a stretch.

This one **changes the marginal** (`preserves_marginal = False`), with both
enforced consequences above: the process loses its Gaussian kernel (use a
velocity/x0/pseudo-force head — the score/noise ones refuse at `validate`),
and sampling needs an explicit `Prior` with matching statistics. It
currently refuses `groups` — a variance rescale needs per-molecule blocks,
not the per-(molecule, element) blocks `groups` carries, and refusing beats
silently fitting ellipsoids to the wrong point sets. Use it per structure.

### `OTCoupling` — minibatch optimal transport (planned)

The general OT plan (POT-based `emd`, Sinkhorn fallback); lands with the OT
milestone. Until then, `PermutationCoupling` covers the
permutation-constrained case and `IdentityCoupling` standard flow matching.


## 5. Writing a custom coupling

```python
class ReflectCoupling(Coupling):
    """Flip each endpoint into the data point's half-space (toy example)."""

    preserves_marginal = False   # sign depends on the data values

    def pair(self, x0, x1, groups=None):
        sign = torch.sign((x0 * x1).flatten(1).sum(-1))
        return x0, x1 * sign.reshape(-1, *([1] * (x1.dim() - 1)))
```

Guidelines:

- **Never draw.** If your "coupling" needs randomness of its own, the
  randomness belongs in a prior; a coupling only rearranges or reshapes what
  it is given.
- Declare `preserves_marginal` honestly, and only for genuine
  re-orderings. If values change as a function of the data, it is `False`.
- Accept `groups` (even if only to raise, as `PCVarianceCoupling` does):
  refusing a granularity you cannot honor beats silently using the wrong
  one.
- Remember the conditional-law subtlety from §2 when choosing which
  parametrizations to recommend alongside your coupling.


## 6. Why this design

**Why couplings never draw.** If the pairing could also draw, "what is
$x_1$?" would have two owners, and `sampling_prior()` could no longer derive
the sampling start from the prior alone. The narrow contract — rearrange or
reshape, never create — is what makes the marginal-preservation question
even *askable*, and with it the whole derived-start machinery.

**Why re-pairing is separate from the prior.** The same shape prior can run
with identity pairing or OT pairing; the same OT pairing can run over a
Gaussian or a structured prior. Two independent choices, two axes — folding
them together would square the class count and hide which choice caused an
observed effect.

**Why a declared flag rather than analysis.** Whether a re-pairing preserves
the marginal is a semantic property of the algorithm, not something the
framework can verify from tensors; the design makes it an explicit contract
with a safe default, in line with the
[assembly-time-validity principle](README.md#4-validity-settles-at-assembly).

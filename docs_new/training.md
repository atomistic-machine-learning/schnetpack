# Training: `MatchingLoss` and `Diffuse`

*Modules: `schnetpack.generative.losses`, `schnetpack.generative.transforms`.*

Every objective in this subpackage — score matching, flow matching, bridge
matching — is the same three moves:

1. draw an endpoint pair and place it on the path at a random time
   (`Process.perturb`),
2. build the [parametrization's target](parametrizations.md) from the pair,
3. regress the model's output against it, with an optional per-time weight.

What distinguishes score matching from flow matching is **which process and
parametrization you hand over — not which loss you call.** There is one
objective, packaged twice for the two ways SchNetPack users train.


## 1. `MatchingLoss` — the tensor-level route

The whole objective in one call, for training loops that drive raw tensors:

$$
\mathcal{L} = \operatorname{mean}\Big(w(t)\,\big\lVert
\mathrm{model}(x_t, t, \mathrm{cond}) - \mathrm{target}(x_0, x_1, t)
\big\rVert^2\Big),
\qquad x_t \sim \texttt{process.perturb}.
$$

```python
loss_fn = MatchingLoss(
    process, parametrization,
    weight=None,      # w(t): per-sample loss weight, default uniform
    t_sampler=None,   # draws training times, default process.sample_t
)
loss = loss_fn(model, x0, x1=None, cond=None, context=None)
```

The pairing is validated at construction — by the time a `MatchingLoss`
exists, the assembly is coherent
([validity at assembly](README.md#4-validity-settles-at-assembly)). An
explicit `x1=` substitutes for the prior draw (still passed through the
coupling); `context` is handed to the prior, exactly as at sampling time.

### The two hooks

**`weight`** — the per-time loss weight $w(t)$. The shipped heads that need
one document it ([parametrizations.md](parametrizations.md)):

| configuration | weight | effect |
| --- | --- | --- |
| score head on `VE` | `lambda t: process.b(t)**2` | equalizes the target's dynamic range; objective becomes eps-matching up to the endpoint scale |
| pseudo-force head on `VE` | `lambda t: (1/process.b(t)**2).clamp(max=1.0)` | undoes the $b$-scaling, clamped so nearly-clean samples don't dominate |
| eps head | none needed | the target is unit variance by construction |

**`t_sampler`** — the training-time density $p(t)$, mapping
`(n, device) -> (n,)`. The default is the process's own `sample_t`, uniform
on $[t_{\min}, t_{\max}]$ — stopping short of $t = 0$ because the *score*
target diverges there. The noise and denoiser targets are well behaved at
0, so for those you may widen the range back; the EDM/GPFF
log-normal-$\sigma$ density enters through this same hook. Note the hook
shape is shared with `Diffuse`, so one sampler can serve both routes.


## 2. `Diffuse` — the data-pipeline route

SchNetPack trains through Lightning tasks whose losses are ordinary
supervised `ModelOutput`s, and its preprocessing runs in dataloader
workers. `Diffuse` is **`MatchingLoss` minus the model call and the MSE**,
repackaged as a `Transform`: it runs the noising inside the dataloader —
parallel, per structure, off the training thread — and writes everything a
supervised loss needs into the batch dict.

```python
Diffuse(
    process, parametrization,
    t_sampler=None,
    diffuse_property=properties.R,   # overwritten with x_t
    label_key="label",               # the parametrization's target
    time_key="t",                    # per-element time, for conditioning
    structure_time_key="t_structure",# per-structure time
    original_key=None,               # optionally keep the clean property
    group_keys=(properties.idx_m, properties.Z),
)
```

Running **per structure, before collation**, exactly one time is drawn per
structure and broadcast along the property's leading axis. For positions
that axis is atoms — which is what lets the tensor-level core apply
unchanged: it only ever asked for a per-sample time, and here the samples
are atoms.

Details that earn their keep:

- **The time is written at two granularities on purpose.** Collation
  concatenates along the leading axis, so `(n_atoms,)` arrives per-atom and
  `(1,)` per-structure. A head predicting one value per structure regressed
  against a per-atom target would not fail — MSE broadcasts
  `(n_structures,)` against `(n_atoms,)` and silently optimizes the wrong
  thing. Two keys make the footgun unloadable.
- **`group_keys` is where atoms become known.** The tensor core sees one
  anonymous sample axis; only this transform knows those rows are atoms,
  which molecule each belongs to and which element it is. It stacks the
  labels and hands them to `perturb`, so a
  [re-pairing coupling](couplings.md#3-the-groups-mechanism-interchangeable-rows)
  keeps its permutation inside one molecule and one element.
- **The batch dict is passed to the prior as `context`**, so a centered
  Gaussian endpoint reads `idx_m` from it — the same mechanism as at
  sampling time.

### Two things `Diffuse` deliberately does *not* do

- **It does not center the structure.** Compose
  `SubtractCenterOfGeometry` before it if your process lives in the
  zero-center subspace — and it must, when the prior centers its draws
  ([why](priors.md#3-gaussianprior-and-per-molecule-centering)).
- **It does not decide how the noise is drawn.** That is the prior's job.
  For molecules the noise must live in the same zero-center subspace as the
  data; express that as a `Prior` on the process, not by editing the
  transform.

And one ordering rule: put any **neighbor list transform after** `Diffuse`,
or it will be built on the clean structure and be wrong for $x_t$.


## 3. Which route, when

| you are… | use |
| --- | --- |
| driving raw tensors (toy problems, notebooks, custom loops) | `MatchingLoss` — the whole objective in one call |
| training through the SchNetPack datamodule/task pipeline | `Diffuse` in the transform list + a plain supervised `ModelOutput` on `label_key` |

The two overlap because SchNetPack splits what `MatchingLoss` fuses:
noising belongs in a dataloader worker, the loss in the task. Use the same
`(process, parametrization)` objects in either route **and** in the
[`Sampler`](sampling.md) — the
[caller's one obligation](README.md#the-one-obligation-this-leaves-the-caller).

A note for `Diffuse` users: targets are built inside dataloader workers, so
they must be picklable — which is why the schedule-derivative machinery
returns plain tensors
([processes.md §2](processes.md#derivatives)). A schedule with learnable
parameters needs analytic derivative overrides and the `MatchingLoss`
route.


## 4. End to end

The full circle, tensor level:

```python
process = VE(sigma_min=0.3, sigma_max=30.0)      # scale matches the data!
param   = EpsParametrization()

loss_fn = MatchingLoss(process, param)
for x0 in data:
    optimizer.zero_grad()
    loss_fn(model, x0).backward()
    optimizer.step()

sampler = Sampler(process, param, Ancestral())    # same objects
samples = sampler.sample(model, shape=x0.shape, n_steps=200)
```

And the pipeline variant of the training half:

```python
transforms = [
    SubtractCenterOfGeometry(),
    Diffuse(process, param),          # writes x_t, "label", "t"
    MatScipyNeighborList(cutoff=...), # AFTER Diffuse: neighbors of x_t
]
# task side: ModelOutput(name="label", loss_fn=MSELoss(), ...)
```

# Flow matching in the $(f, g)$ SDE framework

How the forward and reverse process of flow matching are represented in the
drift/diffusion language of score-based diffusion — the representation
`schnetpack.generative` uses to run every schedule, flow matching included,
through one `ReverseProcess`.

The punchline first: flow matching and Gaussian diffusion are two *charts* on
the same object. The interpolant $x_t = a(t)\,x_0 + b(t)\,x_1$ is one chart;
the linear SDE $\mathrm{d}x = f(t)\,x\,\mathrm{d}t + g(t)\,\mathrm{d}w$ is the
other. For a Gaussian endpoint they describe identical marginals, and $(f, g)$
is *derived* from $(a, b)$ by two lines of moment matching — it is not extra
structure that diffusion has and flow matching lacks. What flow matching lacks
is only a *need* for the second chart in its vanilla sampler; the chart is
there, and using it is what buys stochastic sampling, ancestral steps and
score access for a flow-matching model.

Throughout: data $x_0 \sim p_{\text{data}}$ sits at $t = 0$; the endpoint
$x_1 \sim \mathcal{N}(0, \sigma_1^2 I)$ is drawn independently of $x_0$ and
sits at $t = 1$. The endpoint scale $\sigma_1$ is `prior.std` in the code
(default 1); the noise level of the process is
$\sigma(t) = b(t)\,\sigma_1$ (`Process.sigma`). The score of the marginal
$p_t$ is $s(x, t) = \nabla_x \log p_t(x)$.


## 1. One process, two descriptions

**The interpolant chart** is what flow matching writes down
([Lipman et al., arXiv:2210.02747], [Liu et al., arXiv:2209.03003]):

$$
x_t = a(t)\,x_0 + b(t)\,x_1,
\qquad a(t) = 1 - t,\quad b(t) = t .
$$

A sample at any $t$ costs one draw and one multiply-add — this is what makes
training simulation-free, and it is why the interpolant is the primitive in
`processes.py` rather than the SDE.

**The SDE chart** is what score-based diffusion writes down
([Song et al., arXiv:2011.13456]):

$$
\mathrm{d}x = f(t)\,x\,\mathrm{d}t + g(t)\,\mathrm{d}w .
$$

The claim of this document: for the flow-matching schedule there is exactly one
$(f, g)$ pair whose SDE reproduces the interpolant's conditional marginals,
namely

$$
f(t) = -\frac{1}{1-t},
\qquad
g^2(t) = \frac{2t\,\sigma_1^2}{1-t},
$$

and everything diffusion knows how to do — reverse SDEs, probability-flow
ODEs, exact posteriors — applies to flow matching through it.


## 2. From $(a, b)$ to $(f, g)$: matching the conditional moments

Condition on a fixed $x_0$. Under the interpolant, with $x_1$ Gaussian and
independent of $x_0$, the conditional law is the Gaussian kernel

$$
p(x_t \mid x_0) = \mathcal{N}\!\big(a(t)\,x_0,\; \sigma(t)^2 I\big),
\qquad \sigma(t) = b(t)\,\sigma_1 .
$$

Under the linear SDE, the conditional law is also Gaussian, with mean $m(t)$
and isotropic variance $V(t)$ obeying the moment ODEs

$$
\dot m = f\,m, \qquad \dot V = 2 f V + g^2 ,
$$

with $m(0) = x_0$, $V(0) = 0$ (the flow-matching schedule has $a(0) = 1$,
$b(0) = 0$). Matching the two charts is solving these for $f$ and $g$:

- $m(t) = a(t)\,x_0$ for **every** $x_0$ forces

  $$
  f(t) = \frac{\dot a(t)}{a(t)} = \frac{\mathrm{d}}{\mathrm{d}t}\log a(t).
  $$

- $V(t) = \sigma(t)^2$ then forces

  $$
  g^2(t) = \frac{\mathrm{d}}{\mathrm{d}t}\sigma^2(t) - 2 f(t)\,\sigma^2(t).
  $$

Two remarks before specializing.

**The log-SNR form.** With $\operatorname{SNR} = a^2/b^2$ (the squared
convention of `Process.log_snr`), $\log \operatorname{SNR} = 2\log a - 2\log b$
gives $(\log \operatorname{SNR})' = 2f - (\sigma^2)'/\sigma^2$, so the two
$g^2$ formulas are the same statement:

$$
g^2 = (\sigma^2)' - 2 f \sigma^2
    = -\,\sigma^2\,\frac{\mathrm{d}}{\mathrm{d}t}\log \operatorname{SNR}.
$$

The code computes the right-hand form (`Process.g2`): one log-derivative of
one schedule quantity, no quotient that degenerates where $a$ or $b$ vanish,
and the sign is readable — $g^2 \ge 0$ exactly because signal only ever turns
into noise.

**What was used.** The derivation needed $x_1$ Gaussian, of declared scale,
independent of $x_0$. Those are precisely the conditions
`Process.gaussian_kernel_obstruction` checks; §8 returns to what survives when
they fail. Note that a coupling which re-pairs endpoints *within a batch*
changes the conditional law $p(x_1 \mid x_0)$ even when it preserves $x_1$'s
marginal, so the one-sided kernel — and with it this whole chart — has to be
re-examined per coupling, not assumed from the marginal alone.


## 3. The flow-matching schedule, specialized

| quantity | general | flow matching ($a = 1-t$, $b = t$) |
| --- | --- | --- |
| data coefficient | $a(t)$ | $1 - t$ |
| noise coefficient | $b(t)$ | $t$ |
| noise level | $\sigma(t) = b\,\sigma_1$ | $t\,\sigma_1$ |
| drift | $f = \dot a / a$ | $-\dfrac{1}{1-t}$ |
| log-SNR | $2(\log a - \log b)$ | $2 \log\dfrac{1-t}{t}$ |
| squared diffusion | $-\sigma^2 (\log \operatorname{SNR})'$ | $\dfrac{2t\,\sigma_1^2}{1-t}$ |

(For $g^2$: $(\log\operatorname{SNR})' = -\tfrac{2}{t(1-t)}$, and
$-t^2\sigma_1^2 \cdot \big({-\tfrac{2}{t(1-t)}}\big) = \tfrac{2t\sigma_1^2}{1-t}$.
Equivalently $(\sigma^2)' - 2f\sigma^2 = 2t\sigma_1^2 + \tfrac{2t^2\sigma_1^2}{1-t}$,
the same thing.)

So the **forward SDE of flow matching** is

$$
\mathrm{d}x = -\frac{x}{1-t}\,\mathrm{d}t
            + \sigma_1\sqrt{\frac{2t}{1-t}}\;\mathrm{d}w .
$$

Sanity check by solving the moment ODEs: the mean contracts as
$\exp\!\int_0^t f = \exp\log(1-t) = 1-t$, and $V(t) = t^2\sigma_1^2$ satisfies
$\dot V = 2fV + g^2$ since
$-\tfrac{2t^2\sigma_1^2}{1-t} + \tfrac{2t\sigma_1^2}{1-t} = 2t\sigma_1^2$. Any
initial point is transported to exactly $\mathcal{N}(0, \sigma_1^2 I)$ at
$t = 1$.

### The $t \to 1$ singularity is the price of a finite-time prior

Both $f$ and $g^2$ diverge as $t \to 1$, and this is not an accident of the
schedule but a theorem about the chart: a linear SDE whose conditional mean
reaches **exactly zero at finite time** must have
$\int_0^1 f\,\mathrm{d}t = \log a(1) - \log a(0) = -\infty$. Flow matching
forgets the data completely, in finite time, so its drift cannot stay bounded.
VP diffusion makes the opposite trade — $a(t) > 0$ for all finite $t$, bounded
coefficients, but the prior is only reached asymptotically and $p_{t_{\max}}$
is an approximation.

Crucially the singularity lives in the *chart*, not the object: the
interpolant is perfectly regular at $t = 1$ ($a = 0$, $b = 1$, $x_1$ exact).
This is why `FlowMatching` defaults to $t_{\max} = 1 - 10^{-3}$ — the guard is
for consumers of $f$ and $g^2$ — while pure-ODE use (churn $= 0$, where $g^2$
is never evaluated; see §6) may pass `t_max=1.0` and start from the exact
endpoint.

### The $t \to 0$ side is a different kind of singular

The forward SDE is regular at $t = 0$: $f(0) = -1$, $g^2(0) = 0$. What
diverges there is the **score**, $s \sim -(x - a x_0)/\sigma^2$ with
$\sigma \to 0$ — a property of the marginals themselves (they collapse onto
the data), shared by every process with $\sigma(0) = 0$ and not curable by a
change of chart. That is the singularity behind `t_min`, behind the divergence
of the score *training target* as $b \to 0$, and behind the degeneracy of the
velocity-to-score conversion (§5).


## 4. Same marginals, different paths

One honesty clause. The interpolant and the SDE agree on the marginal law of
$x_t$ at every $t$ (conditionally on $x_0$, and hence also unconditionally),
but **not** on the joint law across times: given $(x_0, x_1)$ the interpolant's
path is a straight line, while the SDE's path wiggles. They are different
processes with the same time-marginals.

This is not a defect — it is the entire mechanism. Training only ever asks for
$x_t$ at one time (the interpolant's specialty), and reverse-time sampling
only needs the marginals plus their score/velocity fields (which is why any
sampler from §6, stochastic or not, may be paired with a model trained on
straight-line draws). Nothing in either direction ever needs the two path
measures to agree.


## 5. The score–velocity identity

Flow matching trains the **velocity**. Differentiating the interpolant along a
pair gives the target

$$
\dot x_t = \dot a\,x_0 + \dot b\,x_1 = x_1 - x_0
\qquad \text{(constant in } t \text{ — the straightness of FM)},
$$

and the regression converges to the marginal velocity field
$v(x, t) = \mathbb{E}[\,x_1 - x_0 \mid x_t = x\,]$.

The bridge to the score chart is Tweedie's formula. For the Gaussian kernel,
$\mathbb{E}[x_0 \mid x_t] = (x_t + \sigma^2 s)/a$, and substituting
$x_1 = (x_t - a x_0)/b$ into the velocity:

$$
v = \dot a\,\mathbb{E}[x_0 \mid x_t] + \frac{\dot b}{b}\big(x_t - a\,\mathbb{E}[x_0 \mid x_t]\big)
  = f x_t + \sigma^2 \Big(f - \frac{\dot b}{b}\Big)\, s
$$

and since $f - \dot b/b = \tfrac{1}{2}(\log\operatorname{SNR})'$ while
$g^2 = -\sigma^2 (\log\operatorname{SNR})'$:

$$
\boxed{\;v(x, t) = f(t)\,x - \tfrac{1}{2}\,g^2(t)\,s(x, t)\;}
$$

— the probability-flow identity, load-bearing across
`parametrizations.py`. The right-hand side is exactly the probability-flow ODE
drift of the forward SDE, so "the FM velocity field" and "the PF-ODE of the
equivalent diffusion" are the same object, with no separate code path.

*Spot check* (point-mass data $p_{\text{data}} = \delta_{x_0}$,
$\sigma_1 = 1$): $p_t = \mathcal{N}((1-t)x_0, t^2)$, so
$s = -(x - (1-t)x_0)/t^2$, and
$f x - \tfrac12 g^2 s = -\tfrac{x}{1-t} + \tfrac{t}{1-t}\cdot\tfrac{x-(1-t)x_0}{t^2}
= \tfrac{x - x_0}{t}$ — which is indeed
$\mathbb{E}[x_1 - x_0 \mid x_t = x] = \tfrac{x - (1-t)x_0}{t} - x_0$.

The identity inverts to

$$
s(x, t) = \frac{2\big(f(t)\,x - v(x, t)\big)}{g^2(t)},
$$

which is how a velocity-predicting model yields a score
(`VelocityParametrization.to_score`). Note the division: $g^2 \to 0$ as
$t \to 0$, so the inversion degenerates exactly where §3 said the score
genuinely explodes. The design consequence is in §6: the churn $= 0$ path is
written so this inversion never runs.


## 6. The reverse process: one family, one knob

Reversing the forward SDE ([Anderson 1982]) gives the reverse-time SDE with
drift $f x - g^2 s$; the probability-flow ODE shares its marginals with drift
$f x - \tfrac12 g^2 s = v$. `ReverseProcess` interpolates between them as a
one-parameter family in the churn $\chi = \eta^2 \in [0, 1]$:

$$
\mathrm{d}x = \Big[\, v(x, t) - \tfrac{1}{2}\,\chi\, g^2(t)\, s(x, t) \,\Big]\mathrm{d}t
            \;+\; \sqrt{\chi}\; g(t)\,\mathrm{d}\bar w ,
$$

integrated from $t_{\max}$ down to $t_{\min}$ (the code passes
$\mathrm{d}t < 0$). Every member shares the forward marginals: the
$\chi$-dependent drift piece contributes
$\tfrac12 \chi g^2\, \nabla\!\cdot(p\, \nabla \log p) = \tfrac12 \chi g^2 \Delta p$
to the Fokker–Planck equation, which is exactly what the
$\chi$-dependent noise term removes — the two cancel for every $\chi$, so the
knob moves the path measure, never the marginals.

**Churn $= 0$ is vanilla flow matching.** The drift is $v$ itself — for a
velocity-predicting model, the raw output, used directly. Neither $g^2$ nor
the score inversion is ever evaluated (`ReverseProcess.drift` returns early),
which is why the ODE path tolerates $t_{\max} = 1$ and priors with no scalar
scale, and why flow matching "costs nothing it shouldn't" in this framework.

**Churn $= 1$ is the stochastic flow-matching sampler.** Explicitly, with
$\tfrac12 g^2 = \tfrac{t\,\sigma_1^2}{1-t}$:

$$
\mathrm{d}x = \Big[\, v(x, t) - \frac{t\,\sigma_1^2}{1-t}\, s(x, t) \,\Big]\mathrm{d}t
            + \sigma_1\sqrt{\frac{2t}{1-t}}\;\mathrm{d}\bar w ,
$$

which is Anderson's reverse SDE of the §3 forward process, and — since $s$ is
obtained from the same model output via the §5 inversion — costs no second
model call. This is the sampler family of the stochastic-interpolants
framework ([Albergo et al., arXiv:2303.08797]) and what SiT
([Ma et al., arXiv:2401.08740]) runs on FM-trained models; the practical
appeal is the same one EDM ([Karras et al., arXiv:2206.00364]) documents for
diffusion — injected noise keeps re-contracting accumulated integration error
toward the true marginals, at the price of stopping at $t_{\max} < 1$ where
$g^2$ is finite.


## 7. What the SDE chart buys a flow-matching model

Beyond stochastic sampling, the Gaussian kernel of §2 hands flow matching the
closed forms usually filed under "diffusion":

- **Exact posterior, hence ancestral/DDIM steps.** With
  $p(x_t \mid x_0) = \mathcal{N}((1-t)x_0,\, t^2\sigma_1^2 I)$ the two-time
  posterior $p(x_s \mid x_t, x_0)$, $s < t$, is Gaussian in closed form
  (`Process.posterior`, with mean ratio $r = a_t/a_s = \tfrac{1-t}{1-s}$).
  An `Ancestral` step — estimate $x_0$ from the model, draw from the exact
  posterior — is therefore a perfectly valid flow-matching sampler, no
  drift/diffusion discretization involved.
- **Score and $x_0$ access.** Guidance, Tweedie denoising and
  likelihood-style computations all read the score, and §5 supplies it from
  the velocity output wherever $g^2 > 0$.
- **Target freedom.** The Gaussian kernel is what the score/noise training
  targets are statements about, so on this schedule they are *available*, not
  just the velocity target: one could train an $\epsilon$-head on the FM
  schedule. The schedule and the prediction target are independent axes.

None of this required adding anything to flow matching — only refusing to
delete $f$ and $g$ from it.


## 8. A non-Gaussian $x_1$: what survives, what dies, what comes back

The $(f, g)$ representation is a theorem about the *configuration*, not the
schedule. Replace the Gaussian endpoint by an arbitrary
$x_1 \sim \rho_1$ — a shape prior, a scaffold, a second dataset — and the
same schedule $a = 1-t$, $b = t$ becomes a general stochastic interpolant
([Albergo et al., arXiv:2303.08797]). Here is precisely what changes.

### 8.1 The linear chart ceases to exist

A linear SDE $\mathrm{d}x = f x\,\mathrm{d}t + g\,\mathrm{d}w$ started at
$x_0$ solves to

$$
x_t = e^{\int_0^t f}\,x_0
    + \int_0^t e^{\int_u^t f}\, g(u)\,\mathrm{d}w_u ,
$$

and the stochastic term is Gaussian for **every** choice of $f$ and $g$ — a
linear map of Brownian noise cannot be anything else. The interpolant's
conditional, $a x_0 + b x_1 \sim$ (shifted, scaled $\rho_1$), is
non-Gaussian, so no $(f, g)$ matches it; and the unconditional marginals fail
too, since the linear SDE's $p_t$ is always a Gaussian-smoothed copy of the
scaled data law while the interpolant's is $\rho_1$-smoothed. This is a
different failure mode from §3's endpoint singularity: there the chart had a
coordinate blow-up; here the object the chart would describe is not there.
$f = \mathrm{d}/\mathrm{d}t \log a$ still exists as a formula, but its
reading as "the forward drift" is gone. To reach a non-Gaussian $\rho_1$ by
an SDE at all, the drift must become state-dependent — which is the next
point.

### 8.2 Marginal-sharing SDEs still exist — but the "chart" is now the model

The velocity field
$v(x, t) = \mathbb{E}[\dot a\,x_0 + \dot b\,x_1 \mid x_t = x]$ satisfies the
continuity equation $\partial_t p_t + \nabla\!\cdot(p_t v) = 0$ for *any*
endpoint law, so the ODE $\mathrm{d}x = v\,\mathrm{d}t$ still transports
$p_{\text{data}} \to \rho_1$ forward and $\rho_1 \to p_{\text{data}}$
backward. And whenever the marginal score $s = \nabla \log p_t$ exists and is
known, the whole stochastic family survives in generalized form: for any
$\varepsilon(t) \ge 0$,

$$
\mathrm{d}x = \big[\, v \pm \varepsilon\, s \,\big]\mathrm{d}t
            + \sqrt{2\varepsilon}\;\mathrm{d}w
\qquad (+\,\varepsilon:\ \text{forward},\quad -\,\varepsilon:\ \text{reverse-time})
$$

shares the interpolant's marginals — the same Fokker–Planck cancellation as
§6, $\nabla\!\cdot(p\,\nabla\log p) = \Delta p$, with no Gaussianity used.
The Gaussian case is recovered at $\varepsilon = \tfrac12 \chi g^2$; in
general there is no canonical $\varepsilon$, which sharpens a remark from §2:
$g^2$ was never an intrinsic property of the interpolant, only the choice
that makes the drift *linear* when that is possible. What is genuinely lost
is the chart's economy — two scalar functions of time. The drift of every
surviving SDE is $v(x, t)$: state-dependent, distribution-dependent, the
learned object itself. The SDE description still exists; it has just stopped
being a *representation* you can write down before training.

### 8.3 What actually dies is score access

Every degradation in practice traces to one root: with $p(x_t \mid x_0)$ no
longer Gaussian, the score has no regression target. The velocity, $x_0$ and
pseudo-force targets are plain conditional expectations — an $L^2$ regression
converges to $\mathbb{E}[\,\cdot \mid x_t]$ whatever the endpoint law — so
training survives untouched. But Tweedie and everything built on it are
statements about the Gaussian kernel, and with it fall: the score/noise
training targets, the §5 identity $v = f x - \tfrac12 g^2 s$ and its
inversion, Tweedie $x_0$-recovery (the *direct* $x_0$ and pseudo-force
recoveries survive — they never formed a score), the closed-form posterior
$p(x_s \mid x_t, x_0)$ and with it ancestral/DDIM steps, and churn $> 0$
sampling through any of those conversions.

That this is genuine and not a formula gap, one example shows: take $\rho_1$
uniform on a sphere of radius $R$ (a caricature shape prior). Then
$p(x_t \mid x_0)$ is supported on a spherical *shell* around $a x_0$ — a
singular measure whose score does not even exist as a function, let alone
equal $-x_1 / (b\,\sigma_1^2)$.

### 8.4 Buying the score back: bridge noise

The stochastic-interpolants route to restoring score access is to add an
independent Gaussian latent to the interpolant:

$$
x_t = a(t)\,x_0 + b(t)\,x_1 + \gamma(t)\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I) .
$$

Now, *conditionally on the pair* $(x_0, x_1)$, $x_t$ is Gaussian with
variance $\gamma^2$ regardless of $\rho_1$, and Gaussian integration by parts
gives the marginal score as a conditional expectation again:

$$
s(x, t) = -\,\frac{\mathbb{E}[\,\epsilon \mid x_t = x\,]}{\gamma(t)} ,
$$

learnable by regressing $\epsilon$ — the same mechanism as denoising score
matching, with the Gaussianity supplied by the bridge term instead of the
endpoint. With $s$ in hand the whole $\varepsilon$-family of §8.2 runs, at
whatever stochasticity the sampler chooses, from a fully structured prior.
This is exactly what `Process.gamma` and the `eps` plumbing through
`perturb`/`target` are reserved for: the training target must see the *same*
$\epsilon$ that entered the interpolant, which is why `perturb` draws and
returns it. (The score is bought back only where $\gamma > 0$; schedules that
switch $\gamma$ off at the endpoints lose it there, and the smoothed
$p_t$ it refers to now includes the $\gamma$-blur.)

### 8.5 The remaining boundaries, and the code's map

Two smaller boundaries, same spirit:

- **Undeclared scale** (`prior.std` is `None`): $\sigma(t)$, and hence
  $g^2$, cannot even be formed; the routes that never form it (churn $= 0$
  velocity sampling, direct $x_0$/pseudo-force recovery) remain.
- **A coupling that re-pairs endpoints** changes $p(x_1 \mid x_0)$ even while
  preserving $x_1$'s marginal, so the one-sided kernel must be re-judged per
  coupling, not read off the marginal.

In the code the judgment is `Process.gaussian_kernel_obstruction` /
`has_gaussian_kernel` — which is why the same `FlowMatching` class is a
Gaussian diffusion under its default prior and a general stochastic
interpolant under a structured one, with the closed forms gated by the
configuration rather than the class. Under a non-Gaussian endpoint the
supported surface is: velocity / $x_0$ / pseudo-force training, churn $= 0$
`Sampler`, and `DirectDenoisingSampler`; the score/noise parametrizations
refuse at `validate`, and the Gaussian closed forms raise when reached.

And the two time-endpoint guards of §3, restated in code terms:
$t_{\max} < 1$ protects consumers of $f, g^2$ (churn $> 0$,
`AncestralDDPM`) from the finite-time-prior singularity, which pure-ODE use
may waive; $t_{\min} > 0$ protects everything that touches the score from
the genuine collapse of $\sigma \to 0$, and no chart waives that.


## 9. Dictionary: math ↔ code

| math | code |
| --- | --- |
| $a(t),\ b(t)$ | `Process.a`, `Process.b` (`FlowMatching`: $1-t$, $t$) |
| $\sigma_1$ | `Process.std` (= `prior.std`) |
| $\sigma(t) = b\,\sigma_1$ | `Process.sigma` |
| $f = \mathrm{d}/\mathrm{d}t \log a$ | `Process.f` (= `Process.log_a_dot`) |
| $g^2 = -\sigma^2\,(\log\operatorname{SNR})'$ | `Process.g2` |
| $\log\operatorname{SNR} = 2(\log a - \log b)$ | `Process.log_snr` |
| $x_t = a x_0 + b x_1$ | `Process.interpolate` / `Process.perturb` |
| $p(x_t \mid x_0) = \mathcal{N}(a x_0, \sigma^2 I)$ | `Process.kernel`, gated by `has_gaussian_kernel` |
| $p(x_s \mid x_t, x_0)$ | `Process.posterior`; stepped by `integrators.Ancestral` |
| velocity target $\dot a\,x_0 + \dot b\,x_1$ | `VelocityParametrization.target` |
| $v = f x - \tfrac12 g^2 s$ | `Parametrization.to_velocity` |
| $s = 2(f x - v)/g^2$ | `VelocityParametrization.to_score` |
| Tweedie $\mathbb{E}[x_0 \mid x_t] = (x + \sigma^2 s)/a$ | `Parametrization.to_x0` / `X0Parametrization.to_score` |
| churn family drift / diffusion, $\chi = \eta^2$ (the $\varepsilon = \tfrac12 \chi g^2$ instance of §8.2) | `ReverseProcess.drift` / `ReverseProcess.diffusion`, `churn` |
| bridge noise $\gamma(t)\,\epsilon$; $s = -\mathbb{E}[\epsilon \mid x_t]/\gamma$ | `Process.gamma`; the `eps` drawn and returned by `Process.perturb` |


## References

- Y. Lipman et al., *Flow Matching for Generative Modeling*, arXiv:2210.02747.
- X. Liu, C. Gong, Q. Liu, *Flow Straight and Fast* (rectified flow), arXiv:2209.03003.
- M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, *Stochastic Interpolants*, arXiv:2303.08797.
- Y. Song et al., *Score-Based Generative Modeling through SDEs*, arXiv:2011.13456.
- B. D. O. Anderson, *Reverse-time diffusion equation models*, Stoch. Proc. Appl. 12 (1982).
- N. Ma et al., *SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers*, arXiv:2401.08740.
- T. Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, arXiv:2206.00364.

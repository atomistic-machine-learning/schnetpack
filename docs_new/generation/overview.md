# Overview

Diffusion-based generative models learn to undo a noising process. A neural network is trained on pairs of clean structures $x_0$, drawn from the data distribution $\rho_0$, and noisy structures $x_t$ derived from them. Generation runs the noising backwards: it starts from a sample $x_1$ of a prior distribution $\rho_1$, which is easy to sample, and follows the learned field to a clean structure.

Noisy structures can come from many sources. We mainly use an **interpolant** between a data sample $x_0$ and a prior sample $x_1$,

$$
x_t=a(t)\,x_0+b(t)\,x_1 ,
$$

where the schedule $a(t)$, $b(t)$ sets how fast the structure is noised. The network learns a **field** $\phi$: given a noisy structure $x_t$ and, optionally, its time $t$, it returns a direction that guides $x_t$ toward a clean structure. The noising process decides which fields can be learned, and the choice among them is the parametrization. To generate, we draw structures from the prior and move them with the field, step by step, until they are clean.

**Conventions.** A structure is a vector $x\in\mathbb R^d$. Time runs over $t\in[0,1]$, with **$t=0$ data and $t=1$ prior**, so generation runs from $t=1$ down to $t=0$. A dot is a time derivative, a hat a network estimate, a star a regression target.

---

## 1 Training

Training starts with the [[noising|noising process]]. We choose a prior $\rho_1$, a [[noising#2 Interpolant|coupling]] $\nu$ that pairs data samples with prior samples, and a [[schedules|schedule]] $a(t)$, $b(t)$. The common choice is a [[noising#3 Gaussian interpolant|Gaussian prior]] $x_1=\sigma_{\max}\epsilon$ with $\epsilon\sim\mathcal N(0,I)$, drawn independently of $x_0$; the noise level along the path is then $\sigma(t)=\sigma_{\max}b(t)$.

Next we choose the **[[parametrizations|parametrization]]**, i.e. the field $\phi$ the network learns, together with its regression target $\phi^\star$. The noising restricts the choice. The pseudo force only needs pairs of clean and noisy structures, the velocity needs an interpolant, and noise and score need a Gaussian prior with independent coupling:

| Field $\phi$ | Target $\phi^\star$ | Requires |
|---|---|---|
| pseudo force $F$ | $x_0-x_t$ | any noising |
| velocity $v$ | $\dot a\,x_0+\dot b\,x_1$ | interpolant |
| noise $\bar\epsilon$ | $\epsilon$ | Gaussian prior, independent coupling |
| score $s$ | $-\epsilon/\sigma$ | Gaussian prior, independent coupling |

We also choose whether the network sees the time. A **[[time|time-aware]]** model $\hat\phi(t,x)$ is told how noisy its input is; a **time-agnostic** model $\hat\phi(x)$ has to infer it from the structure.

Each training step draws a data sample $x_0\sim\rho_0$, a time $t$ from a training distribution and a prior sample $x_1$ from the coupling, forms $x_t$ and computes the target $\phi^\star$. The network is fit with the loss

$$
\mathcal L=\mathbb E\,w(t)\,\big\lVert\hat\phi(t,x_t)-\phi^\star\big\rVert^2 ,
$$

where the optional loss weight $w(t)\ge0$ normalizes the magnitude of the target, which for most fields changes strongly with the noise level. The same noisy structure can arise from many clean ones, so the target is itself noisy, and the network learns its conditional mean $\phi(t,x)=\mathbb E[\phi^\star\mid x_t=x]$.

---

## 2 Structure generation

Once the field model is trained, it can generate new structures. Generation starts from structures $x_1\sim\rho_1$, usually drawn from the prior used in training, and moves them step by step with the field until they are clean. There are three approaches, described in [[generation]]:

| Approach     | Field            | Noising                              | Model                                           | Result                |
| ------------ | ---------------- | ------------------------------------ | ----------------------------------------------- | --------------------- |
| Optimization | pseudo force $F$ | any                                  | time-agnostic                                   | relaxed structures    |
| ODE sampling | velocity $v$     | interpolant                          | time-aware or time-agnostic | samples from $\rho_0$ |
| SDE sampling | score $s$        | Gaussian prior, independent coupling | time-aware or time-agnostic | samples from $\rho_0$ |

**Sampling** runs the path backwards on a time grid from $t=1$ to $t=0$. An ODE sampler follows the velocity, e.g. with the Euler step $x\leftarrow x-h\,\hat v(t,x)$ of size $h$; an SDE sampler follows the score and adds fresh noise in every step. With an exact field and small steps, both return samples from the data distribution $\rho_0$. **Optimization** has no time grid. It repeats $x\leftarrow x+\hat F(x)$, a jump to the model's estimate of the clean structure, until the force is small. It returns structures the model considers clean, close to the most likely structures of the data, but not samples from $\rho_0$.

Which approaches are available is decided in training. Every parametrization converts into the fields its noising allows, so with a Gaussian prior and independent coupling a velocity model can also drive the SDE, and a score model the ODE. For sampling, the model can be time-aware or time-agnostic. Optimization does not follow a schedule, so the time of the current structure is unknown and time-aware models cannot be used. Since a time-agnostic model does not rely on a given time, it can also start from structures other than prior samples, e.g. perturbed versions of known structures.

Generation can also be [[generation#3 Constraints and guidance|steered]]. Constraints restrict it, e.g. to keep part of a structure fixed, and conditioning guides it toward structures with desired properties.

---

## 3 Validation

How generated structures are evaluated is described in [[validation]].

---

## Notation

**Distributions and samples**

| Symbol | Meaning | Defined in |
|---|---|---|
| $\rho_0$, $x_0$ | data distribution, data sample | [[noising]] |
| $\rho_1$, $x_1$ | prior, prior sample | [[noising#2 Interpolant|noising §2]] |
| $\nu$ | coupling: joint distribution of $(x_0,x_1)$ with marginals $\rho_0$, $\rho_1$ | [[noising#2 Interpolant|noising §2]] |
| $\tilde x$ | noisy sample of a general noising process | [[noising#1 Noising process|noising §1]] |
| $x_t$, $\rho_t$ | noisy sample at time $t$, its marginal distribution | [[noising#2 Interpolant|noising §2]] |

**Interpolant and schedule**

| Symbol | Meaning | Defined in |
|---|---|---|
| $a(t)$, $b(t)$ | interpolant coefficients, $x_t=a\,x_0+b\,x_1$, with $a(0)=b(1)=1$, $b(0)=0$ | [[noising#2 Interpolant|noising §2]] |
| $D(t)$ | $a\dot b-\dot a\,b>0$ | [[noising#2 Interpolant|noising §2]] |
| $\epsilon$, $\sigma_{\max}$ | unit Gaussian noise and prior width, $x_1=\sigma_{\max}\epsilon$ (Gaussian level) | [[noising#3 Gaussian interpolant|noising §3]] |
| $\sigma(t)$ | noise level $\sigma_{\max}\,b(t)$ (Gaussian level) | [[noising#3 Gaussian interpolant|noising §3]] |
| $f(t)$, $g^2(t)$ | drift $\dot a/a$ and diffusion $2\sigma_{\max}\sigma D/a$ (Gaussian level) | [[parametrizations#4 Velocity|parametrizations §4]] |
| $\mathrm{SNR}(t)$, $\lambda(t)$ | signal-to-noise ratio $a^2/\sigma^2$ and its logarithm | [[schedules#2 Schedules for Gaussian interpolants|schedules §2]] |
| $\mathrm{TV}(t)$ | total variance $a^2+\sigma^2$ | [[schedules#2 Schedules for Gaussian interpolants|schedules §2]] |

**Fields**

| Symbol | Meaning | Defined in |
|---|---|---|
| $\eta_0(t,x)$ | denoiser $\mathbb E[x_0\mid x_t=x]$ | [[noising#1 Noising process|noising §1]] |
| $F(t,x)$ | pseudo force $\eta_0-x$ | [[noising#1 Noising process|noising §1]] |
| $v(t,x)$ | velocity $\mathbb E[\dot x_t\mid x_t=x]$ | [[noising#2 Interpolant|noising §2]] |
| $s(t,x)$ | score $\nabla_x\log\rho_t(x)$ | [[noising#3 Gaussian interpolant|noising §3]] |
| $\bar\epsilon(t,x)$ | expected noise $\mathbb E[\epsilon\mid x_t=x]$ | [[noising#3 Gaussian interpolant|noising §3]] |

**Training**

| Symbol | Meaning | Defined in |
|---|---|---|
| $\phi$ | the field the network learns, one of $F$, $v$, $\bar\epsilon$, $s$ | [[parametrizations#1 Training|parametrizations §1]] |
| $\phi^\star$, $\hat\phi$ | regression target, network output | [[parametrizations#1 Training|parametrizations §1]] |
| $w(t)$ | loss weight, normalizes the target magnitude | [[parametrizations#1 Training|parametrizations §1]] |
| $\hat t(x)$ | time predictor | [[time#2.1 Explicit time estimation|time §2.1]] |

**Generation**

| Symbol | Meaning | Defined in |
|---|---|---|
| $X_t$ | sampler iterate | [[generation#2 Sampling|generation §2]] |
| $h$ | sampler time step, one step goes from $t$ to $t-h$ | [[generation#2 Sampling|generation §2]] |
| $t_{\min}$, $t_{\max}$ | ends of the sampling time grid | [[generation#2.3 Time grid|generation §2.3]] |
| $\gamma$ | stochasticity of the reverse SDE | [[generation#2.2 SDE sampling|generation §2.2]] |
| $\kappa$ | optimizer step size | [[generation#1 Optimization|generation §1]] |
| $F_{\max}$, $n_{\text{steps}}$ | optimizer stopping threshold and step limit | [[generation#1 Optimization|generation §1]] |

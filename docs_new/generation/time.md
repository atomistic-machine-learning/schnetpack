# Time

The fields of an interpolant depend on time, $\phi(t,x)$. A model can receive the time as an input or not. This choice decides which generation methods the model can drive and from which structures generation can start. Notation follows the [[overview#Notation|overview]].

| Model                                      | Network         | Optimization | Sampling |
| ------------------------------------------ | --------------- | ------------ | -------- |
| [[#1 Time-aware models\|time-aware]]       | $\hat\phi(t,x)$ | no           | yes      |
| [[#2 Time-agnostic models\|time-agnostic]] | $\hat\phi(x)$   | yes          | yes      |

---

## 1 Time-aware models

A time-aware model $\hat\phi(t,x)$ is told how noisy its input is and learns the field $\phi(t,x)=\mathbb E[\phi^\star\mid x_t=x]$ at every time. During sampling, the time comes from the sampler's time grid.

The model is trained only on pairs $(t,x_t)$ from the noising, so it is reliable only where $x$ is a typical noisy structure at time $t$. Generation therefore has to start from the prior, or from structures whose time is known. Optimization follows no schedule, so the time of the current structure is unknown and time-aware models cannot be used.

---

## 2 Time-agnostic models

A time-agnostic model does not receive the time during inference from outside. It infers the noise level from the structure, either explicitly with a time predictor or implicitly inside the network.

### 2.1 Explicit time estimation

A time predictor $\hat t(x)$ estimates the time of a structure, and its estimate is passed as the time input of the field model, $\hat\phi(\hat t(x),x)$. The combination is time-agnostic. In practice, both share one representation $h(x)$ with two output heads: the time head predicts $\hat t$ from $h$, and the field head takes $h$ and $\hat t$. Both are trained jointly on the noising pairs,

$$
\mathcal L=\mathbb E\Big[(1-\lambda)\,w(t)\,\big\lVert\hat\phi\big(\hat t(x_t),x_t\big)-\phi^\star\big\rVert^2+\lambda\,\big(\hat t(x_t)-t\big)^2\Big],
$$

with $\lambda=0.1$ in MoreRed [1]. The field loss also trains the time head, since its gradient passes through $\hat t$.

### 2.2 Implicit time estimation

The network $\hat\phi(x)$ only sees the structure. Trained on the same pairs as a time-aware model, it learns the conditional mean over all times,

$$
\phi(x)=\mathbb E[\phi^\star\mid x_t=x]=\mathbb E\big[\phi(t,x)\mid x_t=x\big],
$$

i.e. the field averaged over the times that could have produced $x$, weighted by the training distribution of $t$ [2].

In theory, every field can be learned this way, as in blind denoising [2]. In practice, sampling quality suffers: removing the time input from a diffusion model strongly reduces the validity of the generated structures [3]. Fields whose magnitude scales with the noise level, and exploding schedules, are different: the noise level, and with it the time, is encoded in the magnitude of the structure and of the prediction. GPFF, for example, uses a VE schedule and the pseudo force, whose target is $-\sigma\epsilon$. It estimates the noise level from the variance present in the structure, and therefore also in the prediction [3],

$$
\hat\sigma=\operatorname{std}\big(\hat F(x)\big),
$$

the standard deviation over the $d$ components of $\hat F$.

---

## 3 Time in generation

**Optimization** needs a time-agnostic model, since it has no time to give.

**Sampling** steps on a time grid, and the conversions to the velocity and the score use $a(t)$ and $b(t)$ of the current time. A time-aware model gets the grid time as input. A time-agnostic model ignores it, and the sampler uses the grid time only for the step and the conversion. With a time predictor, the sampler can also use $\hat t(x)$ instead of the grid time.

**Starting structures.** A time-agnostic model does not rely on a given time, so generation can start from structures other than prior samples, e.g. perturbed versions of known structures. A time predictor or the pseudo-force magnitude then gives the time at which sampling starts [1, 3].

---

## References

1. Kahouli, Hessmann, Müller, Nakajima, Gugler, Gebauer (2024). *Molecular Relaxation by Reverse Diffusion with Time Step Prediction.* Machine Learning: Science and Technology 5, 035038. arXiv:2404.10935.
2. Sun, Jiang, Zhao, He (2025). *Is Noise Conditioning Necessary for Denoising Generative Models?* ICML. arXiv:2502.13129.
3. Hessmann, Kahouli, Gugler, Plainer, Noé, Müller, Gebauer (2026). *Generative Pseudo-Force Fields for Molecular Generation.* arXiv:2605.19050.

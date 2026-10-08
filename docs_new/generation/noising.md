# Noising

The generative models learn to undo noising by training a denoising field $\phi$. The network takes a noisy sample $\tilde x$, and optionally its time $t$, as input and predicts one of the field targets defined by the parametrization. The key ingredient is that every noisy sample $\tilde x$ is assigned to a clean structure $x_0$, and the noising process generates these pairs. Depending on the parametrization, the noising has to meet different requirements, and different parametrizations allow different generative processes. Notation follows the [[overview#Notation|overview]].

| Level | Noising | Fields | Generation |
|---|---|---|---|
| [[#1 Noising process\|1 Noising process]] | any pairs $(x_0,\tilde x)$ | $\eta_0$, $F$ | optimization |
| [[#2 Interpolant\|2 Interpolant]] | prior, coupling, schedule $a$, $b$ | $+\ v$ | $+$ ODE sampling |
| [[#3 Gaussian interpolant\|3 Gaussian interpolant]] | Gaussian prior, independent coupling | $+\ \bar\epsilon$, $s$ | $+$ SDE sampling |

Each level is a special case of the previous one: a more restricted noising allows more fields.

---

## 1 Noising process

A general noising process is the least restricted way to noise. It only generates a noisy sample $\tilde x\sim q(\cdot\mid x_0)$ for each clean sample $x_0\sim\rho_0$. There are no restrictions on a prior distribution, the noise structure or a schedule; the process need not be known in closed form, and it may or may not carry a time label. Examples are Gaussian noise with any schedule, random perturbations, or clean and perturbed pairs that come with the data.

The pairs define the **denoiser** and the **pseudo force**,

$$
\eta_0(x)=\mathbb E[x_0\mid\tilde x=x],\qquad F(x)=\eta_0(x)-x ,
$$

the best estimate of the clean structure and the expected displacement toward it [1]. The trade-off is that without a schedule, only fields that are independent of the schedule and the noise structure can be trained, i.e. the denoiser and the [[parametrizations#3 Pseudo force|pseudo force]]. And without a schedule there is no path in time from a prior to the data, so generation is limited to [[generation#1 Optimization|optimization]].

---

## 2 Interpolant

Interpolants are noising processes with more restrictions. They need a prior distribution $\rho_1$ and a schedule $a(t)$, $b(t)$ [2–4]. For each training pair, a data sample $x_0\sim\rho_0$ and a prior sample $x_1\sim\rho_1$ are drawn from a coupling $\nu$:

- **Independent**, $\nu=\rho_0\otimes\rho_1$: each data sample gets a fresh prior sample. The default.
- **Paired**: $x_1$ is a partner of $x_0$ given by the data.
- **Optimal transport**: data and prior samples in a minibatch are re-paired to minimize $\sum\lVert x_0-x_1\rVert^2$, which straightens the paths [5, 6].

From $x_0$, $x_1$ and the schedule, the noisy structure is

$$
x_t=a(t)\,x_0+b(t)\,x_1,\qquad a(0)=1,\ b(0)=0,\ b(1)=1 .
$$

We assume $D(t):=a\dot b-\dot a\,b>0$, i.e. the noise-to-signal ratio $b/a$ strictly increases, $\tfrac{d}{dt}(b/a)=D/a^2$, so the structure gets noisier with $t$. Generation starts from $\rho_1$, so $x_{t=1}=a(1)\,x_0+x_1$ must be (close to) $\rho_1$. Either $a(1)=0$, as in flow matching ($a=1-t$, $b=t$), or the prior is much wider than the data, as in variance exploding (VE) diffusion ($a\equiv1$). The common choices of $a$ and $b$ are collected in [[schedules]].

Introducing the schedule gives access to the **velocity**, the expected time derivative of the noisy structure,

$$
v(t,x)=\mathbb E[\dot x_t\mid x_t=x]=\mathbb E[\dot a\,x_0+\dot b\,x_1\mid x_t=x] .
$$

It satisfies the continuity equation $\partial_t\rho_t+\nabla\cdot(v\,\rho_t)=0$, so integrating it from $t=1$ to $t=0$ transports $\rho_1$ to $\rho_0$ [3, 7, 8]. The [[parametrizations#4 Velocity|velocity parametrization]] therefore allows [[generation#2.1 ODE sampling|ODE sampling]] in addition to optimization. Sampling reproduces only the marginal $\rho_0$, not the pairing of the coupling.

---

## 3 Gaussian interpolant

Gaussian interpolants are interpolants with further constraints. The prior sample is Gaussian noise, $x_1=\sigma_{\max}\epsilon$ with $\epsilon\sim\mathcal N(0,I)$, and the coupling must be independent. Replacing $x_1$ by the noise gives

$$
x_t=a(t)\,x_0+\sigma(t)\,\epsilon,\qquad \sigma(t)=\sigma_{\max}\,b(t) .
$$

The more restricted noising allows more parametrizations. The conditional law $\rho_t(x\mid x_0)=\mathcal N(x;\,a\,x_0,\,\sigma^2 I)$ is Gaussian, so besides the [[parametrizations#5 Noise|expected noise]] $\bar\epsilon=\mathbb E[\epsilon\mid x_t]$ the **[[parametrizations#6 Score|score]]** is available [9],

$$
s(t,x)=\nabla_x\log\rho_t(x)=-\frac{\mathbb E[\epsilon\mid x_t=x]}{\sigma} .
$$

The score allows [[generation#2.2 SDE sampling|SDE sampling]] in addition to ODE sampling. Both constraints are needed: with a non-Gaussian prior, or with a paired or optimal-transport coupling, $x_1$ given $x_0$ is not Gaussian, and $-\mathbb E[\epsilon\mid x_t]/\sigma$ is not the score.

---

## References

1. Hessmann, Kahouli, Gugler, Plainer, Noé, Müller, Gebauer (2026). *Generative Pseudo-Force Fields for Molecular Generation.* arXiv:2605.19050.
2. Albergo, Vanden-Eijnden (2023). *Building Normalizing Flows with Stochastic Interpolants.* ICLR. arXiv:2209.15571.
3. Albergo, Boffi, Vanden-Eijnden (2023). *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.* arXiv:2303.08797.
4. Kahouli, Ripken, Gugler, Unke, Müller, Nakajima (2025). *Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models.* arXiv:2502.08598.
5. Pooladian, Ben-Hamu, Domingo-Enrich, Amos, Lipman, Chen (2023). *Multisample Flow Matching: Straightening Flows with Minibatch Couplings.* ICML. arXiv:2304.14772.
6. Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf, Bengio (2024). *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.* TMLR. arXiv:2302.00482.
7. Lipman, Chen, Ben-Hamu, Nickel, Le (2023). *Flow Matching for Generative Modeling.* ICLR. arXiv:2210.02747.
8. Liu, Gong, Liu (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR. arXiv:2209.03003.
9. Vincent (2011). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation 23(7), 1661–1674.

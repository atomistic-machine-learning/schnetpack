---
aliases: [Schedule, noise schedule, interpolant schedule]
tags: [spk3, generative-models, schedules]
sources: []
created: 2026-10-07
updated: 2026-10-07
---

# Schedules

The schedule $a(t)$, $b(t)$ sets how fast a structure moves from the data at $t=0$ to the prior at $t=1$. It fixes the noise level at every time, and with it which noise levels training and sampling see. This page collects the common schedules of flow matching and diffusion models, in the form they are known in the literature. Notation follows the [[overview#Notation|overview]].

---

## 1 Schedules for arbitrary interpolants

A schedule for an arbitrary [[noising#2 Interpolant|interpolant]] defines $a$ and $b$ directly. It works with any prior and has to satisfy $a(0)=1$, $b(0)=0$, $b(1)=1$ and $D=a\dot b-\dot a\,b>0$. These are the schedules of flow matching. With a Gaussian prior $\mathcal N(0,\sigma_{\max}^2I)$, the noise level is $\sigma(t)=\sigma_{\max}b(t)$, usually with $\sigma_{\max}=1$.

### 1.1 Linear

The default of flow matching [1, 2]:

$$
a(t)=1-t,\qquad b(t)=t .
$$

The conditional paths are straight lines from data to prior, and the velocity target $x_1-x_0$ has the same scale at every $t$. Lipman et al. [2] keep a small noise level at the data end, $b(t)=\sigma_{\min}+(1-\sigma_{\min})\,t$.

### 1.2 Convex

The linear schedule generalizes to

$$
a(t)=1-\psi(t),\qquad b(t)=\psi(t),
$$

with any increasing $\psi$ from $\psi(0)=0$ to $\psi(1)=1$, e.g. a polynomial [3].

### 1.3 Trigonometric

A path on which $a^2+b^2=1$ [4, 5]:

$$
a(t)=\cos\tfrac{\pi t}{2},\qquad b(t)=\sin\tfrac{\pi t}{2} .
$$

It is used for stochastic interpolants [4] and as the GVP path of SiT [5]. With a unit Gaussian prior it is a variance-preserving schedule, the cosine schedule of §2.1 with offset $s=0$.

---

## 2 Schedules for Gaussian interpolants

For a [[noising#3 Gaussian interpolant|Gaussian interpolant]], $x_t=a\,x_0+\sigma\,\epsilon$, a schedule can be defined in two ways.

**Signal and noise.** Diffusion models define $a(t)$ and a noise level $\sigma(t)$ that grows to a largest value $\sigma_{\max}$ at $t=1$. In the interpolant view, $\sigma_{\max}$ belongs to the prior, $\rho_1=\mathcal N(0,\sigma_{\max}^2I)$, and the schedule is the normalized noise level

$$
b(t)=\frac{\sigma(t)}{\sigma_{\max}} .
$$

**Total variance and SNR.** Alternatively, a schedule is fixed by its total variance and its signal-to-noise ratio [6],

$$
\mathrm{TV}(t)=a^2+\sigma^2,\qquad \mathrm{SNR}(t)=\frac{a^2}{\sigma^2},\qquad \lambda(t)=\log\mathrm{SNR}(t),
$$

which can be chosen independently. They give back

$$
a=\sqrt{\frac{\mathrm{TV}\cdot\mathrm{SNR}}{1+\mathrm{SNR}}},\qquad \sigma=\sqrt{\frac{\mathrm{TV}}{1+\mathrm{SNR}}} .
$$

The SNR decides how much of the data is still visible at time $t$, and the condition $D>0$ means that it strictly decreases. The TV decides the overall scale of $x_t$.

> [!note] Unit variance
> TV is the variance of $x_t$ only for data with zero mean and unit variance, the assumption of [6]. Structures usually do not have unit variance, and then $\operatorname{Var}(x_t)=a^2\operatorname{Var}(x_0)+\sigma^2$. TV still characterizes the schedule, but not the actual scale of $x_t$.

### 2.1 Variance preserving (VP)

VP diffusion defines the schedule through a noise rate $\beta(t)$ in the forward SDE $dx=-\tfrac12\beta(t)\,x\,dt+\sqrt{\beta(t)}\,dW$ [7], the continuous limit of DDPM [8]:

$$
a(t)=\exp\Big(-\tfrac12\int_0^t\beta(s)\,ds\Big),\qquad \sigma(t)=\sqrt{1-a(t)^2} .
$$

The total variance is constant, $\mathrm{TV}=1$. The prior is $\mathcal N(0,I)$, so $\sigma_{\max}=1$ and $b=\sigma$. DDPM writes $\bar\alpha_t$ for $a^2$.

- **Linear $\beta$** [7, 8]: $\beta(t)=\beta_{\min}+t\,(\beta_{\max}-\beta_{\min})$, typically $\beta_{\min}=0.1$, $\beta_{\max}=20$. Then $a(t)=\exp\big(-\tfrac12\beta_{\min}t-\tfrac14(\beta_{\max}-\beta_{\min})\,t^2\big)$. $a(1)\approx0.007$ is small but not zero, so the noisy structure at $t=1$ is only approximately the prior.
- **Cosine** [9]: $a(t)=\cos\big(\tfrac\pi2\tfrac{t+s}{1+s}\big)\big/\cos\big(\tfrac\pi2\tfrac{s}{1+s}\big)$ with a small offset $s=0.008$. Here $a(1)=0$ exactly.

### 2.2 Variance exploding (VE)

VE diffusion keeps the signal and only adds noise, $a\equiv1$, so the total variance $\mathrm{TV}=1+\sigma^2$ grows with the noise level. The prior is $\mathcal N(0,\sigma_{\max}^2I)$, and $\sigma_{\max}$ must be much larger than the spread of the data so that $x_0+\sigma_{\max}\epsilon$ is close to the prior. A common choice is the largest distance between two data points [10].

- **Geometric** [7, 11]: $\sigma(t)=\sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t$, so $b(t)=(\sigma_{\min}/\sigma_{\max})^{1-t}$, with e.g. $\sigma_{\min}=0.01$. Since $b(0)=\sigma_{\min}/\sigma_{\max}>0$, the path starts at slightly noised data, and samplers stop there.
- **EDM** [12]: time equals noise level, $\sigma(t)=\sigma_{\max}t$, so $b(t)=t$, with $\sigma_{\max}=80$. This is the $b$ of the linear schedule, but with $a\equiv1$. The sampling grid and the training distribution of EDM are separate choices, not part of the schedule.

### 2.3 TV/SNR schedules

Kahouli et al. [6] choose TV and SNR separately. VP and VE schedules with the same SNR differ only in their TV, and replacing an exploding TV by a constant one often improves generation. Their VP-ISSNR schedule combines a constant total variance with an inverse-sigmoid SNR,

$$
\mathrm{TV}(t)=1,\qquad \mathrm{SNR}(t)=e^{2c}\Big(\frac{1}{\tilde t}-1\Big)^{2k},\qquad \tilde t=t_{\min}+(t_{\max}-t_{\min})\,t ,
$$

with steepness $k>0$ and offset $c$ (called $\eta$ and $\kappa$ in [6]). For $k=1$, $c=0$, $t_{\min}=0$, $t_{\max}=1$, the SNR is $\big((1-t)/t\big)^2$, the SNR of the linear schedule. The values recommended in [6] are $k=1$, $c=2$, $t_{\min}=0.01$, $t_{\max}=0.99$.

---

## 3 Summary

| Schedule | $a(t)$ | $b(t)$ | Typical $\sigma_{\max}$ | Used in |
|---|---|---|---|---|
| Linear | $1-t$ | $t$ | $1$ | flow matching |
| Trigonometric | $\cos\frac{\pi t}{2}$ | $\sin\frac{\pi t}{2}$ | $1$ | flow matching, stochastic interpolants |
| VP, linear $\beta$ | $e^{-\frac12\beta_{\min}t-\frac14(\beta_{\max}-\beta_{\min})t^2}$ | $\sqrt{1-a^2}$ | $1$ | diffusion |
| VP, cosine | $\cos\big(\frac\pi2\frac{t+s}{1+s}\big)/\cos\big(\frac\pi2\frac{s}{1+s}\big)$ | $\sqrt{1-a^2}$ | $1$ | diffusion |
| VE, geometric | $1$ | $(\sigma_{\min}/\sigma_{\max})^{1-t}$ | data spread | diffusion |
| VE, EDM | $1$ | $t$ | $80$ | diffusion |
| VP-ISSNR | $\sqrt{\mathrm{SNR}/(1+\mathrm{SNR})}$ | $\sqrt{1/(1+\mathrm{SNR})}\,/\,\sigma_{\max}$ | $\approx1$ | diffusion |

---

## References

1. Liu, Gong, Liu (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR. arXiv:2209.03003.
2. Lipman, Chen, Ben-Hamu, Nickel, Le (2023). *Flow Matching for Generative Modeling.* ICLR. arXiv:2210.02747.
3. Lipman, Havasi, Holderrieth, Shaul, Le, Karrer, Chen, Lopez-Paz, Ben-Hamu, Gat (2024). *Flow Matching Guide and Code.* arXiv:2412.06264.
4. Albergo, Boffi, Vanden-Eijnden (2023). *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.* arXiv:2303.08797.
5. Ma, Goldstein, Albergo, Boffi, Vanden-Eijnden, Xie (2024). *SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers.* ECCV. arXiv:2401.08740.
6. Kahouli, Ripken, Gugler, Unke, Müller, Nakajima (2025). *Disentangling Total-Variance and Signal-to-Noise-Ratio Improves Diffusion Models.* arXiv:2502.08598.
7. Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR. arXiv:2011.13456.
8. Ho, Jain, Abbeel (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS. arXiv:2006.11239.
9. Nichol, Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML. arXiv:2102.09672.
10. Song, Ermon (2020). *Improved Techniques for Training Score-Based Generative Models.* NeurIPS. arXiv:2006.09011.
11. Song, Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS. arXiv:1907.05600.
12. Karras, Aittala, Aila, Laine (2022). *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS. arXiv:2206.00364.

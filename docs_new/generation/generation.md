# Generation

Generation starts from noisy structures $\tilde x$ or $x_t$ and turns them into clean equilibrium structures. This can be done in two ways. **Optimization**, also called relaxation, follows the field with a step size until it vanishes. **Sampling** follows an ODE or SDE along a time schedule from the prior to the data. Both end in equilibrium structures, but sampling obeys the data distribution, while optimization finds the nearest equilibrium structure, so the distribution of its results depends on the starting structures. Both can be steered with constraints and guidance. Which fields are available depends on the noising used in training, see the [[parametrizations#2 Hierarchy|hierarchy of parametrizations]]. Notation follows the [[overview#Notation|overview]].

---

## 1 Optimization

Any field that guides a noisy structure to equilibrium can be used for optimization: the physical force $F=-\nabla E$ of an MLFF, the [[parametrizations#3 Pseudo force|pseudo force]], but also velocity, noise or score. The field model must be [[time|time-agnostic]], since there is no time schedule, and the results do not follow the data distribution.

Physical and pseudo forces suit optimization best: they are large far from equilibrium and vanish at it, so a fixed step size gives steps that shrink near the minimum. The noise keeps the same magnitude, and the score grows near equilibrium, so they would need a changing step size. The velocity follows $\dot b$: for the linear schedule it keeps its magnitude like the noise.

**Step rules.** They stop once the largest force component is below $F_{\max}$, or after $n_{\text{steps}}$.

- *Direct denoising* [1]: the pseudo energy $U=\tfrac12\lVert x-x_0\rVert^2$ is quadratic with identity Hessian, so its Newton step is exact and jumps to the clean estimate, $x\leftarrow x+\hat F(x)=\hat\eta_0(x)$. It does not apply to physical forces.
- *Gradient descent*: $x\leftarrow x+\kappa\,\hat F(x)$, for physical and pseudo forces.
- *L-BFGS* [2]: quasi-Newton steps, for physical and pseudo forces.

---

## 2 Sampling

Sampling transports a distribution. A sampler starts from the prior and integrates a differential equation whose marginals follow $\rho_t$, on a time grid from $t_{\max}$ to $t_{\min}$; one step of size $h>0$ goes from $t$ to $t-h$. The grid is fixed in advance, and the number of steps sets the discretization error. With an exact field and small steps, the result is a sample from $\rho_0$, including its diversity. The model can be time-aware or time-agnostic, see [[time#3 Time in generation|time in generation]].

### 2.1 ODE sampling

On any [[noising#2 Interpolant|interpolant]], the [[parametrizations#4 Velocity|velocity]] transports the prior to the data [3, 4],

$$
\frac{dX}{dt}=v(t,X),\qquad X_1\sim\rho_1,\quad t:1\to0 ,
$$

so that $X_0\sim\rho_0$. With a paired or optimal-transport coupling, the ODE reproduces only the marginal $\rho_0$, not the pairing. On a Gaussian interpolant it is the probability-flow ODE $v=f\,X-\tfrac12 g^2 s$ [5].

**Integrators.**

- *Euler* (one evaluation per step):
  $$X_{t-h}=X_t-h\,\hat v(t,X_t).$$
- *Heun* (two evaluations per step, second order; the EDM sampler [6]):
  $$\tilde X=X_t-h\,\hat v(t,X_t),\qquad X_{t-h}=X_t-\tfrac h2\big(\hat v(t,X_t)+\hat v(t-h,\tilde X)\big).$$

### 2.2 SDE sampling

On a [[noising#3 Gaussian interpolant|Gaussian interpolant]], the [[parametrizations#6 Score|score]] gives a family of stochastic samplers. They reverse the forward SDE

$$
dX=f\,X\,dt+g\,dW,\qquad X_0\sim\rho_0 ,
$$

whose transition kernel is $\mathcal N(a\,x_0,\sigma^2 I)$, so its marginals are the $\rho_t$ of the interpolant [5]. It is never simulated; it only defines $f$ and $g$. For any $\gamma\ge0$, the reverse SDE, run backward in time,

$$
dX=\Big(f\,X-\tfrac12\big(1+\gamma^2\big)\,g^2\,s(t,X)\Big)\,dt+\gamma\,g\;d\bar W,\qquad X_1\sim\mathcal N(0,\sigma_{\max}^2 I),
$$

has the same marginals [5, 7]. $\gamma=0$ is the probability-flow ODE of §2.1, $\gamma=1$ the reverse-time SDE, and $\gamma$ plays the role of $\eta$ in DDIM [8]. In velocity form the drift is $v-\tfrac12\gamma^2 g^2 s$: the ODE drift plus a pull toward higher density that balances the injected noise.

**Integrators.** With $\xi\sim\mathcal N(0,I)$:

- *Euler–Maruyama*:
  $$X_{t-h}=X_t-h\Big(f\,X_t-\tfrac12(1+\gamma^2)\,g^2\,\hat s(t,X_t)\Big)+\gamma\,g(t)\,\sqrt h\;\xi .$$
- *Heun*: the Heun step of §2.1 on the drift, plus the same noise increment $\gamma\,g(t)\sqrt h\,\xi$.
- *Ancestral*: estimate $\hat x_0=\hat\eta_0(t,X_t)$, from a score by Tweedie, $(X_t+\sigma_t^2\,\hat s)/a_t$, then sample the exact Gaussian posterior $p(x_{t-h}\mid x_t,\hat x_0)$. With $r=a_t/a_{t-h}$ and $c^2=\sigma_t^2-r^2\sigma_{t-h}^2$:
  $$X_{t-h}=\frac{r\,\sigma_{t-h}^2}{\sigma_t^2}\,X_t+\frac{a_{t-h}\,c^2}{\sigma_t^2}\,\hat x_0+\sqrt{\frac{\sigma_{t-h}^2\,c^2}{\sigma_t^2}}\;\xi .$$
  This is the DDPM step on VP [9] and the NCSN step on VE [10]. It is intrinsically stochastic and has no $\gamma$.

### 2.3 Time grid

Sampling runs on a fixed grid of times $t_{\max}=t_0>t_1>\dots>t_N=t_{\min}$, from the prior near $t=1$ to the data near $t=0$, with step sizes $h_i=t_i-t_{i+1}$. The ends can be singular, so usually $t_{\max}\le1$ and $t_{\min}>0$, and a final step returns the denoiser $\hat\eta_0(t_{\min},X)$ instead of the iterate. Common grids:

- **Uniform in $t$**: $t_i=t_{\max}-\tfrac iN\,(t_{\max}-t_{\min})$.
- **Quadratic in $t$**: denser steps near the data, as in DDIM [8].
- **Geometric in $\sigma$**: $\sigma(t_i)$ evenly spaced in $\log\sigma$, as in NCSN [10].
- **Uniform in $\lambda$**: evenly spaced log-SNR, as in DPM-Solver [11].
- **EDM**: $\sigma_i=\big(\sigma_{\max}^{1/\rho}+\tfrac iN(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho})\big)^\rho$ with $\rho=7$, denser at small noise levels [6].

The grids given in $\sigma$ or $\lambda$ are mapped to times by inverting the schedule.

---

## 3 Constraints and guidance

Optimization and sampling can both be steered. Constraints act on the iterate between steps; guidance changes the field itself.

### 3.1 Constraints

A constraint restricts the iterate directly. To keep part of a structure fixed, e.g. a known scaffold, its coordinates are overwritten with the reference before and after every step. The model then always sees the clean fixed part, and the run ends on it.

### 3.2 Guidance

Guidance adds a weighted term to the field. It therefore acts at every evaluation of the field, also inside steps with several evaluations such as Heun or L-BFGS, and several terms simply add up.

- **On a force** (optimization): $F\leftarrow F+\omega\,G$ with a force-like term $G$, e.g. the restraint force $G=-\nabla E_{\mathrm r}$ of a restraint energy $E_{\mathrm r}$ that holds selected distances near target values.
- **On a score** (sampling): $s\leftarrow s+\omega\,\nabla_x\log p_t(y\mid x)$, which steers toward structures with a property $y$ (classifier guidance) [5, 12]. The classifier $p_t(y\mid x)$ has to be trained on noisy structures. A restraint enters the same way, as $-\omega\nabla E_{\mathrm r}$. The velocity and the denoiser follow from the guided score through the conversions in [[parametrizations#6 Score|parametrizations §6]].

---

## References

1. Hessmann, Kahouli, Gugler, Plainer, Noé, Müller, Gebauer (2026). *Generative Pseudo-Force Fields for Molecular Generation.* arXiv:2605.19050.
2. Liu, Nocedal (1989). *On the Limited Memory BFGS Method for Large Scale Optimization.* Mathematical Programming 45, 503–528.
3. Lipman, Chen, Ben-Hamu, Nickel, Le (2023). *Flow Matching for Generative Modeling.* ICLR. arXiv:2210.02747.
4. Albergo, Boffi, Vanden-Eijnden (2023). *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.* arXiv:2303.08797.
5. Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR. arXiv:2011.13456.
6. Karras, Aittala, Aila, Laine (2022). *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS. arXiv:2206.00364.
7. Anderson (1982). *Reverse-Time Diffusion Equation Models.* Stochastic Processes and their Applications 12(3), 313–326.
8. Song, Meng, Ermon (2021). *Denoising Diffusion Implicit Models.* ICLR. arXiv:2010.02502.
9. Ho, Jain, Abbeel (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS. arXiv:2006.11239.
10. Song, Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS. arXiv:1907.05600.
11. Lu, Zhou, Bao, Chen, Li, Zhu (2022). *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.* NeurIPS. arXiv:2206.00927.
12. Dhariwal, Nichol (2021). *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS. arXiv:2105.05233.

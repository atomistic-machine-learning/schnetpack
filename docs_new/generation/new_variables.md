# Several variables

The other pages treat one generated quantity $x\in\mathbb R^d$, the positions. A structure can carry several: positions, atom types, a cell. Each is a **variable** with its own noising, parametrization and head, and all of them share one time $t$ and one network. This page describes what changes when there is more than one: how training noises them together, and how sampling steps them together. §4 adds the atom types, the first discrete variable, and §5 the cell, which moves the atoms with it. The notation follows the [[overview#Notation|overview]]; one variable is the special case of everything below.

---

## 1 Variables

A variable $x^{(k)}$ is one block of the batch: the positions per atom, the atom types per atom, or the cell per structure. It brings

- a **noising**, which gives its path $x^{(k)}_t$ and, on an [[noising#2 Interpolant|interpolant]], its prior $\rho^{(k)}_1$ and schedule $(a_k,b_k)$;
- a **parametrization**, which gives the training target of its head and the conversions between fields (see [[parametrizations]]);
- a **loss weight** $w_k(t,x^{(k)}_t)$ per row, e.g. the SNR weight of an $x_0$ loss or a mask that counts only some atoms;
- an optional **decode**: what else in the structure has to change when this variable changes. A cell moves the atoms with it, at fixed fractional coordinates.

The prior of each variable draws only its own block. Its rows are atoms or structures, and a centered prior stays centered per structure, also for the noise of a sampling step (§3.3).

---

## 2 Training

**One time.** Every structure draws one $t$, and every variable is noised at it, $x^{(k)}_t$ from its own schedule. The noise levels $\sigma_k(t)$ are fixed functions of that one $t$, so conditioning the network on $t$ gives it every noise level at once.

**Ordered noising.** The variables are noised in a fixed order, and each one sees the noisy, decoded values of the variables before it. Without a decode the order does not matter: given $t$ the variables are noised independently. With one, it matters. When the cell is noised first and its decode moves the atoms into the noisy cell, the clean positions the position noising sees are $x^{(R)}_0=f_0\,L_t$, the clean fractional coordinates $f_0$ in the noisy cell $L_t$, as in GEODE [1].

**Losses.** Each variable has its own label (its parametrization's target) and weight, and its loss is

$$
\mathcal L_k=\frac1{N_k}\sum_{i}w_k\big(t,x^{(k)}_{t,i}\big)\,\big\lVert \hat y^{(k)}_i-y^{(k)}_i\big\rVert^2 ,
$$

summed over the components of a row and averaged over the $N_k$ rows. A parametrization with a loss of its own, such as a cross-entropy over atom types, uses that instead. The training loss is $\sum_k\lambda_k\mathcal L_k$ with factors chosen by the training script. For the positions alone this is three times the mean squared error over coordinates, since it sums over $x$, $y$ and $z$.

---

## 3 Sampling

### 3.1 Updates

A sampler steps all variables along one time grid with one model call per stage. Each variable is stepped by one **update**, and the updates can differ between variables:

- *SDE update*: the reverse SDE of [[generation#2.2 SDE sampling|generation §2.2]] on the variable's score, with its own $\gamma$, from the probability-flow ODE ($\gamma=0$) to the reverse-time SDE ($\gamma=1$). Needs a Gaussian interpolant.
- *ODE update*: $dx=\hat v\,dt$ on the variable's velocity, [[generation#2.1 ODE sampling|generation §2.1]]. On a Gaussian interpolant it equals the SDE update at $\gamma=0$.
- *Ancestral update*: the noising's own reverse transition given $\hat x_0$. On a Gaussian interpolant that is the posterior step of [[generation#2.2 SDE sampling|generation §2.2]], the DDPM step on VP and the NCSN step on VE. A masked variable such as atom types is unmasked instead (§4.4); it has no score and no velocity, so this is its only update.

SDE and ODE updates take an integrator (Euler or Heun), and the integrator sets how many model calls, or **stages**, the update needs per step. An ancestral update has one stage, at $t$.

### 3.2 One step

A step from $t$ to $t'=t-h$ runs every stage in turn. At each stage, the model is evaluated once on the current stage point, at the stage time. Then every update that has this stage reads its field from the outputs and moves its variable to its next stage point. After the last stage, the new values of all variables are written in a fixed order, starting from the state at $t$, and each variable's decode runs after it is written.

With Heun on the positions and an ancestral update on the types:

1. model call at $t$; positions: drift $d_0$, predictor $\tilde x=x_t-h\,d_0$ at $t'$; types: unmask, now at $t'$;
2. model call at $t'$ on $(\tilde x, z_{t'})$; positions: drift $d_1$, then $x_{t'}=x_t-\tfrac h2(d_0+d_1)$ plus noise;
3. write the positions, then the types.

### 3.3 Rules

**One time per stage.** The model sees one $t$, so all updates that have a stage $k$ must place it at the same time. An update that has finished its stages already sits at $t'$, so any later stage must be at $t'$ too. Euler (stage at $t$) and Heun (stages at $t$ and $t'$) always agree.

**An update draws once.** An update draws its new value, noise included, at its own last stage. Later model calls in the same step see that value at $t'$, so the second Heun call above sees the types already unmasked. Writing the step at the end only stores what was drawn.

**The noise is the variable's.** The noise of an SDE or ancestral step is drawn by the variable's prior, as unit noise in the prior's subspace. For centered positions it is centered per structure, so the noise never moves the center of a molecule. An integrator adds no noise of its own.

**Every variable once.** Each variable has exactly one update, and the update has to fit the noising: SDE updates only for Gaussian interpolants, ODE updates for interpolants, ancestral updates for Gaussian interpolants and masked variables.

### 3.4 One variable

With one variable and one update this is sampling as in [[generation#2 Sampling|generation §2]]: an SDE update is Euler–Maruyama or Heun on the reverse SDE with its $\gamma$, an ODE update Euler or Heun on the velocity, an ancestral update the posterior step. The only difference is the noise of a centered variable, which is drawn in its subspace, so the noise never moves the center of a molecule. The field itself can: a network whose predicted noise has a nonzero mean per molecule still shifts the center.

### 3.5 Order of accuracy

Heun is second order for the variables it steps only when every variable is smooth in $t$. An ancestral or masked variable changes once per step, from the first call, so a step that mixes them is first order: the variables are advanced one after the other within the step (operator splitting).

### 3.6 Guidance

Guidance ([[generation#3.2 Guidance|generation §3.2]]) acts on the positions. With several variables the calculator adds it to the fields it is named for, typically the positions' score, at every model call, so also at every stage of a step. The other variables see it only through the positions.

---

## 4 Atom types

Atom types are the first discrete variable. Each atom carries a class $z\in\{m,1,\dots,K\}$: its atomic number, or the mask state $m$. The batch stores the mask as $Z=0$, since no element has $Z=0$. Types are noised by masking, an absorbing-state discrete diffusion [2] as in GEODE [1], and are generated next to the positions at one shared $t$ (§2). Everything in §1–§3 holds for them; this section adds what is specific.

### 4.1 Masking

Each atom keeps its type with probability $a(t)$ and is masked otherwise, independently of the other atoms:

$$
z_t=\begin{cases}z_0&\text{with probability }a(t),\\ m&\text{with probability }1-a(t),\end{cases}\qquad a(t)=1-t .
$$

$a$ is the linear schedule of [[schedules#1.1 Linear|schedules §1.1]], read as a keep probability. At $t=1$ every atom is masked. That is the prior and the start of sampling; there is no other noise. The mask is absorbing: an atom masked at $s$ stays masked at every $t>s$, and between them a kept atom is masked with probability $1-a(t)/a(s)$. A structure at $t$ has on average a fraction $1-a(t)=t$ of its atoms masked.

Masking has a schedule and a prior, but it does not interpolate between $x_0$ and $x_1$. It is a noising process of the first level, [[noising#1 Noising process|noising §1]]: its only field is the denoiser (§4.2), and only the ancestral update can step it (§3.3).

### 4.2 Denoiser

For the one-hot encoding $e_z$ of a type, the denoiser of [[noising#1 Noising process|noising §1]] is a distribution,

$$
\eta_0=\mathbb E\big[e_{z_0}\mid\text{noisy structure}\big]=p\big(z_0\mid\text{noisy structure}\big),
$$

the probability of each clean type given everything at $t$: positions, revealed types, masks and $t$ itself. For a revealed atom it is one-hot at $z_t$, since a revealed type is the clean one. For a masked atom the network predicts it. The head gives $K+1$ logits per atom, one per class index $Z$, and the denoiser is their softmax with the mask's logit set to $-\infty$, since no clean type is masked. The loss (§4.3) and the sampler (§4.4) both use this distribution.

The ancestral update reads the denoiser of every variable it steps. For a Gaussian variable that is the mean $\hat x_0$; for types it is the class distribution.

### 4.3 Loss

The loss is the variational bound of the reverse step [2]. For a step from $t$ to $s<t$, let

$$
r(t,s)=\frac{a(s)-a(t)}{1-a(t)}
$$

be the probability that an atom masked at $t$ is revealed at $s$. Given the clean type, the true reverse step $q(z_s\mid z_t=m,z_0)$ reveals $z_0$ with probability $r$ and keeps the mask otherwise. The model's step reveals with the same probability and draws the type from the denoiser $\hat p$. The two steps differ only in the revealed type, so

$$
\mathrm{KL}\big(q\,\Vert\,p_\theta\big)=r\,\big(-\log\hat p(z_0)\big)=r\,\mathrm{CE}(\hat p,z_0).
$$

A revealed atom keeps its type in both steps, and its KL is 0. For a small step $h$, $r\approx h\,(-\dot a)/(1-a)$, which is $h/t$ for $a=1-t$. With $T$ steps, $h=1/T$, so the loss weight per atom is

$$
w_Z(t,z_t)=\begin{cases}\dfrac1{T\,t}+\lambda_{\mathrm{ce}}&z_t=m,\\[6pt]0&z_t\ne m,\end{cases}
$$

with $\lambda_{\mathrm{ce}}=0.005$, as in GEODE's Eq. 16 [1]. The bound term becomes small at large $t$, where $1/(Tt)\to1/T$, and the plain cross-entropy keeps a signal there. The $1/t$ is the weight of the continuous-time bound of masked diffusion [3, 4]. The $1/T$ is a constant, so the training script's factor $\lambda_Z$ absorbs it.

**Normalization.** The weighted cross-entropy is summed over the masked atoms of the batch and divided by their number, so every masked atom counts the same, as in the bound. GEODE's Eq. 16 divides per structure, by $N_m$ inside the expectation over $t$. That gives a structure with few masked atoms (small $t$) as much weight as one with many.

The positions' loss is unchanged and counts every atom, masked or not. The training loss is $\lambda_R\mathcal L_R+\lambda_Z\mathcal L_Z$. GEODE's factors give $\lambda_R=10$ (its $20\cdot\tfrac12$) and $\lambda_Z=150$. They were tuned on MP-20, so on QM9 both terms are logged on a validation split and the ratio is adjusted from there.

### 4.4 Unmasking

The ancestral update of a masked variable is the model's reverse step of §4.3. From $t$ to $s$, each atom masked at $t$ is revealed with probability $r(t,s)=(t-s)/t$, and its type is drawn from $\hat p$; the other masked atoms stay masked. Revealed types are kept: nothing is masked again, and the step adds no noise. Over a grid the probabilities telescope, $\prod_k\big(1-r(t_k,t_{k+1})\big)=\prod_k t_{k+1}/t_k$, so an atom is still masked at $s$ with probability $s$, as in training.

**The last step reveals every atom.** The types' time range ends at $t_{\min}=1/T$, where the weight $1/(Tt)$ reaches 1; it is unbounded below. A grid that ends at $t_{\min}$ would leave a fraction $t_{\min}$ of the atoms masked. So the step that reaches $t_{\min}$ reveals every remaining atom, $r=1$. That is GEODE's last step from $1/T$ to $0$, where $a=1$. The grid therefore has to reach the types' $t_{\min}$. The default grid ends at the largest $t_{\min}$ of all schedules, which is the types' $1/T$, and the sampler raises for a grid that ends above it.

**Excluded elements.** Elements that should never be generated get probability zero, and $\hat p$ is renormalized over the rest. On QM9 that is every element except H, C, N, O and F; on MP-20 it is GEODE's 24 [1]. Training keeps all $K$ classes.

**Atoms revealed together are drawn independently.** Each draws from its own marginal given the structure at $t$. That is exact only for small steps: two atoms revealed in one step can get types that are each likely but unlikely together. A step reveals on average a fraction $h/t$ of the masked atoms, so about $N h$ atoms. With $T=1000$ steps and at most 29 atoms in a QM9 molecule, that is mostly one atom or none. Sampling with few steps reveals many atoms at once.

**Next to the positions.** Types have one stage, at $t$. With Heun on the positions, the second model call already sees the types revealed at $t'$ (§3.2, §3.3), and the joint step is first order (§3.5).

### 4.5 Start

Sampling starts at $t=1$ with every atom masked and the positions drawn from their prior. The number of atoms is not generated: each structure draws it from the training set's histogram, as in EDM [5]. Training at $t=1$ gives the same state. $a(1)=0$ masks every atom, and the positions are close to their prior when $\sigma_{\max}$ is much wider than the molecule ([[schedules#2.2 Variance exploding (VE)|schedules §2.2]]).

The mask is a class of its own. The network's type embedding has $K+1$ rows, with row $0$ for the mask, and its head predicts $K+1$ logits per atom, one per class index.

### 4.6 When types are decided

Both variables run on one $t$, so their schedules fix when the types are decided relative to the geometry. Take linear masking with the QM9 VE positions ($\sigma_{\min}=0.01$ Å, $\sigma_{\max}=10$ Å). At $t=0.5$ half the types are still masked, while $\sigma=0.32$ Å and the geometry has mostly formed. The network places the atoms first and names most of them afterwards, from the local geometry. A keep probability that stays near 1 for longer, such as $a=1-t^2$, would decide the types earlier; the weight of §4.3 follows from $a$. GEODE's linear masking comes first, and this stays a knob for later.

---

## 5 Cell

A crystal has a cell, and its positions are periodic. The cell is the first variable with one row per structure, and the only one whose decode moves another variable. This section follows the diffusion part of GEODE [1] without its symmetry, in space group P1: its template, its symmetrized noise and fields, and its orbit-shared types all reduce to the identity there. The lattice vectors are the rows of the cell $L$, so a position is $x=f\,L$ with fractional coordinates $f\in[0,1)^3$.

### 5.1 Cell parameters

The cell is generated as its six parameters $\ell=(a,b,c,\alpha,\beta,\gamma)$, three lengths and three angles. A cell matrix has nine entries, and three of them only rotate the crystal. Parameters are rotation invariant, and every set of them gives a valid cell, as long as the angles are possible for a cell.

**Reduce.** Before noising, the cell of each training structure is Niggli-reduced [6], once. That gives a unique, compact basis of the same lattice, and the atoms stay where they are. This is a data transform of its own, not part of the cell variable. The orientation does not matter, since the decode builds every cell in standard orientation.

**Whitening.** The cell variable is the whitened parameter vector

$$
y=\frac{\ell-\mu(N)}{s},\qquad \mu(N)=\big(\bar\ell N^{1/3}\cdot\mathbf 1_3,\ \bar\theta\cdot\mathbf 1_3\big),\qquad s^2=\tfrac1{64}\big(v_{\mathrm{len}}\cdot\mathbf 1_3,\ v_{\mathrm{ang}}\cdot\mathbf 1_3\big),
$$

with the mean and variance of the lengths over $N^{1/3}$ and of the angles in the training set. GEODE's MP-20 values are $\bar\ell=2.94$ Å, $\bar\theta=99.7°$, $v_{\mathrm{len}}=2.52$ Å² and $v_{\mathrm{ang}}=297$ deg² [1, App. D.1]. The mean length grows with $N^{1/3}$, so the prior fixes the volume per atom rather than the cell's size. The factor $1/64$ narrows the prior to 0.2 Å and 2.2°, so a noisy cell never has a length near zero or a degenerate angle. In P1 all six parameters are free; GEODE fixes some by crystal system.

**Decode.** After every change of $y$, its cell $L(\mu+s\,y)$ in standard orientation replaces the old one, with $a$ along $x$ and $b$ in the $xy$-plane, and every atom keeps its fractional coordinates. That is the cell variable's decode (§1). It runs in training, where it moves the clean atoms into the noisy cell, and after every sampling step of the cell. Parameters do not record handedness, so a left-handed cell is replaced by a right-handed one and the crystal by its mirror image. A mirror image is a valid crystal, of the opposite chirality.

### 5.2 Noising

In $y$, GEODE's lattice process [1, Eq. 10] is an ordinary variance-preserving interpolant with a unit prior,

$$
y_t=a(t)\,y_0+\sigma(t)\,\varepsilon,\qquad a(t)=\frac{\cos\big(\tfrac\pi2\,\tfrac{t+\kappa}{1+\kappa}\big)}{\cos\big(\tfrac\pi2\,\tfrac{\kappa}{1+\kappa}\big)},\qquad \sigma=\sqrt{1-a^2},
$$

the cosine schedule [7] with $\kappa=0.008$. $a(1)=0$, so at $t=1$ the cell is drawn from the prior, $\ell\sim\mathcal N(\mu(N),\operatorname{diag}s^2)$. At $t=1/1000$, $1-a^2=4.1\cdot10^{-5}$, GEODE's first step. Training data sit 8 to 16 prior widths from $\mu$, and the head has to cover that distance.

### 5.3 Head and loss

**Deformation head.** The head deforms the noisy cell into the clean one [1, Eq. 5],

$$
\hat L_0=L_t\,(I-S)^{\top},
$$

with a symmetric $3\times3$ tensor $S$ per structure, and returns the whitened parameters $\hat y_0$ of $\hat L_0$. Under a rotation $Q$ of the crystal, $L_t\to L_tQ^\top$ and $S\to QSQ^\top$, so $\hat L_0\to\hat L_0Q^\top$ and $\hat y_0$ is unchanged. $S$ transforms like a stress tensor. GEODE reads it out of $l=2$ features. PaiNN has only scalars $s_i$ and vectors, so $S$ is the mean over atoms of $s_i I+\tfrac12(u_iv_i^\top+v_iu_i^\top)$, from two vectors $u_i,v_i$ of each atom.

**Loss.** GEODE regresses the noise implied by $\hat y_0$, $\hat\varepsilon=(y_t-a\hat y_0)/\sigma$ [1, Eq. 8, 9]. Since

$$
\lVert\hat\varepsilon-\varepsilon\rVert^2=\frac{a^2}{\sigma^2}\,\lVert\hat y_0-y_0\rVert^2,
$$

that is an $x_0$ loss with the SNR weight $w_L=a^2/\sigma^2$ per structure (§2). The head stays an $x_0$ head, and the sampler reads $\hat y_0$ directly. The training loss gains the term $\lambda_L\mathcal L_L$ with GEODE's $\lambda_L=0.9$.

### 5.4 Periodic positions

**Noising.** The cell is noised first, so its decode has placed the clean fractional coordinates $f_0$ in the noisy cell. The positions are then noised in that cell on the VE schedule of the molecules, $\sigma$ from 0.01 to 10 Å, and wrapped back into it [1, Eq. 13]:

$$
x_t=\operatorname{wrap}_{L_t}\big(f_0L_t+\sigma(t)\,\varepsilon\big).
$$

The order of the variables is the cell, the types, then the positions. The types do not interact with the others in training, so their place does not matter.

**Target.** A wrapped $x_t$ can be reached from $x_0$ through any lattice image. With $d$ the shortest displacement from $x_0$ to $x_t$, the expected noise given both is

$$
\varepsilon^\star=\frac1\sigma\sum_{n\in\mathbb Z^3}w_n\,(d+nL_t),\qquad w_n\propto\exp\Big(-\frac{\lVert d+nL_t\rVert^2}{2\sigma^2}\Big),
$$

GEODE's wrapped-normal target [1, Eq. 6] up to its sign, $\varepsilon^\star=-\sigma s^\star$. It is the noise target of the interpolant with the endpoint $x_1$ replaced by its expectation over the images, $\sigma_{\max}\,\varepsilon^\star$. Every field is linear in $x_1$, so the eps head, its score and its $\hat x_0$ all become the wrapped ones, and the parametrization stays as it is.

**Image box.** The sum runs over $n\in\{-3,\dots,3\}^3$, 343 images. The box has to grow with $\sigma$: on a 6–7 Å triclinic cell at $\sigma=10$ Å, the rms error of the target against a brute-force sum is 0.13 with $\pm2$ images, 0.05 with $\pm3$ and 0.02 with $\pm4$. For scale, the target's rms is 1.7 at small $\sigma$ and close to 0 above $\sigma\approx5$ Å. The error mostly adds variance, since its mean is close to zero. Above about the cell height, a sum over the reciprocal lattice converges in a few terms.

**No center.** A crystal has no origin, so neither the positions nor their noise are centered. A shift of every atom by the same vector gives the same crystal, which the network cannot see; the mean of the noise over a structure is a part of the loss no head can fit.

### 5.5 Sampling

**Start.** $N$ is drawn from the training set's histogram, $y$ from $\mathcal N(0,I)$ and decoded into a cell, every type is masked, and the positions are uniform in the cell. At $\sigma_{\max}=10$ Å the wrapped Gaussian is flat on every MP-20 cell, so this is the $t=1$ marginal of training. The cell is drawn first, since the positions are drawn in it.

**Updates.** GEODE's Algorithm 1 [1, App. B] is three ancestral updates, in the order positions, cell, types:

- positions: the VE posterior mean $x_t+(\sigma_t^2-\sigma_s^2)\,\hat s$, Algorithm 1's line 7, plus noise of std $\sqrt{\sigma_t^2-\sigma_s^2}$ (line 16), wrapped into the cell. That noise exceeds the posterior's by $\sigma_t/\sigma_s$, 1.007 on 1000 geometric steps;
- cell: the VP posterior given $\hat y_0$ (lines 8, 9 and 15); the atoms follow at fixed fractional coordinates;
- types: unmasking, §4.4 (lines 10 to 13).

**Noise order.** Each update draws its whole step, and the new values are written in list order (§3.2): the positions first, then the cell, whose decode carries them along. The position noise is therefore drawn in the old cell and moved with the cell, where Algorithm 1 adds it after the cell has moved. The difference is the noise deformed by one cell step, a relative change of the size of that step.

**Heun with a moving cell.** With Heun on the positions, the second drift is taken in the cell at $t'$ but added in the cell at $t$. Unless it is mapped back to the start cell, the step is first order. Ancestral types make the joint step first order anyway (§3.5), so this matters only for smooth updates on positions and cell without types.

**Neighbor list.** The cell changes at every step, so every structure's neighbor list is rebuilt at every step.

**End.** The last positions still carry $\sigma(t_{\min})\approx0.01$ Å of noise, and the last cell $\sigma(t_{\min})\approx0.006$ prior widths. Reading the model's $\hat x_0$ at the end removes the first.

### 5.6 Left out

Besides the symmetry, GEODE remaps equivalent atoms to the closest reference sites before it forms the target, and conditions the types on their last prediction (self-conditioning). Neither is part of this section: without remapping each atom is pulled back to its own site, and the types see only the noisy structure.

---

## References

1. Lou, Ganose (2026). *Symmetry-Preserving Cartesian Diffusion for Crystal Generation.* arXiv:2610.01898.
2. Austin, Johnson, Ho, Tarlow, van den Berg (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS. arXiv:2107.03006.
3. Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush, Kuleshov (2024). *Simple and Effective Masked Diffusion Language Models.* NeurIPS. arXiv:2406.07524.
4. Shi, Han, Wang, Doucet, Titsias (2024). *Simplified and Generalized Masked Diffusion for Discrete Data.* NeurIPS. arXiv:2406.04329.
5. Hoogeboom, Satorras, Vignac, Welling (2022). *Equivariant Diffusion for Molecule Generation in 3D.* ICML. arXiv:2203.17003.
6. Křivý, Gruber (1976). *A unified algorithm for determining the reduced (Niggli) cell.* Acta Cryst. A32, 297–298.
7. Nichol, Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML. arXiv:2102.09672.

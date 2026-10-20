# Several variables

The other pages treat one generated quantity $x\in\mathbb R^d$, the positions. A structure can carry several: positions, atom types, a cell. Each is a **variable** with its own noising, parametrization and head, and all of them share one time $t$ and one network. This page describes what changes when there is more than one: how training noises them together, and how sampling steps them together. The notation follows the [[overview#Notation|overview]]; one variable is the special case of everything below.

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
- *Ancestral update*: the noising's own reverse transition given $\hat x_0$. On a Gaussian interpolant that is the posterior step of [[generation#2.2 SDE sampling|generation §2.2]], the DDPM step on VP and the NCSN step on VE. A masked variable such as atom types is unmasked instead; it has no score and no velocity, so this is its only update.

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

**Every variable once.** Each variable has exactly one update, and the update has to fit the noising: SDE and ODE updates only for Gaussian interpolants, ancestral updates for everything.

### 3.4 One variable

With one variable and one update this is sampling as in [[generation#2 Sampling|generation §2]]: an SDE update is Euler–Maruyama or Heun on the reverse SDE with its $\gamma$, an ODE update Euler or Heun on the velocity, an ancestral update the posterior step. The only difference is the noise of a centered variable, which is drawn in its subspace, so the noise never moves the center of a molecule. The field itself can: a network whose predicted noise has a nonzero mean per molecule still shifts the center.

### 3.5 Order of accuracy

Heun is second order for the variables it steps only when every variable is smooth in $t$. An ancestral or masked variable changes once per step, from the first call, so a step that mixes them is first order: the variables are advanced one after the other within the step (operator splitting).

### 3.6 Guidance

Guidance ([[generation#3.2 Guidance|generation §3.2]]) acts on the positions. With several variables the calculator adds it to the fields it is named for, typically the positions' score, at every model call, so also at every stage of a step. The other variables see it only through the positions.

---

## References

1. Lou, Ganose (2026). *Symmetry-Preserving Cartesian Diffusion for Crystal Generation.* arXiv:2610.01898.

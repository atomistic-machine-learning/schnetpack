"""
Experimental: lattice and atom-type diffusion, on top of the generative axes.

Kept apart from the rest of the subpackage because these are unsettled — the
representations here are research choices, not the small set of things every
diffusion model agrees on. Nothing else imports this.

Three things, split by whether they are Gaussian:

  MatterGenLatticePath(VPPath)
      Continuous Gaussian -> IS a Path. Standard VP on the density-centered
      lattice Ltilde = L - Lbar(N). Inherits interpolate / diffuse /
      marginal_prob / f / g2 unchanged; adds only centering. Its prior is a
      Prior, not a path method — see :class:`CenteredLatticePrior`.

  TypeEncoding + JointDiffuse
      Types relaxed into a continuous vector (one-hot or analog bits), which
      makes them *just another diffused property*. Then the entire Gaussian
      stack applies unchanged: every parametrization, churn, both samplers.
      :class:`JointDiffuse` is the one genuinely new piece — it noises several
      properties at a single shared time, which the library's `Diffuse` cannot
      do because two instances would draw two independent times.

  D3PMTypeDiffusion
      Categorical -> NOT a Path (no Gaussian interpolate, no score). Composes a
      VPPath for the *schedule only*: retain probability = alpha(t)^2, so types
      obey the same alpha^2 + sigma^2 = 1 timing as everything else. It cannot
      use the Gaussian samplers at all, which is exactly the cost the continuous
      relaxation buys its way out of.

Naming note: Path.gamma() is the bridge-noise coefficient in this codebase.
The D3PM retain probability is therefore called `keep_prob`, never gamma.
"""

from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from schnetpack import properties
from schnetpack.generative.couplings import Coupling, IndependentCoupling
from schnetpack.generative.parametrizations import Parametrization
from schnetpack.generative.paths import Path, VPPath
from schnetpack.generative.priors import Prior
from schnetpack.transform.base import Transform

__all__ = [
    "MatterGenLatticePath",
    "CenteredLatticePrior",
    "TypeEncoding",
    "OneHotEncoding",
    "AnalogBitsEncoding",
    "EncodeTypes",
    "PadGhosts",
    "DiffuseField",
    "JointDiffuse",
    "D3PMTypeDiffusion",
]


# =========================================================================== #
# 1. Lattice: VP on the density-centered lattice
# =========================================================================== #
class MatterGenLatticePath(VPPath):
    """
    VP diffusion on Ltilde = L - Lbar(N), with a density-sized mean lattice

        Lbar(N) = (N * v0)^(1/3) I_3,   v0 = mean cell volume per atom.

    The t_max prior on the physical lattice is N(Lbar(N), I): a random cell at
    the right density for N atoms. All machinery (interpolate, diffuse, f, g2,
    marginal_prob) is inherited and operates on the *centered* variable, which
    keeps the forward drift f(t) * Ltilde correct: the mean shift is constant
    in t and must not be dragged toward zero by the drift.

    Training::

        Lt = path.center(L0, n_atoms)
        L_t, noise = path.diffuse(Lt, t)          # inherited, unchanged

    Sampling::

        x = CenteredLatticePrior(path).sample(...)   # ~ N(0, I), centered frame
        x = sampler.sample(model, ...)               # reverse in centered frame
        L = path.uncenter(x, n_atoms)                # back to physical lattice

    Caveats vs. the MatterGen reference implementation:
      * Their code expresses the same idea via a limiting distribution with
        N-dependent scale; mean-shift vs. rescaling place the N-dependence
        differently. Check the Zeni et al. code if you need bit-level agreement.
      * Centering does not fix the O(3) / lattice-vector gauge; use it on a
        rotation-reduced (e.g. symmetric / Niggli) representation.
    """

    def __init__(self, vol_per_atom: float = 20.0, **vp_kwargs):
        """
        Args:
            vol_per_atom: mean cell volume per atom in Angstrom^3; fit it from
                your dataset rather than trusting the default
        """
        super().__init__(**vp_kwargs)
        self.v0 = vol_per_atom

    def mean_lattice(self, n_atoms: torch.Tensor, device=None) -> torch.Tensor:
        """(B,) atom counts -> (B, 3, 3) isotropic mean lattices."""
        device = device or n_atoms.device
        edge = (n_atoms.to(torch.float32) * self.v0).pow(1.0 / 3.0)
        eye = torch.eye(3, device=device).expand(n_atoms.shape[0], 3, 3)
        return edge.view(-1, 1, 1) * eye

    def center(self, L: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
        return L - self.mean_lattice(n_atoms, L.device)

    def uncenter(self, Lt: torch.Tensor, n_atoms: torch.Tensor) -> torch.Tensor:
        return Lt + self.mean_lattice(n_atoms, Lt.device)


class CenteredLatticePrior(Prior):
    """
    The t_max prior of :class:`MatterGenLatticePath`, in the centered frame.

    Standard normal, because the whole point of centering is that the physical
    prior N(Lbar(N), I) becomes N(0, I) once the density-sized mean is removed.
    A Prior rather than a path method: a path is geometry, and where you start
    sampling is a distribution — see :mod:`schnetpack.generative.priors`.
    """

    def __init__(self, path: MatterGenLatticePath):
        self.path = path

    def sample(self, shape: Sequence[int], dtype=None, device=None) -> torch.Tensor:
        return torch.randn(*shape, dtype=dtype, device=device)


# =========================================================================== #
# 2. Atom types, continuous relaxation: embed -> any Path -> decode
# =========================================================================== #
class TypeEncoding:
    """
    Maps types {0..K-1} <-> a continuous vector the Gaussian machinery can diffuse.

    This is the whole trick of continuous type diffusion: once types are a
    vector, they are not a special case of anything. They are a property, and
    :class:`JointDiffuse` noises them with the same path, parametrization and
    sampler as the coordinates.
    """

    dim: int

    def encode(self, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class OneHotEncoding(TypeEncoding):
    """
    DiffCSP / molecular-EDM style: z0 = scale * onehot(a), decode by argmax.

    ``scale`` sets how far apart the classes sit relative to unit noise, and so
    at which sigma the type information is destroyed. With scale = 1 the classes
    are sqrt(2) apart, so they blur once sigma approaches 1 — much earlier than
    coordinates spread over several Angstrom. If types and positions share a
    schedule, that mismatch is real: raise the scale to keep them legible for
    longer.

    Chemically uninformative by construction — every element is equidistant from
    every other, so the model cannot know carbon is more like nitrogen than like
    bromine.
    """

    def __init__(self, K: int, scale: float = 1.0):
        self.K, self.dim, self.scale = K, K, scale

    def encode(self, a):
        return self.scale * F.one_hot(a, self.K).float()

    def decode(self, z):
        return z.argmax(dim=-1)


class AnalogBitsEncoding(TypeEncoding):
    """
    FlowMM / Chen et al. style: a -> {-1,+1}^ceil(log2 K), decode by sign.

    ~7 dims for the periodic table instead of ~100. Unused bit patterns exist
    when K is not a power of two, and get clamped to the nearest valid class.
    """

    def __init__(self, K: int, scale: float = 1.0):
        self.K = K
        self.dim = max(1, int(torch.tensor(K - 1).log2().floor()) + 1)
        self.scale = scale
        ar = torch.arange(K)
        bits = (ar.unsqueeze(-1) >> torch.arange(self.dim)) & 1
        self.codebook = 2.0 * bits.float() - 1.0

    def encode(self, a):
        return self.scale * self.codebook.to(a.device)[a]

    def decode(self, z):
        sign_bits = (z > 0).long()
        idx = (sign_bits * (2 ** torch.arange(self.dim, device=z.device))).sum(-1)
        return idx.clamp_(max=self.K - 1)


class EncodeTypes(Transform):
    """
    Write the continuous encoding of the atom types into the batch.

    Runs before :class:`JointDiffuse`, which then treats the result as an
    ordinary property. Note it encodes ``Z`` through a caller-supplied index map
    rather than the atomic number itself: one-hot over the periodic table would
    be 100 mostly-empty channels, and QM9 uses five elements.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        encoding: TypeEncoding,
        z_to_index: Dict[int, int],
        output_key: str = "type_vec",
        index_key: str = "type_idx",
    ):
        super().__init__()
        self.encoding = encoding
        self.output_key = output_key
        self.index_key = index_key
        lookup = torch.zeros(int(max(z_to_index)) + 1, dtype=torch.long)
        for z, i in z_to_index.items():
            lookup[z] = i
        self.register_buffer("lookup", lookup)

    def forward(self, inputs):
        idx = self.lookup[inputs[properties.Z]]
        inputs[self.index_key] = idx
        inputs[self.output_key] = self.encoding.encode(idx)
        return inputs


class PadGhosts(Transform):
    """
    Pad every structure to a fixed atom count with "ghost" atoms.

    A diffusion model works on a fixed-size tensor, but molecules are not a
    fixed size. Ghosts resolve that: every structure is padded to n_max with
    atoms of a reserved type, and the *number* of atoms becomes something the
    model predicts — as a type, through the machinery that already exists —
    rather than something it must be told.

    Ghost positions are fresh Gaussian noise. Two consequences worth being
    explicit about:

    - The ghost's position is unpredictable by construction, so it contributes
      an irreducible term to any position loss. Under VP the best possible
      prediction of a ghost's noise is sigma * x_t (since alpha^2 + sigma^2 = 1
      makes the posterior mean exactly that), leaving a floor of alpha(t)^2.
      Expect the position metric to *rise* when ghosts are switched on; it is
      not a regression, and the ghosts' coordinates are discarded anyway.
    - The scale is deliberate. With unit-scale noise a ghost's position is
      already distributed like the VP prior, so diffusing it changes nothing
      distributionally: a ghost is noise at t = 0 and noise at t = 1. That is
      the property that makes them cheap.

    Run before centering — the padded system is what lives in the zero-COM
    subspace, so the mean must be taken over the ghosts too.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, n_max: int, ghost_z: int = 0, position_scale: float = 1.0):
        """
        Args:
            n_max: pad to this many atoms; the largest structure in the dataset
            ghost_z: atomic number reserved for ghosts (0 is ASE's dummy "X")
            position_scale: std of the ghost coordinates; match the prior
        """
        super().__init__()
        self.n_max = n_max
        self.ghost_z = ghost_z
        self.position_scale = position_scale

    def forward(self, inputs):
        R = inputs[properties.R]
        n = R.shape[0]
        pad = self.n_max - n
        if pad < 0:
            raise ValueError(f"structure has {n} atoms, more than n_max={self.n_max}")
        if pad == 0:
            return inputs

        ghosts = self.position_scale * torch.randn(pad, 3, dtype=R.dtype)
        inputs[properties.R] = torch.cat([R, ghosts], dim=0)
        inputs[properties.Z] = torch.cat(
            [inputs[properties.Z], torch.full((pad,), self.ghost_z, dtype=torch.long)]
        )
        # collation builds idx_m from this; it must count the ghosts
        inputs[properties.n_atoms] = torch.tensor([self.n_max])
        return inputs


class DiffuseField:
    """One property to diffuse, and how."""

    def __init__(
        self,
        property_key: str,
        parametrization: Parametrization,
        label_key: str,
        coupling: Optional[Coupling] = None,
    ):
        """
        Args:
            property_key: batch key to noise; overwritten with x_t
            parametrization: what the head predicts for this field, and the path
            label_key: batch key to write the training target to
            coupling: how this field's second endpoint is drawn. Per field on
                purpose — molecular coordinates need COM-free noise, types do
                not, and they must not be forced to share.
        """
        self.property_key = property_key
        self.parametrization = parametrization
        self.label_key = label_key
        self.coupling = coupling if coupling is not None else IndependentCoupling()


class JointDiffuse(Transform):
    """
    Noise several properties of a structure at one shared time.

    The library's :class:`~schnetpack.generative.transforms.Diffuse` draws its
    own time, so two of them would put the coordinates and the types at two
    unrelated points on the schedule — the model would see a clean geometry with
    scrambled elements and have no way to know which. Joint diffusion means one
    time for everything, and that is the only reason this class exists.

    Each field keeps its own parametrization (hence its own path) and its own
    coupling, so the fields stay independent in every respect except when.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        fields: List[DiffuseField],
        t_sampler: Optional[Callable[[int, torch.device], torch.Tensor]] = None,
        time_key: str = "t",
        structure_time_key: Optional[str] = "t_structure",
    ):
        """
        Args:
            fields: the properties to noise
            t_sampler: draws the shared time, (n, device) -> (n,); defaults to
                uniform on the first field's usable range
            time_key: key for the per-atom time, for conditioning
            structure_time_key: key for the per-structure time; None to skip
        """
        super().__init__()
        self.fields = fields
        self.t_sampler = t_sampler if t_sampler is not None else self._uniform_t
        self.time_key = time_key
        self.structure_time_key = structure_time_key

    @property
    def path(self) -> Path:
        """The first field's path — the one the shared time is drawn against."""
        return self.fields[0].parametrization.path

    def _uniform_t(self, n: int, device) -> torch.Tensor:
        span = self.path.t_max - self.path.t_min
        return self.path.t_min + span * torch.rand(n, device=device)

    def forward(self, inputs):
        anchor = inputs[self.fields[0].property_key]
        t = self.t_sampler(1, anchor.device).to(anchor.dtype)
        t_atoms = t.repeat(anchor.shape[0])

        for field in self.fields:
            x0 = inputs[field.property_key]
            x0, x1 = field.coupling.sample(x0)
            path = field.parametrization.path
            inputs[field.property_key] = path.interpolate(x0, x1, t_atoms)
            inputs[field.label_key] = field.parametrization.target(x0, x1, t_atoms)

        inputs[self.time_key] = t_atoms
        if self.structure_time_key is not None:
            inputs[self.structure_time_key] = t
        return inputs


# =========================================================================== #
# 3. Atom types: discrete VP (D3PM toward the dataset marginal)
# =========================================================================== #
class D3PMTypeDiffusion:
    """
    Categorical diffusion on types a in {0..K-1}, sharing a VPPath's schedule.

    Cumulative kernel, with keep_prob(t) = alpha(t)^2::

        q(a_t = y | a_0 = x) = keep_prob * [x == y] + (1 - keep_prob) * m_y

    Keep the true type w.p. alpha^2, else resample from the dataset marginal m.
    At t_max, alpha ~ 0 and a_t ~ m: composition drawn from the data, the
    discrete analogue of the Gaussian prior. The network predicts clean-type
    logits (a denoiser); there is no score, and therefore none of the reverse
    machinery in this package applies. That is the trade against
    :class:`OneHotEncoding` + :class:`JointDiffuse`, which give up the exact
    categorical prior to keep the whole Gaussian stack.
    """

    def __init__(self, schedule: Path, marginal: torch.Tensor):
        """
        Args:
            schedule: a Path (typically VPPath) supplying alpha(t) and
                t_min / t_max; used for timing only
            marginal: (K,) element frequencies from the dataset (normalized here)
        """
        self.schedule = schedule
        self.m = marginal / marginal.sum()
        self.K = int(self.m.shape[0])

    def keep_prob(self, t: torch.Tensor) -> torch.Tensor:
        """
        Retain probability = alpha(t)^2, the discrete signal coefficient.

        Named keep_prob, not gamma: Path.gamma is the bridge-noise coefficient.
        """
        return self.schedule.alpha(t) ** 2

    # -- forward ---------------------------------------------------------- #

    def q_probs(self, a0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """q(a_t | a0) as (B, K) rows."""
        k = self.keep_prob(t).unsqueeze(-1)
        onehot = F.one_hot(a0, self.K).to(k.dtype)
        return k * onehot + (1.0 - k) * self.m.to(a0.device)

    def corrupt(self, a0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample a_t ~ q(a_t | a0). The categorical `diffuse`."""
        k = self.keep_prob(t)
        keep = torch.rand(a0.shape, device=a0.device) < k
        resampled = torch.multinomial(
            self.m.to(a0.device), a0.numel(), replacement=True
        ).reshape(a0.shape)
        return torch.where(keep, a0, resampled)

    def sample_prior(self, shape, device=None) -> torch.Tensor:
        """t_max prior: types drawn from the dataset marginal."""
        flat = torch.multinomial(
            self.m.to(device), int(torch.tensor(shape).prod()), replacement=True
        )
        return flat.reshape(shape)

    # -- training --------------------------------------------------------- #

    def loss(self, model: Callable, a0: torch.Tensor, t: torch.Tensor, cond=None):
        """
        Cross-entropy on clean types — the simplified D3PM objective.

        Swap in the full variational bound if you need likelihoods.
        """
        a_t = self.corrupt(a0, t)
        return F.cross_entropy(model(a_t, t, cond), a0)

    # -- reverse ---------------------------------------------------------- #

    def posterior_probs(
        self,
        a_t: torch.Tensor,
        a0_probs: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        p(a_s | a_t) with predicted a0, for a reverse step t -> s (s < t).

        Uses the closure of this kernel family under composition: cumulative
        keep-probs multiply, so the s->t step has keep k_st = k_t / k_s. Then

            p(a_s = j | a_t) ~ q(a_t | a_s = j) * [k_s a0_probs + (1-k_s) m]_j

        i.e. the unnormalized product with the a0-marginalization folded in,
        normalized at the end.
        """
        m = self.m.to(a0_probs.device)
        k_t = self.keep_prob(t).unsqueeze(-1)
        k_s = self.keep_prob(s).unsqueeze(-1)
        k_st = k_t / k_s

        at_onehot = F.one_hot(a_t, self.K).to(a0_probs.dtype)
        m_at = m[a_t].unsqueeze(-1)

        lik = k_st * at_onehot + (1.0 - k_st) * m_at  # q(a_t | a_s=.)
        pri = k_s * a0_probs + (1.0 - k_s) * m  # q(a_s=. | a0_hat)
        post = lik * pri
        return post / post.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def reverse_step(self, a_t, a0_probs, t, s) -> torch.Tensor:
        return torch.multinomial(self.posterior_probs(a_t, a0_probs, t, s), 1).squeeze(
            -1
        )

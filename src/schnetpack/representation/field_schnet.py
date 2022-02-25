from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.init import zeros_

import schnetpack.properties as structure
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
from schnetpack.representation.schnet import SchNetInteraction
from schnetpack.utils import required_fields_from_properties

from schnetpack import properties
import schnetpack.nn as snn

__all__ = ["FieldSchNet", "NuclearMagneticMomentEmbedding"]


class FieldSchNetFieldInteraction(nn.Module):
    """
    Model interaction of dipole features with external fields (see Ref. [#field2]_).
    Computes the overall update to the scalar features.

    Args:
        external_fields (list(str)): List of external fields
        n_atom_basis (int): Number of atomic features
        activation (Callable): Activation function for internal transformations.

    References:
    .. [#field2] Gastegger, Schütt, Müller:
       Machine learning of solvent effects on molecular spectra and reactions.
       Chemical Science, 12(34), 11473-11483. 2021.
    """

    def __init__(
        self,
        external_fields: List[str],
        n_atom_basis: int,
        activation: Callable = shifted_softplus,
    ):
        super(FieldSchNetFieldInteraction, self).__init__()
        self.f2out = nn.ModuleDict(
            {
                field: Dense(n_atom_basis, n_atom_basis, activation=activation)
                for field in external_fields
            }
        )
        self.external_fields = external_fields

    def forward(
        self, mu: Dict[str, torch.Tensor], external_fields: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the update based on the fields.

        Args:
            mu (dict(str, torch.Tensor): Model dipole features.
            external_fields (dict(str, torch.Tensor): External fields

        Returns:
            torch.Tensor: Field update of scalar features.
        """
        dq = 0.0

        for field in self.external_fields:
            v = torch.sum(mu[field] * external_fields[field], dim=1, keepdim=True)
            v = self.f2out[field](v)
            dq = dq + v

        return dq


class DipoleUpdate(nn.Module):
    """
    Update the dipole moment features based on the scalar representations on the neighbor atoms.

    Args:
        external_fields list(str): List of external fields.
        n_atom_basis (int): Number of atomic features.
    """

    def __init__(self, external_fields: List[str], n_atom_basis: int):
        super(DipoleUpdate, self).__init__()
        self.external_fields = external_fields

        # zero init is important here, otherwise updates grow uncontrollably
        self.transform = nn.ModuleDict(
            {
                field: Dense(
                    n_atom_basis,
                    n_atom_basis,
                    activation=None,
                    bias=False,
                )
                for field in external_fields
            }
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: Dict[str, torch.Tensor],
        v_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform dipole feature update.

        Args:
            q (torch.Tensor): Scalar features
            mu (dict(str, torch.Tensor): Dipole features.
            v_ij (torch.Tensor): Unnormalized direction vectors.
            idx_i (torch.Tensor): Indices of neighbor pairs atom i.
            idx_j (torch.Tensor): Indices of neighbor pairs atom j.
            rcut_ij (torch.Tensor): Cutoff for distances between i and j.

        Returns:
            dict(str, torch.Tensor): Updated dipole features for all fields.
        """
        for field in self.external_fields:
            qi = self.transform[field](q)
            dmu_ij = qi[idx_j] * rcut_ij[:, None, None] * v_ij[:, :, None]
            dmu_i = scatter_add(dmu_ij, idx_i, dim_size=q.shape[0])
            mu[field] = mu[field] + dmu_i

        return mu


class DipoleInteraction(nn.Module):
    def __init__(
        self,
        external_fields: List[str],
        n_atom_basis: int,
        n_rbf: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Compute the update to the scalar features based on the interactions between the dipole features.
        This uses the classical dipole-dipole interaction Tensor weighted by a radial basis function, as introduced in
        [#field3]_

        Args:
            external_fields (list(str)): List of external fields.
            n_atom_basis (int): Number of atomic features.
            n_rbf (int): Number of radial basis functions used in distance expansion.
            activation (Callable): Activation function.

        References:
        .. [#field3] Gastegger, Schütt, Müller:
           Machine learning of solvent effects on molecular spectra and reactions.
           Chemical Science, 12(34), 11473-11483. 2021.
        """
        super(DipoleInteraction, self).__init__()
        self.external_fields = external_fields

        self.transform = nn.ModuleDict(
            {
                field: Dense(n_atom_basis, n_atom_basis, activation=activation)
                for field in external_fields
            }
        )
        self.filter_network = nn.ModuleDict(
            {
                field: nn.Sequential(
                    Dense(n_rbf, n_atom_basis, activation=activation),
                    Dense(
                        n_atom_basis, n_atom_basis, activation=None, weight_init=zeros_
                    ),
                )
                for field in external_fields
            }
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: Dict[str, torch.Tensor],
        f_ij: torch.Tensor,
        d_ij: torch.Tensor,
        v_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the update to the scalar features based on the dipole-dipole interactions.

        Args:
            q (torch.Tensor): Scalar features
            mu (dict(str, torch.Tensor): Dipole features.
            f_ij (torch.Tensor): Distance expansion of interatomic distances.
            d_ij (torch.Tensor): Interatomic distances.
            v_ij (torch.Tensor): Unnormalized direction vectors.
            idx_i (torch.Tensor): Indices of neighbor pairs atom i.
            idx_j (torch.Tensor): Indices of neighbor pairs atom j.
            rcut_ij (torch.Tensor): Cutoff for distances between i and j

        Returns:
            torch.Tensor: Scalar update.
        """
        dq = 0.0

        for field in self.external_fields:
            Wij = self.filter_network[field](f_ij) * rcut_ij[..., None]
            Wij = Wij.unsqueeze(1)

            mu_ij = mu[field][idx_j]
            # Dipole - dipole interaction tensor
            tensor_ij = mu_ij * d_ij[:, None, None] ** 2 - 3.0 * v_ij[
                :, :, None
            ] * torch.sum(v_ij[:, :, None] * mu_ij, dim=1, keepdim=True)
            tensor_ij = tensor_ij * Wij / d_ij[:, None, None] ** 5
            tensor_i = scatter_add(tensor_ij, idx_i, dim_size=q.shape[0])
            dq_i = torch.sum(mu[field] * tensor_i, dim=1, keepdim=True)
            dq_i = self.transform[field](dq_i)

            dq = dq + dq_i

        return dq


class NuclearMagneticMomentEmbedding(nn.Module):
    """
    Special embedding for nuclear magnetic moments, since they can scale differently based on an atoms gyromagnetic
    ratio.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        max_z (int): Maximum number of atom types used in embedding.
    """

    def __init__(self, n_atom_basis: int, max_z: int):
        super(NuclearMagneticMomentEmbedding, self).__init__()
        self.gyromagnetic_ratio = nn.Embedding(max_z, 1, padding_idx=0)
        self.vector_mapping = snn.Dense(1, n_atom_basis, activation=None, bias=False)

    def forward(self, Z: torch.Tensor, nuclear_magnetic_moments: torch.Tensor):
        gamma = self.gyromagnetic_ratio(Z).unsqueeze(-1)
        delta_nmm = self.vector_mapping(nuclear_magnetic_moments.unsqueeze(-1))

        # for linear f f(a*x) = a * f(x)
        dmu = gamma * delta_nmm

        return dmu


class FieldSchNet(nn.Module):
    """FieldSchNet architecture for modeling interactions with external fields and response properties as described in
    [#field4]_.

    References:
    .. [#field4] Gastegger, Schütt, Müller:
       Machine learning of solvent effects on molecular spectra and reactions.
       Chemical Science, 12(34), 11473-11483. 2021.
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        external_fields: List[str] = [],
        response_properties: Optional[List[str]] = None,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = shifted_softplus,
        n_filters: int = None,
        shared_interactions: bool = False,
        max_z: int = 100,
        electric_field_modifier: Optional[nn.Module] = None,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            external_fields (list(str)): List of required external fields. Either this or the requested response
                                         properties needs to be specified.
            response_properties (list(str)): List of requested response properties. If this is not None, it is used to
                                             determine the required external fields.
            cutoff_fn: cutoff function
            activation (callable): activation function for nonlinearities.
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z (int): Maximum number of atom types used in embedding.
            electric_field_modifier (torch.nn.Module): If provided, use this module to modify the electric field. E.g.
                                                       for solvent models or fields from point charges in QM/MM.
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn

        if response_properties is not None:
            external_fields = required_fields_from_properties(response_properties)

        self.external_fields = external_fields
        self.electric_field_modifier = electric_field_modifier

        # layers
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        if properties.magnetic_field in self.external_fields:
            self.nmm_embedding = NuclearMagneticMomentEmbedding(
                n_atom_basis=n_atom_basis, max_z=max_z
            )
        else:
            self.nmm_embedding = None

        self.interactions = snn.replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

        # External field interactions
        self.field_interaction = snn.replicate_module(
            lambda: FieldSchNetFieldInteraction(
                external_fields=self.external_fields,
                n_atom_basis=n_atom_basis,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

        # Dipole interaction
        self.dipole_interaction = snn.replicate_module(
            lambda: DipoleInteraction(
                external_fields=self.external_fields,
                n_atom_basis=n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

        # Dipole updates
        self.initial_dipole_update = DipoleUpdate(
            external_fields=self.external_fields, n_atom_basis=n_atom_basis
        )
        self.dipole_update = snn.replicate_module(
            lambda: DipoleUpdate(
                external_fields=self.external_fields, n_atom_basis=n_atom_basis
            ),
            n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]
        idx_m = inputs[properties.idx_m]

        # Bring fields to final shape for model
        external_fields = {
            field: inputs[field][idx_m].unsqueeze(-1) for field in self.external_fields
        }

        # Apply field modifier
        if self.electric_field_modifier is not None:
            external_fields[properties.electric_field] = external_fields[
                properties.electric_field
            ] + self.electric_field_modifier(inputs)

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        q = self.embedding(atomic_numbers)[:, None]
        qs = q.shape
        mu = {
            field: torch.zeros((qs[0], 3, qs[2]), device=q.device)
            for field in self.external_fields
        }

        # First dipole update based on embeddings
        mu = self.initial_dipole_update(q, mu, r_ij, idx_i, idx_j, rcut_ij)

        if self.nmm_embedding is not None:
            mu[properties.magnetic_field] = mu[
                properties.magnetic_field
            ] + self.nmm_embedding(
                atomic_numbers, inputs[properties.nuclear_magnetic_moments]
            )

        for (
            i,
            (interaction, field_interaction, dipole_interaction, dipole_update),
        ) in enumerate(
            zip(
                self.interactions,
                self.field_interaction,
                self.dipole_interaction,
                self.dipole_update,
            )
        ):
            # Basic SchNet update
            dq = interaction(q.squeeze(1), f_ij, idx_i, idx_j, rcut_ij).unsqueeze(1)

            # Field and dipole updates
            dq_field = field_interaction(mu, external_fields)
            dq_dipole = dipole_interaction(
                q, mu, f_ij, d_ij, r_ij, idx_i, idx_j, rcut_ij
            )

            dq = dq + dq_field + dq_dipole
            q = q + dq

            mu = dipole_update(dq, mu, r_ij, idx_i, idx_j, rcut_ij)

        inputs["scalar_representation"] = q.squeeze(1)
        return inputs

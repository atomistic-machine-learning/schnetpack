from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

import schnetpack.properties as properties
import schnetpack.nn as snn
from schnetpack.representation import PaiNNInteraction, PaiNNMixing
from schnetpack.representation.field_schnet import (
    NuclearMagneticMomentEmbedding,
    _setup_external_fields,
)
from schnetpack.utils import required_fields_from_properties

__all__ = ["FieldPaiNN"]


class FieldInteraction(nn.Module):
    def __init__(
        self, external_fields: List[str], n_atom_basis: int, activation: Callable
    ):
        """
        Model the interaction of a molecule with an external field for PaiNN. Constructs a pseudo-polarizability from
        the scalar and vector features which is then multiplied with the field to update the PaiNN vector features.

        Args:
            external_fields (list(str)): List of external fields.
            n_atom_basis (int): number of features to describe atomic environments.
            activation (callable): if None, no activation function is used.
        """
        super(FieldInteraction, self).__init__()

        self.external_fields = external_fields

        scalar_component = {
            field: nn.Sequential(
                snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
                snn.Dense(n_atom_basis, n_atom_basis, activation=None),
            )
            for field in self.external_fields
        }

        vector_component = {
            field: snn.Dense(n_atom_basis, n_atom_basis, activation=None, bias=False)
            for field in self.external_fields
        }

        self.scalar_component = nn.ModuleDict(scalar_component)
        self.vector_component = nn.ModuleDict(vector_component)

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        external_fields: Dict[str, torch.Tensor],
    ):
        dmu = 0
        for field in self.external_fields:
            # Compute field interaction
            alpha_scalar = self.scalar_component[field](q)
            alpha_vector = self.vector_component[field](mu)
            dmu_field = (
                alpha_scalar * external_fields[field]
                - torch.sum(alpha_vector * external_fields[field], dim=1, keepdim=True)
                * alpha_vector
            )
            dmu = dmu + dmu_field

        mu = mu + dmu

        return q, mu


class FieldPaiNN(nn.Module):
    """Field dependent PaiNN architecture. This combines aspects of FieldSchNet [#field1]_ and PaiNN [#painn1]_ and
    can be used in the same manner as FieldSchNet to model response properties or molecular environments (solvent
    effects, QM/MM, ...).

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    .. [#field1] Gastegger, Schütt, Müller:
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
        activation: Optional[Callable] = F.silu,
        max_z: int = 100,
        electric_field_modifier: Optional[nn.Module] = None,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            n_atom_basis (int): number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions (int): number of interaction blocks.
            radial_basis (torch.nn.Module): layer for expanding interatomic distances in a basis set
            external_fields (list(str)): List of required external fields. Either this or the requested response
                                         properties needs to be specified.
            response_properties (list(str)): List of requested response properties. If this is not None, it is used to
                                             determine the required external fields.
            cutoff_fn (callable): cutoff function.
            activation (callable): activation function for nonlinearities.
            max_z (int): Maximum number of atom types used in embedding.
            electric_field_modifier (torch.nn.Module): If provided, use this module to modify the electric field. E.g.
                                                       for solvent models or fields from point charges in QM/MM.
            shared_interactions (bool): Share all interaction blocks.
            shared_filters (bool): Share all radial filters.
            epsilon (float): Small numerical offset for stable computation of vector norm derivatives.
        """
        super(FieldPaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis

        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        if response_properties is not None:
            external_fields = required_fields_from_properties(response_properties)

        self.external_fields = external_fields
        self.electric_field_modifier = electric_field_modifier

        self.share_filters = shared_filters

        if shared_filters:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf, 3 * n_atom_basis, activation=None
            )
        else:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * n_atom_basis * 3,
                activation=None,
            )

        if properties.magnetic_field in self.external_fields:
            self.nmm_embedding = NuclearMagneticMomentEmbedding(
                n_atom_basis=n_atom_basis, max_z=max_z
            )
        else:
            self.nmm_embedding = None

        self.interactions = snn.replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.n_atom_basis, activation=activation
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.field_interactions = snn.replicate_module(
            lambda: FieldInteraction(
                external_fields=self.external_fields,
                n_atom_basis=n_atom_basis,
                activation=activation,
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = snn.replicate_module(
            lambda: PaiNNMixing(
                n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon
            ),
            self.n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        idx_m = inputs[properties.idx_m]
        n_atoms = atomic_numbers.shape[0]

        field_components = _setup_external_fields(inputs, self.external_fields)
        # Bring fields to final shape for model
        external_fields = {
            field: field_components[field][idx_m].unsqueeze(-1)
            for field in self.external_fields
        }

        # Modify electric field if requested (e.g. build from point charges)
        if (
            self.electric_field_modifier is not None
            and properties.electric_field in self.external_fields
        ):
            external_fields[properties.electric_field] = self.electric_field_modifier(
                inputs
            )

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.embedding(atomic_numbers)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        # Embed nuclear magnetic moment information for spin-spin coupling and shielding
        if self.nmm_embedding is not None:
            mu = mu + self.nmm_embedding(
                atomic_numbers, field_components[properties.nuclear_magnetic_moments]
            )

        for i, (interaction, field_interaction, mixing) in enumerate(
            zip(self.interactions, self.field_interactions, self.mixing)
        ):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = field_interaction(q, mu, external_fields)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        output_dict = {"scalar_representation": q, "vector_representation": mu}
        # Add field components for derivatives
        output_dict.update(field_components)

        return output_dict

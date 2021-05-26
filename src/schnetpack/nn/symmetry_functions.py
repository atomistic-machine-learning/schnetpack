import torch
from torch import nn
import numpy as np

from typing import Union, Sequence

__all__ = [
    "ElementBasis",
    "OneHotElements",
    "WeightedElements",
    "AngularBasis",
    "AngularBasisANI",
]


class ElementBasis(nn.Module):
    """Base class for introducing element-dependence in symmetry function representations."""

    def __init__(self, n_ebf: int, n_ebf_combined: int):
        """
        Args:
            n_ebf (int): number of basis function used for each element.
            n_ebf_combined (int): number of basis functions resulting from combining two elemental vectors with
                                  `_combine_elements`, e.g. for angular representations.
        """
        super(ElementBasis, self).__init__()

        # number of features in original and combined elemental basis
        self.n_ebf = n_ebf
        self.n_ebf_combined = n_ebf_combined

        # elemental embedding should be initialized in _init_basis
        self.elemental_embedding = torch.nn.Identity()

    def _init_basis(self):
        """Routine for initializing components of the basis"""
        pass

    def forward(self, Zj: torch.Tensor):
        """
        Args:
            Zj (torch.Tensor): atomic numbers.

        Returns:
            torch.Tensor: elemental basis vectors.
        """
        return self.elemental_embedding(Zj)

    def combine_elements(self, Zj: torch.Tensor, Zk: torch.Tensor):
        """
        Args:
            Zj (torch.Tensor): atomic numbers or elemental representation for atoms j.
            Zk (torch.Tensor): atomic numbers or elemental representation for atoms k.

        Returns:
            torch.Tensor: combination of the two elemental basis vectors.
        """
        return Zj * Zk


class WeightedElements(ElementBasis):
    """
    Elemental basis for wACSF type symmetry functions.

    References:
    .. [#wacsf1] Gastegger, Schwiedrzik, Bittermann, Berzsenyi, Marquetand:
       wACSF -- Weighted atom-centered symmetry functions as descriptors in machine learning potentials.
       The Journal of Chemical Physics 148 (24), 241709. 2018.
    """

    def __init__(self):
        super(WeightedElements, self).__init__(1, 1)

    def forward(self, Zj: torch.Tensor):
        """
        Args:
            Zj (torch.Tensor): atomic numbers.

        Returns:
            torch.Tensor: atomic numbers.
        """
        return Zj.unsqueeze(-1)


class OneHotElements(ElementBasis):
    """
    One hot element encoding as used in the original atom-centered symmetry functions.

    References:
    .. [#acsf1] Behler:
       Atom-centered symmetry functions for constructing high-dimensional neural network potentials.
       The Journal of Chemical Physics 134. 074106. 2011.
    """

    def __init__(self, elements: Union[int, Sequence[int]], max_z: int = 100):
        """
        Args:
            elements (list(int)): list of elements to be encoded.
            max_z (int, optional): maximum nuclear charge that can be treated in embedding.
        """
        super(ElementBasis, self).__init__()
        n_elements = len(elements)
        n_elements_pairs = int((n_elements * (n_elements + 1)) / 2)

        super(OneHotElements, self).__init__(n_elements, n_elements_pairs)

        self.elements = elements
        self.n_elements = n_elements
        self.max_z = max_z

        # Initialize elemental basis
        elemental_weights = self._init_basis()
        self.elemental_embedding = nn.Embedding.from_pretrained(
            elemental_weights, freeze=True
        )

        # indices of upper triangular matrix for combination of two one-hot encodings (for angular SF)
        self._idx_j, self._idx_k = torch.triu_indices(self.n_ebf, self.n_ebf, offset=0)

    def _init_basis(self):
        """
        Initialize the one hot encoding for the requested elements.

        Returns:
            torch.Tensor: one-hot elemental weight matrix.
        """
        elemental_weights = torch.zeros(self.max_z, self.n_ebf)

        for idx, element in enumerate(self.elements):
            elemental_weights[element, idx] = 1.0

        return elemental_weights

    def combine_elements(self, Zj: torch.Tensor, Zk: torch.Tensor):
        """
        Combine two one hot encodings for angular representations.

        Args:
            Zj (torch.Tensor): one-hot encoding for atoms j.
            Zk (torch.Tensor): one-hot encoding for atoms k.

        Returns:
            torch.Tensor: combined one-hot representation.
        """
        Zjk = Zj[:, :, None] * Zk[:, None, :]
        Zjk = torch.triu(Zjk, diagonal=1) + Zjk.transpose(1, 2)
        Zjk = Zjk[:, self._idx_j, self._idx_k]
        return Zjk


class AngularBasis(nn.Module):
    """
    Angular component of standard atom-centered symmetry functions.

    References:
    .. [#acsf1] Behler:
       Atom-centered symmetry functions for constructing high-dimensional neural network potentials.
       The Journal of Chemical Physics 134. 074106. 2011.
    """

    def __init__(self, zetas: Union[int, Sequence[int]]):
        """
        Args:
            zetas list(int): list of exponents regulating the resolution of the angular term.
        """
        super(AngularBasis, self).__init__()
        self.register_buffer("zetas", torch.Tensor(zetas).unsqueeze(0))
        self.n_abf = 2 * len(zetas)

    def forward(self, cos_theta_ijk: torch.Tensor):
        """
        Compute the angular component.

        Args:
            cos_theta_ijk (torch.Tensor): cosines of all angles spanned by atoms i, j and k.

        Returns:
            torch.Tensor: vector of angular symmetry function components.
        """
        angular_norm = torch.pow(2.0, 1 - self.zetas)
        angular_pos = angular_norm * torch.pow(
            1.0 + cos_theta_ijk.unsqueeze(-1), self.zetas
        )
        angular_neg = angular_norm * torch.pow(
            1.0 - cos_theta_ijk.unsqueeze(-1), self.zetas
        )

        return torch.cat((angular_pos, angular_neg), dim=1)


class AngularBasisANI(nn.Module):
    """
    Angular component of the Justin-Smith variant of Behler type angular functions.

    References:
    .. [#ani1] Smith, Isayev, Roitberg:
       ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost.
       Chemical science 8(4). 3192--3203. 2017.
    """

    def __init__(self, n_abf: int, zetas: Union[int, Sequence[int]]):
        """
        Args:
            n_abf (int): number of angle increments.
            zetas list(int): list of exponents regulating the resolution of the angular term.
        """
        super(AngularBasisANI, self).__init__()
        self.register_buffer(
            "angular_offsets", torch.linspace(0, np.pi, n_abf).unsqueeze(0)
        )
        self.register_buffer("zetas", torch.Tensor(zetas))
        self.n_abf = 2 * len(zetas) * n_abf

    def forward(self, cos_theta_ijk: torch.Tensor, eps: float = 1e-6):
        """
        Compute the angular component.

        Args:
            cos_theta_ijk (torch.Tensor): cosines of all angles spanned by atoms i, j and k.
            eps (float, optional): small offset to avoid numerical issues with the arccos function.

        Returns:
            torch.Tensor: vector of angular symmetry function components of the Justin-Smith type.
        """
        angular_norm = torch.pow(2.0, 1 - self.zetas)

        # Ensure numerical stability
        cos_theta_ijk = cos_theta_ijk.clamp(-1.0 + eps, 1.0 - eps)
        angular = torch.cos(
            torch.acos(cos_theta_ijk.unsqueeze(-1)) - self.angular_offsets
        )

        angular_pos = angular_norm * torch.pow(
            (1.0 + angular)[:, :, None], self.zetas[None, None, :]
        )
        angular_neg = angular_norm * torch.pow(
            (1.0 - angular)[:, :, None], self.zetas[None, None, :]
        )

        angular = torch.cat((angular_pos, angular_neg), dim=1)
        angular = angular.view(-1, self.n_abf)

        return angular

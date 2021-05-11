import torch
from torch import nn
import numpy as np


class ElementBasis(nn.Module):
    def __init__(self, n_ebf, n_ebf_combined):
        super(ElementBasis, self).__init__()

        # number of features in original and combined elemental basis
        self.n_ebf = n_ebf
        self.n_ebf_combined = n_ebf_combined

        # elemental embedding should be initialized in _init_basis
        self.elemental_embedding = torch.nn.Identity()

    def _init_basis(self):
        pass

    def forward(self, Zj):
        return self.elemental_embedding(Zj)

    def combine_elements(self, Zj, Zk):
        return Zj * Zk


class WeightedElements(ElementBasis):
    def __init__(self):
        super(WeightedElements, self).__init__(1, 1)

    def forward(self, Zj):
        return Zj.unsqueeze(-1)


class OneHotElements(ElementBasis):
    def __init__(self, elements: list, max_z: int = 100):
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
        elemental_weights = torch.zeros(self.max_z, self.n_ebf)

        for idx, element in enumerate(self.elements):
            elemental_weights[element, idx] = 1.0

        return elemental_weights

    def combine_elements(self, Zj, Zk):
        Zjk = Zj[:, :, None] * Zk[:, None, :]
        Zjk = torch.triu(Zjk, diagonal=1) + Zjk.transpose(1, 2)
        Zjk = Zjk[:, self._idx_j, self._idx_k]
        return Zjk


class AngularBasis(nn.Module):
    def __init__(self, zetas: list):
        super(AngularBasis, self).__init__()
        self.register_buffer("zetas", torch.Tensor(zetas).unsqueeze(0))
        self.n_abf = 2 * len(zetas)

    def forward(self, cos_theta_ijk):
        angular_norm = torch.pow(2.0, 1 - self.zetas)
        angular_pos = angular_norm * torch.pow(
            1.0 + cos_theta_ijk.unsqueeze(-1), self.zetas
        )
        angular_neg = angular_norm * torch.pow(
            1.0 - cos_theta_ijk.unsqueeze(-1), self.zetas
        )

        return torch.cat((angular_pos, angular_neg), dim=1)


class AngularBasisANI(nn.Module):
    def __init__(self, n_abf, zetas: list):
        super(AngularBasisANI, self).__init__()
        self.register_buffer(
            "angular_offsets", torch.linspace(0, np.pi, n_abf).unsqueeze(0)
        )
        self.register_buffer("zetas", torch.Tensor(zetas))
        self.n_abf = 2 * len(zetas) * n_abf

    def forward(self, cos_theta_ijk, eps=1e-6):
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

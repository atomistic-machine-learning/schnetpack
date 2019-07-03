import torch
from torch import nn

from schnetpack.nn.cutoff import CosineCutoff

__all__ = [
    "AngularDistribution",
    "BehlerAngular",
    "GaussianSmearing",
    "RadialDistribution",
]


class AngularDistribution(nn.Module):
    """
    Routine used to compute angular type symmetry functions between all atoms i-j-k, where i is the central atom.

    Args:
        radial_filter (callable): Function used to expand distances (e.g. Gaussians)
        angular_filter (callable): Function used to expand angles between triples of atoms (e.g. BehlerAngular)
        cutoff_functions (callable): Cutoff function
        crossterms (bool): Include radial contributions of the distances r_jk
        pairwise_elements (bool): Recombine elemental embedding vectors via an outer product. If e.g. one-hot encoding
            is used for the elements, this is equivalent to standard Behler functions
            (default=False).

    """

    def __init__(
        self,
        radial_filter,
        angular_filter,
        cutoff_functions=CosineCutoff,
        crossterms=False,
        pairwise_elements=False,
    ):
        super(AngularDistribution, self).__init__()
        self.radial_filter = radial_filter
        self.angular_filter = angular_filter
        self.cutoff_function = cutoff_functions
        self.crossterms = crossterms
        self.pairwise_elements = pairwise_elements

    def forward(self, r_ij, r_ik, r_jk, triple_masks=None, elemental_weights=None):
        """
        Args:
            r_ij (torch.Tensor): Distances to neighbor j
            r_ik (torch.Tensor): Distances to neighbor k
            r_jk (torch.Tensor): Distances between neighbor j and k
            triple_masks (torch.Tensor): Tensor mask for non-counted pairs (e.g. due to cutoff)
            elemental_weights (tuple of two torch.Tensor): Weighting functions for neighboring elements, first is for
                                                            neighbors j, second for k

        Returns:
            torch.Tensor: Angular distribution functions

        """

        nbatch, natoms, npairs = r_ij.size()

        # compute gaussilizated distances and cutoffs to neighbor atoms
        radial_ij = self.radial_filter(r_ij)
        radial_ik = self.radial_filter(r_ik)
        angular_distribution = radial_ij * radial_ik

        if self.crossterms:
            radial_jk = self.radial_filter(r_jk)
            angular_distribution = angular_distribution * radial_jk

        # Use cosine rule to compute cos( theta_ijk )
        cos_theta = (torch.pow(r_ij, 2) + torch.pow(r_ik, 2) - torch.pow(r_jk, 2)) / (
            2.0 * r_ij * r_ik
        )

        # Required in order to catch NaNs during backprop
        if triple_masks is not None:
            cos_theta[triple_masks == 0] = 0.0

        angular_term = self.angular_filter(cos_theta)

        if self.cutoff_function is not None:
            cutoff_ij = self.cutoff_function(r_ij).unsqueeze(-1)
            cutoff_ik = self.cutoff_function(r_ik).unsqueeze(-1)
            angular_distribution = angular_distribution * cutoff_ij * cutoff_ik

            if self.crossterms:
                cutoff_jk = self.cutoff_function(r_jk).unsqueeze(-1)
                angular_distribution = angular_distribution * cutoff_jk

        # Compute radial part of descriptor
        if triple_masks is not None:
            # Filter out nan divisions via boolean mask, since
            # angular_term = angular_term * triple_masks
            # is not working (nan*0 = nan)
            angular_term[triple_masks == 0] = 0.0
            angular_distribution[triple_masks == 0] = 0.0

        # Apply weights here, since dimension is still the same
        if elemental_weights is not None:
            if not self.pairwise_elements:
                Z_ij, Z_ik = elemental_weights
                Z_ijk = Z_ij * Z_ik
                angular_distribution = (
                    torch.unsqueeze(angular_distribution, -1)
                    * torch.unsqueeze(Z_ijk, -2).float()
                )
            else:
                # Outer product to emulate vanilla SF behavior
                Z_ij, Z_ik = elemental_weights
                B, A, N, E = Z_ij.size()
                pair_elements = Z_ij[:, :, :, :, None] * Z_ik[:, :, :, None, :]
                pair_elements = pair_elements + pair_elements.permute(0, 1, 2, 4, 3)
                # Filter out lower triangular components
                pair_filter = torch.triu(torch.ones(E, E)) == 1
                pair_elements = pair_elements[:, :, :, pair_filter]
                angular_distribution = torch.unsqueeze(
                    angular_distribution, -1
                ) * torch.unsqueeze(pair_elements, -2)

        # Dimension is (Nb x Nat x Nneighpair x Nrad) for angular_distribution and
        # (Nb x Nat x NNeigpair x Nang) for angular_term, where the latter dims are orthogonal
        # To multiply them:
        angular_distribution = (
            angular_distribution[:, :, :, :, None, :]
            * angular_term[:, :, :, None, :, None]
        )
        # For the sum over all contributions
        angular_distribution = torch.sum(angular_distribution, 2)
        # Finally, we flatten the last two dimensions
        angular_distribution = angular_distribution.view(nbatch, natoms, -1)

        return angular_distribution


class BehlerAngular(nn.Module):
    """
    Compute Behler type angular contribution of the angle spanned by three atoms:

    :math:`2^{(1-\zeta)} (1 + \lambda \cos( {\\theta}_{ijk} ) )^\zeta`

    Sets of zetas with lambdas of -1 and +1 are generated automatically.

    Args:
        zetas (set of int): Set of exponents used to compute angular Behler term (default={1})

    """

    def __init__(self, zetas={1}):
        super(BehlerAngular, self).__init__()
        self.zetas = zetas

    def forward(self, cos_theta):
        """
        Args:
            cos_theta (torch.Tensor): Cosines between all pairs of neighbors of the central atom.

        Returns:
            torch.Tensor: Tensor containing values of the angular filters.
        """
        angular_pos = [
            2 ** (1 - zeta) * ((1.0 - cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        angular_neg = [
            2 ** (1 - zeta) * ((1.0 + cos_theta) ** zeta).unsqueeze(-1)
            for zeta in self.zetas
        ]
        angular_all = angular_pos + angular_neg
        return torch.cat(angular_all, -1)


def gaussian_smearing(distances, offset, widths, centered=False):
    r"""Smear interatomic distance values using Gaussian functions.

    Args:
        distances (torch.Tensor): interatomic distances of (N_b x N_at x N_nbh) shape.
        offset (torch.Tensor): offsets values of Gaussian functions.
        widths: width values of Gaussian functions.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).

    Returns:
        torch.Tensor: smeared distances (N_b x N_at x N_nbh x N_g).

    """
    if not centered:
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # if Gaussian functions are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # if Gaussian functions are centered, no offset is subtracted
        diff = distances[:, :, :, None]
    # compute smear distance values
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianSmearing(nn.Module):
    r"""Smear layer using a set of Gaussian functions.

    Args:
        start (float, optional): center of first Gaussian function, :math:`\mu_0`.
        stop (float, optional): center of last Gaussian function, :math:`\mu_{N_g}`
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`.
        centered (bool, optional): If True, Gaussians are centered at the origin and
            the offsets are used to as their widths (used e.g. for angular functions).
        trainable (bool, optional): If True, widths and offset of Gaussian functions
            are adjusted during training process.

    """

    def __init__(
        self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False
    ):
        super(GaussianSmearing, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("width", widths)
            self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        return gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )


class RadialDistribution(nn.Module):
    """
    Radial distribution function used e.g. to compute Behler type radial symmetry functions.

    Args:
        radial_filter (callable): Function used to expand distances (e.g. Gaussians)
        cutoff_function (callable): Cutoff function
    """

    def __init__(self, radial_filter, cutoff_function=CosineCutoff):
        super(RadialDistribution, self).__init__()
        self.radial_filter = radial_filter
        self.cutoff_function = cutoff_function

    def forward(self, r_ij, elemental_weights=None, neighbor_mask=None):
        """
        Args:
            r_ij (torch.Tensor): Interatomic distances
            elemental_weights (torch.Tensor): Element-specific weights for distance functions
            neighbor_mask (torch.Tensor): Mask to identify positions of neighboring atoms

        Returns:
            torch.Tensor: Nbatch x Natoms x Nfilter tensor containing radial distribution functions.
        """

        nbatch, natoms, nneigh = r_ij.size()

        radial_distribution = self.radial_filter(r_ij)

        # If requested, apply cutoff function
        if self.cutoff_function is not None:
            cutoffs = self.cutoff_function(r_ij)
            radial_distribution = radial_distribution * cutoffs.unsqueeze(-1)

        # Apply neighbor mask
        if neighbor_mask is not None:
            radial_distribution = radial_distribution * torch.unsqueeze(
                neighbor_mask, -1
            )

        # Weigh elements if requested
        if elemental_weights is not None:
            radial_distribution = (
                radial_distribution[:, :, :, :, None]
                * elemental_weights[:, :, :, None, :].float()
            )

        radial_distribution = torch.sum(radial_distribution, 2)
        radial_distribution = radial_distribution.view(nbatch, natoms, -1)
        return radial_distribution

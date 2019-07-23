import torch
import torch.nn as nn

import schnetpack.nn as snn
from schnetpack.data import StatisticsAccumulator
from schnetpack import Properties


class HDNNException(Exception):
    pass


class SymmetryFunctions(nn.Module):
    """
    Compute atom-centered symmetry functions [#acsf1]_ and weighted variant thereof as described
    in Reference [#wacsf1]_.
    By default, the atomic number is used as element depended weight. However, by specifying the
    trainz=True, a more general elemental embedding is learned instead.

    Args:
        n_radial (int):  Number of radial functions
        n_angular (int): Number of angular functions
        zetas (set of int): Set of exponents used to compute the angular term, default is zetas={1}
        cutoff (callable): Cutoff function, default is the typical cosine cutoff function
        cutoff_radius (float): Cutoff radius, default are 5 Angstrom
        centered (bool): Whether centered Gaussians should be used for radial functions. Angular functions use centered Gaussians by default.
        crossterms (bool): Whether cutoff and exponential terms of the distance r_jk between both neighbors should be included in the angular functions. Default is False
        elements (set of int): Nuclear charge present in the molecules, default is {1,6,7,8,9} (H,C,N,O and F).
        sharez (bool): Whether angular and radial functions should use the same elemental weighting. The default is true.
        trainz (bool): If set to true, elemental weights are initialized at random and learned during training. (default is False)
        initz (str): How elemental weights are initialized. Allowed are (default='weighted'):
                        weighted: Weigh symmetry functions with nuclear charges (wACSF)
                        onehot: Represent elements by onehot vectors. This can be used in combination with pairwise_elements
                                in order to emulate the behavior of classic Behler symmetry functions.
                        embedding: Use random embeddings, which can e.g. be trained. Embedding length is modified via
                                   len_embedding (default=False).
        len_embedding (int): Number of elemental weights, default is 1. If more are used, embedding vectors similar to SchNet can be obtained.
        pairwise_elements (bool): Recombine elemental embedding vectors in the angular functions via an outer product. If e.g. one-hot encoding
                                  is used for the elements, this is equivalent to standard Behler functions
                                  (default=False).

    References
    ----------
        .. [#acsf1] Behler:
           Atom-centered symmetry functions for constructing high-dimensional neural network potentials.
           The Journal of Chemical Physics 134. 074106. 2011.
        .. [#wacsf1] Gastegger, Schwiedrzik, Bittermann, Berzsenyi, Marquetand:
           wACSF -- Weighted atom-centered symmetry functions as descriptors in machine learning potentials.
           The Journal of Chemical Physics 148 (24), 241709. 2018.
    """

    def __init__(
        self,
        n_radial=22,
        n_angular=5,
        zetas={1},
        cutoff=snn.CosineCutoff,
        cutoff_radius=5.0,
        centered=False,
        crossterms=False,
        elements=frozenset((1, 6, 7, 8, 9)),
        sharez=True,
        trainz=False,
        initz="weighted",
        len_embedding=5,
        pairwise_elements=False,
    ):

        super(SymmetryFunctions, self).__init__()

        self.n_radial = n_radial
        self.n_angular = n_angular
        self.len_embedding = len_embedding
        self.n_elements = None
        self.n_theta = 2 * len(zetas)

        # Initialize cutoff function
        self.cutoff_radius = cutoff_radius
        self.cutoff = cutoff(cutoff=self.cutoff_radius)

        # Check for general stupidity:
        if self.n_angular < 1 and self.n_radial < 1:
            raise ValueError("At least one type of SF required")

        if self.n_angular > 0:
            # Get basic filters
            self.theta_filter = snn.BehlerAngular(zetas=zetas)
            self.angular_filter = snn.GaussianSmearing(
                start=1.0,
                stop=self.cutoff_radius - 0.5,
                n_gaussians=n_angular,
                centered=True,
            )
            self.ADF = snn.AngularDistribution(
                self.angular_filter,
                self.theta_filter,
                cutoff_functions=self.cutoff,
                crossterms=crossterms,
                pairwise_elements=pairwise_elements,
            )
        else:
            self.ADF = None

        if self.n_radial > 0:
            # Get basic filters (if centered Gaussians are requested, start is set to 0.5
            if centered:
                radial_start = 1.0
            else:
                radial_start = 0.5
            self.radial_filter = snn.GaussianSmearing(
                start=radial_start,
                stop=self.cutoff_radius - 0.5,
                n_gaussians=n_radial,
                centered=centered,
            )
            self.RDF = snn.RadialDistribution(
                self.radial_filter, cutoff_function=self.cutoff
            )
        else:
            self.RDF = None

        # Initialize the atomtype embeddings
        self.radial_Z = self.initz(initz, elements)

        # check whether angular functions should use the same embedding
        if sharez:
            self.angular_Z = self.radial_Z
        else:
            self.angular_Z = self.initz(initz, elements)

        # Turn of training of embeddings unless requested explicitly
        if not trainz:
            # Turn off gradients
            self.radial_Z.weight.requires_grad = False
            self.angular_Z.weight.requires_grad = False

        # Compute total number of symmetry functions
        if not pairwise_elements:
            self.n_symfuncs = (
                self.n_radial + self.n_angular * self.n_theta
            ) * self.n_elements
        else:
            # if the outer product is used, all unique pairs of elements are considered, leading to the factor of
            # (N+1)/2
            self.n_symfuncs = (
                self.n_radial
                + self.n_angular * self.n_theta * (self.n_elements + 1) // 2
            ) * self.n_elements

    def initz(self, mode, elements):
        """
        Subroutine to initialize the element dependent weights.

        Args:
            mode (str): Manner in which the weights are initialized. Possible are:
                weighted: Weigh symmetry functions with nuclear charges (wACSF)
                onehot: Represent elements by onehot vectors. This can be used in combination with pairwise_elements
                        in order to emulate the behavior of classic Behler symmetry functions.
                embedding: Use random embeddings, which can e.g. be trained. Embedding length is modified via
                           len_embedding (default=False).
            elements (set of int): List of elements present in the molecule.

        Returns:
            torch.nn.Embedding: Embedding layer of the initialized elemental weights.

        """

        maxelements = max(elements)
        nelements = len(elements)

        if mode == "weighted":
            weights = torch.arange(maxelements + 1)[:, None]
            z_weights = nn.Embedding(maxelements + 1, 1)
            z_weights.weight.data = weights
            self.n_elements = 1
        elif mode == "onehot":
            weights = torch.zeros(maxelements + 1, nelements)
            for idx, Z in enumerate(elements):
                weights[Z, idx] = 1.0
            z_weights = nn.Embedding(maxelements + 1, nelements)
            z_weights.weight.data = weights
            self.n_elements = nelements
        elif mode == "embedding":
            z_weights = nn.Embedding(maxelements + 1, self.len_embedding)
            self.n_elements = self.len_embedding
        else:
            raise NotImplementedError(
                "Unregognized option {:s} for initializing elemental weights. Use 'weighted', 'onehot' or 'embedding'.".format(
                    mode
                )
            )

        return z_weights

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Nbatch x Natoms x Nsymmetry_functions Tensor containing ACSFs or wACSFs.

        """
        positions = inputs[Properties.R]
        Z = inputs[Properties.Z]
        neighbors = inputs[Properties.neighbors]
        neighbor_mask = inputs[Properties.neighbor_mask]

        cell = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]

        # Compute radial functions
        if self.RDF is not None:
            # Get atom type embeddings
            Z_rad = self.radial_Z(Z)
            # Get atom types of neighbors
            Z_ij = snn.neighbor_elements(Z_rad, neighbors)
            # Compute distances
            distances = snn.atom_distances(
                positions,
                neighbors,
                neighbor_mask=neighbor_mask,
                cell=cell,
                cell_offsets=cell_offset,
            )
            radial_sf = self.RDF(
                distances, elemental_weights=Z_ij, neighbor_mask=neighbor_mask
            )
        else:
            radial_sf = None

        if self.ADF is not None:
            # Get pair indices
            try:
                idx_j = inputs[Properties.neighbor_pairs_j]
                idx_k = inputs[Properties.neighbor_pairs_k]

            except KeyError as e:
                raise HDNNException(
                    "Angular symmetry functions require "
                    + "`collect_triples=True` in AtomsData."
                )
            neighbor_pairs_mask = inputs[Properties.neighbor_pairs_mask]

            # Get element contributions of the pairs
            Z_angular = self.angular_Z(Z)
            Z_ij = snn.neighbor_elements(Z_angular, idx_j)
            Z_ik = snn.neighbor_elements(Z_angular, idx_k)

            # Offset indices
            offset_idx_j = inputs[Properties.neighbor_offsets_j]
            offset_idx_k = inputs[Properties.neighbor_offsets_k]

            # Compute triple distances
            r_ij, r_ik, r_jk = snn.triple_distances(
                positions,
                idx_j,
                idx_k,
                offset_idx_j=offset_idx_j,
                offset_idx_k=offset_idx_k,
                cell=cell,
                cell_offsets=cell_offset,
            )

            angular_sf = self.ADF(
                r_ij,
                r_ik,
                r_jk,
                elemental_weights=(Z_ij, Z_ik),
                triple_masks=neighbor_pairs_mask,
            )
        else:
            angular_sf = None

        # Concatenate and return symmetry functions
        if self.RDF is None:
            symmetry_functions = angular_sf
        elif self.ADF is None:
            symmetry_functions = radial_sf
        else:
            symmetry_functions = torch.cat((radial_sf, angular_sf), 2)

        return symmetry_functions


class BehlerSFBlock(SymmetryFunctions):
    """
    Utility layer for fast initialisation of ACSFs and wACSFs.

    Args:
        n_radial (int):  Number of radial functions
        n_angular (int): Number of angular functions
        zetas (set of int): Set of exponents used to compute the angular term, default is zetas={1}
        cutoff_radius (float): Cutoff radius, default are 5 Angstrom
        elements (set of int): Nuclear charge present in the molecules, default is {1,6,7,8,9} (H,C,N,O and F).
        centered (bool): Whether centered Gaussians should be used for radial functions. Angular functions use centered Gaussians by default.
        crossterms (bool): Whether cutoff and exponential terms of the distance r_jk between both neighbors should be included in the angular functions. Default is False
        mode (str): Manner in which the weights are initialized. Possible are:
            weighted: Weigh symmetry functions with nuclear charges (wACSF)
            onehot: Represent elements by onehot vectors. This can be used in combination with pairwise_elements
                    in order to emulate the behavior of classic Behler symmetry functions (ACSF).
            embedding: Use random embeddings, which can e.g. be trained. Embedding length is modified via
                       len_embedding (default=False).
    """

    def __init__(
        self,
        n_radial=22,
        n_angular=5,
        zetas={1},
        cutoff_radius=5.0,
        elements=frozenset((1, 6, 7, 8, 9)),
        centered=False,
        crossterms=False,
        mode="weighted",
    ):
        # Determine mode.
        if mode == "weighted":
            initz = "weighted"
            pairwise_elements = False
        elif mode == "Behler":
            initz = "onehot"
            pairwise_elements = True
        else:
            raise NotImplementedError("Unrecognized symmetry function %s" % mode)

        # Construct symmetry functions.
        super(BehlerSFBlock, self).__init__(
            n_radial=n_radial,
            n_angular=n_angular,
            zetas=zetas,
            cutoff_radius=cutoff_radius,
            centered=centered,
            crossterms=crossterms,
            elements=elements,
            initz=initz,
            pairwise_elements=pairwise_elements,
        )


class StandardizeSF(nn.Module):
    """
    Compute mean and standard deviation of all symmetry functions computed for the molecules in the data loader
    and use them to standardize the descriptor vectors,

    Args:
        SFBlock (callable): Object for computing the descriptor vectors
        data_loader (object): DataLoader containing the molecules used for computing the statistics. If None, dummy
                              vectors are generated instead
        cuda (bool): Cuda flag
    """

    def __init__(self, SFBlock, data_loader=None, cuda=False):

        super(StandardizeSF, self).__init__()

        device = torch.device("cuda" if cuda else "cpu")

        self.n_symfuncs = SFBlock.n_symfuncs

        if data_loader is not None:
            symfunc_statistics = StatisticsAccumulator(batch=True, atomistic=True)
            SFBlock = SFBlock.to(device)

            for sample in data_loader:
                if cuda:
                    sample = {k: v.to(device) for k, v in sample.items()}
                symfunc_values = SFBlock.forward(sample)
                symfunc_statistics.add_sample(symfunc_values.detach())

            SF_mean, SF_stddev = symfunc_statistics.get_statistics()

        else:
            SF_mean = torch.zeros(self.n_symfuncs)
            SF_stddev = torch.ones(self.n_symfuncs)

        self.SFBlock = SFBlock
        self.standardize = snn.Standardize(SF_mean, SF_stddev)

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Standardized representations.
        """
        representation = self.SFBlock(inputs)
        return self.standardize(representation)

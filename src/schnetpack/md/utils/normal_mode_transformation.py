import numpy as np
import torch

__all__ = ["NormalModeTransformer"]


class NormalModeTransformer:
    """
    Class for transforming between bead and normal mode representation of the ring polymer, used e.g. in propagating the
    ring polymer during simulation. An in depth description of the transformation can be found e.g. in [#rpmd3]_. Here,
    a simple matrix multiplication is used instead of a Fourier transformation, which can be more performant in certain
    cases. On the GPU however, no significant performance gains where observed when using a FT based transformation over
    the matrix version.

    This transformation operates on the first dimension of the property tensors (e.g. positions, momenta) defined in the
    system class. Hence, the transformation can be carried out for several molecules at the same time.

    Args:
        n_beads (int): Number of beads in the ring polymer.
        device (str): Computation device (default='cuda').

    References
    ----------
    .. [#rpmd3] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133, 124105. 2010.
    """

    def __init__(self, n_beads, device="cuda"):
        self.n_beads = n_beads

        self.device = device

        # Initialize the transformation matrix
        self.c_transform = self._init_transformation_matrix()

    def _init_transformation_matrix(self):
        """
        Build the normal mode transformation matrix. This matrix only has to be built once and can then be used during
        the whole simulation. The matrix has the dimension n_beads x n_beads, where n_beads is the number of beads in
        the ring polymer

        Returns:
            torch.Tensor: Normal mode transformation matrix of the shape n_beads x n_beads
        """
        # Set up basic transformation matrix
        c_transform = np.zeros((self.n_beads, self.n_beads))

        # Get auxiliary array with bead indices
        n = np.arange(1, self.n_beads + 1)

        # for k = 0
        c_transform[0, :] = 1.0

        for k in range(1, self.n_beads // 2 + 1):
            c_transform[k, :] = np.sqrt(2) * np.cos(2 * np.pi * k * n / self.n_beads)

        for k in range(self.n_beads // 2 + 1, self.n_beads):
            c_transform[k, :] = np.sqrt(2) * np.sin(2 * np.pi * k * n / self.n_beads)

        if self.n_beads % 2 == 0:
            c_transform[self.n_beads // 2, :] = (-1) ** n

        # Since matrix is initialized as C(k,n) does not need to be transposed
        c_transform /= np.sqrt(self.n_beads)
        c_transform = torch.from_numpy(c_transform).float().to(self.device)

        return c_transform

    def beads2normal(self, x_beads):
        """
        Transform a system tensor (e.g. momenta, positions) from the bead representation to normal mode representation.

        Args:
            x_beads (torch.Tensor): System tensor in bead representation with the general shape
                                    n_beads x n_molecules x ...

        Returns:
            torch.Tensor: System tensor in normal mode representation with the same shape as the input tensor.
        """
        return torch.mm(self.c_transform, x_beads.view(self.n_beads, -1)).view(
            x_beads.shape
        )

    def normal2beads(self, x_normal):
        """
        Transform a system tensor (e.g. momenta, positions) in normal mode representation back to bead representation.

        Args:
            x_normal (torch.Tensor): System tensor in normal mode representation with the general shape
                                    n_beads x n_molecules x ...

        Returns:
            torch.Tensor: System tensor in bead representation with the same shape as the input tensor.
        """
        return torch.mm(
            self.c_transform.transpose(0, 1), x_normal.view(self.n_beads, -1)
        ).view(x_normal.shape)

import numpy as np
import torch
from ase import units


class MDUnits:
    """
    Basic conversion factors to atomic units used internally:
        fs2atu (time): femtoseconds to atomic time units
        eV2Ha (energy): electron Volt to Hartree
        d2amu (mass): Dalton to atomic mass units
        angs2bohr (length): Angstrom to Bohr
        auforces2aseforces (forces): Hartee per Bohr to electron Volt per Angstrom

    Definitions for constants:
        kB: Boltzmann constant in units of Hartree per Kelvin.
        hbar: Reduced Planck constant in atomic units.
    """
    # Unit conversions
    fs2atu = 1e-15 / units._aut
    eV2Ha = 1.0 / units.Ha
    d2amu = units._amu / units._me
    angs2bohr = 1.0 / units.Bohr
    auforces2aseforces = angs2bohr / eV2Ha

    # Constants
    kB = units.kB / units.Ha
    hbar = 1.0


class YSWeights:
    """
    Weights for Yoshida-Suzuki integration used in propagating the Nose-Hoover chain thermostats.

    Args:
        device (str): Device used for computation (default='cuda').
    """
    YS_weights = {3: np.array([1.35120719195966,
                               -1.70241438391932,
                               1.35120719195966]),
                  5: np.array([0.41449077179438,
                               0.41449077179438,
                               -0.65796308717750,
                               0.41449077179438,
                               0.41449077179438]),
                  7: np.array([-1.17767998417887,
                               0.23557321335936,
                               0.78451361047756,
                               1.31518632068390,
                               0.78451361047756,
                               0.23557321335936,
                               -1.17767998417887])}

    def __init__(self, device):
        self.device = device

    def get_weights(self, order):
        """
        Get the weights required for an integration scheme of the desired order.

        Args:
            order (int): Desired order of the integration scheme.

        Returns:
            torch.Tensor: Tensor of the integration weights
        """
        if order not in self.YS_weights:
            raise ValueError('Order {:d} not supported for YS integration weights'.format(order))
        else:
            ys_weights = torch.from_numpy(self.YS_weights[order]).float().to(self.device)
        return ys_weights


def compute_centroid(ensemble):
    """
    Compute centroids of the system properties (e.g. positions, momenta) given in ensemble with respect to the replica
    dimension (0). The original dimensionality of the tensor is kept for the purpose of broadcasting and logging. This
    routine is primarily intended to be used for ring polymer simulations.

    Args:
        ensemble (torch.Tensor): System property tensor (e.g. positions, momenta) of the general dimension
                                 n_replicas x n_molecules x ...

    Returns:
        torch.Tensor: Centroid averaged over the replica dimension with the general shape 1 x n_molecules x ...
    """
    centroid = torch.mean(ensemble, 0, keepdim=True)
    return centroid


def batch_inverse(tensor):
    """
    Compute the matrix inverse of a batch of square matrices. This routine is used for removing rotational motion
    during the molecular dynamics simulation. Taken from https://stackoverflow.com/questions/46595157

    Args:
        tensor (torch.Tensor):  Tensor of square matrices with the shape n_batch x dim1 x dim1

    Returns:
        torch.Tensor: Tensor of the inverted square matrices with the same shape as the input tensor.
    """
    eye = tensor.new_ones(tensor.size(-1), device=tensor.device).diag().expand_as(tensor)
    tensor_inv, _ = torch.gesv(eye, tensor)
    return tensor_inv


def load_gle_matrices(filename):
    """
    Load GLE thermostat files formatted in raw format as generated via http://gle4md.org/index.html?page=matrix
    The generated matrices are torch tensors of the shape normal_modes x s+1 x s+1, where normal_modes is 1 except in
    the case of the PIGLET thermostat and s is the number of degrees of freedom added via GLE. Automatically recognizes
    used units and converts them to atomic units.

    Args:
        filename (str): Path to the file the GLE thermostat parameters should be loaded from.

    Returns:
        tuple: Tuple of two square torch tensors containing the a_matrix and c_matrix parameters required to
               initialize GLE type thermostats.
    """
    a_matrix = GLEMatrixParser('A MATRIX:', stop='C MATRIX:', split='Matrix for normal mode')
    c_matrix = GLEMatrixParser('C MATRIX:', split='Matrix for normal mode')

    try:
        with open(filename) as glefile:
            for line in glefile:
                a_matrix.read_line(line)
                c_matrix.read_line(line)
    except FileNotFoundError:
        raise FileNotFoundError('Could not open {:s} for reading. Please use GLE parameter files '
                                'generated via http://gle4md.org/index.html?page=matrix'.format(filename))

    return a_matrix.matrix, c_matrix.matrix


class GLEMatrixParser:
    """
    General parser for GLE thermostat files. Reads from start string until end of file or a given stop string. If the
    argument split is specified, the read matrices are split at the given token.  Automatically recognizes
    used units and converts them to atomic units.

    Args:
        start (str): Token when to start reading.
        stop (str): Token when to stop reading. If None (default) reads until eno of file.
        split (str): If the given token is encountered, matrices are split at this point. If None (default), no split is
                     performed.
    """
    # Automatically recognized format and converts to units
    unit_conversions = {
        'atomic time units^-1': 1,
        'picoseconds^-1': 1 / 1000 / MDUnits.fs2atu,
        'seconds^-1': units._aut,
        'femtoseconds^-1': 1 / MDUnits.fs2atu,
        'eV': 1 / units.Ha,
        'atomic energy units': 1,
        'K': MDUnits.kB
    }

    def __init__(self, start, stop=None, split=None):
        self.start = start
        self.stop = stop
        self.split = split
        self.read = False
        self.units = None
        self._matrix = []
        self._tmp_matrix = []

    def read_line(self, line):
        """
        Read and parse a line obtained from an open file object containing GLE parameters.

        Args:
            line (str): Line of a GLE parameter file.
        """
        line = line.strip()
        # Filter for empty lines
        if line:
            # Check if start token is present
            if self.start in line:
                self.read = True
                # Get units used
                unit_name = line.split('(')[-1].replace(')', '')
                self.units = self.unit_conversions[unit_name]
            elif self.read:
                if line.startswith('#'):
                    # Check for stop and split tokens
                    if self.stop is not None and self.stop in line:
                        self.read = False
                    if self.split is not None and self.split in line:
                        if len(self._tmp_matrix) > 0:
                            self._matrix.append(self._tmp_matrix)
                            self._tmp_matrix = []
                else:
                    # Otherwise read and parse line
                    self._tmp_matrix.append([float(x) for x in line.split()])

    @property
    def matrix(self):
        """
        Property to get parsed matrices converted to numpy arrays using atomic units.

        Returns:
            numpy.array: Array of the parsed GLE matrix with the shape normal_modes x s+1 x s+1, where normal_modes is 1
                         except in the case of the PIGLET thermostat and s is the number of degrees of freedom added via
                         GLE. If no matrix is found, None is returned.
        """
        # Write out last buffer
        if len(self._tmp_matrix) > 0:
            self._matrix.append(self._tmp_matrix)
        # Convert to numpy array
        _matrix = np.array(self._matrix)
        # Perform unit conversion
        if self.units is not None:
            return _matrix * self.units
        else:
            return None


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

    def __init__(self, n_beads, device='cuda'):
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
        return torch.mm(self.c_transform, x_beads.view(self.n_beads, -1)).view(x_beads.shape)

    def normal2beads(self, x_normal):
        """
        Transform a system tensor (e.g. momenta, positions) in normal mode representation back to bead representation.

        Args:
            x_normal (torch.Tensor): System tensor in normal mode representation with the general shape
                                    n_beads x n_molecules x ...

        Returns:
            torch.Tensor: System tensor in bead representation with the same shape as the input tensor.
        """
        return torch.mm(self.c_transform.transpose(0, 1), x_normal.view(self.n_beads, -1)).view(x_normal.shape)


class RunningAverage:
    """
    Running average class for logging purposes. Accumulates the average of a given tensor over the course of the
    simulation.
    """

    def __init__(self):
        # Initialize running average and item count
        self.average = 0
        self.counts = 0

    def update(self, value):
        """
        Update the running average.

        Args:
            value (torch.Tensor): Tensor containing the property whose average should be accumulated.
        """
        self.average = (self.counts * self.average + value) / (self.counts + 1)
        self.counts += 1

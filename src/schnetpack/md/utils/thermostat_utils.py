import numpy as np
import torch
from ase import units

from schnetpack.md.utils import MDUnits

# __all__ = [
#     "load_gle_matrices",
#     "GLEMatrixParser",
#     "YSWeights"
# ]


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
    a_matrix = GLEMatrixParser(
        "A MATRIX:", stop="C MATRIX:", split="Matrix for normal mode"
    )
    c_matrix = GLEMatrixParser("C MATRIX:", split="Matrix for normal mode")

    try:
        with open(filename) as glefile:
            for line in glefile:
                a_matrix.read_line(line)
                c_matrix.read_line(line)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not open {:s} for reading. Please use GLE parameter files "
            "generated via http://gle4md.org/index.html?page=matrix".format(filename)
        )

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
        "atomic time units^-1": 1,
        "picoseconds^-1": 1 / 1000 / MDUnits.fs2atu,
        "seconds^-1": units._aut,
        "femtoseconds^-1": 1 / MDUnits.fs2atu,
        "eV": 1 / units.Ha,
        "atomic energy units": 1,
        "K": MDUnits.kB,
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
                unit_name = line.split("(")[-1].replace(")", "")
                self.units = self.unit_conversions[unit_name]
            elif self.read:
                if line.startswith("#"):
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


class YSWeights:
    """
    Weights for Yoshida-Suzuki integration used in propagating the Nose-Hoover chain thermostats.

    Args:
        device (str): Device used for computation (default='cuda').
    """

    YS_weights = {
        3: np.array([1.35120719195966, -1.70241438391932, 1.35120719195966]),
        5: np.array(
            [
                0.41449077179438,
                0.41449077179438,
                -0.65796308717750,
                0.41449077179438,
                0.41449077179438,
            ]
        ),
        7: np.array(
            [
                -1.17767998417887,
                0.23557321335936,
                0.78451361047756,
                1.31518632068390,
                0.78451361047756,
                0.23557321335936,
                -1.17767998417887,
            ]
        ),
    }

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
            raise ValueError(
                "Order {:d} not supported for YS integration weights".format(order)
            )
        else:
            ys_weights = (
                torch.from_numpy(self.YS_weights[order]).float().to(self.device)
            )
        return ys_weights

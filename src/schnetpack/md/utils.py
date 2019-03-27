import numpy as np
import torch
from ase import units
import h5py
import logging
import json
from schnetpack.data import Structure


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
    eV2Ha = units.eV / units.Ha
    d2amu = units._amu / units._me
    angs2bohr = units.Angstrom / units.Bohr
    auforces2aseforces = angs2bohr / eV2Ha

    # Constants
    kB = units.kB / units.Ha
    hbar = 1.0

    # Conversion units use when reading in MD inputs
    conversions = {
        "kcal": units.kcal / units.Hartree,
        "mol": 1 / units.mol,
        "ev": units.eV / units.Hartree,
        "bohr": 1.0,
        "angstrom": units.Bohr,
        "a": units.Bohr,
        "angs": units.Bohr,
        "hartree": 1.0,
        "ha": 1.0,
    }

    @staticmethod
    def parse_mdunit(unit):
        """
        Auxiliary functions, used to determine the conversion factor of position and force units between MD propagation
        and the provided ML Calculator. Allowed units are:
            mol, kcal, eV, Bohr, Angstrom, Hartree and all combinations using '/' thereof (e.g. kcal/mol/Angstrom).

        Args:
            unit (str/float): Unit to be used to convert forces from Calculator units to atomic units used in the MD.
                              Can be a str, which is converted to the corresponding numerical value or a float, which
                              is returned.

        Returns:
            float: Factor used for conversion in the Calculator.

        """
        if type(unit) == str:
            parts = unit.lower().replace(" ", "").split("/")
            scaling = 1.0
            for part in parts:
                if part not in MDUnits.conversions:
                    raise KeyError("Unrecognized unit {:s}".format(part))
                scaling *= MDUnits.conversions[part]
            return scaling
        else:
            return unit


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
    eye = (
        tensor.new_ones(tensor.size(-1), device=tensor.device).diag().expand_as(tensor)
    )
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


class HDF5LoaderError(Exception):
    """
    Exception for HDF5 loader class.
    """

    pass


class HDF5Loader:
    """
    Class for loading HDF5 datasets written by the FileLogger. By default, this requires at least a MoleculeSteam to be
    present. PropertyData is also read by default, but can be disabled.

    Args:
        hdf5_database (str): Path to the database file.
        skip_initial (int): Skip the initial N configurations in the trajectory, e.g. to account for equilibration
                            (default=0).
        load_properties (bool): Extract and reconstruct the property data stored by a PropertyStream (e.g. forces,
                                energies, etc.), enabled by default.
    """

    def __init__(self, hdf5_database, skip_initial=0, load_properties=True):
        self.database = h5py.File(hdf5_database, "r", swmr=True, libver="latest")
        self.skip_initial = skip_initial
        self.data_groups = list(self.database.keys())

        self.properties = {}

        # Load basic structure properties and MD info
        if "molecules" not in self.data_groups:
            raise HDF5LoaderError(
                "Molecule data not found in {:s}".format(hdf5_database)
            )
        else:
            self._load_molecule_data()

        # If requested, load other properties predicted by the model stored via PropertyStream
        if load_properties:
            if "properties" not in self.data_groups:
                raise HDF5LoaderError(
                    "Molecule properties not found in {:s}".format(hdf5_database)
                )
            else:
                self._load_properties()

        # Do formatting for info
        loaded_properties = list(self.properties.keys())
        if len(loaded_properties) == 1:
            loaded_properties = str(loaded_properties[0])
        else:
            loaded_properties = (
                ", ".join(loaded_properties[:-1]) + " and " + loaded_properties[-1]
            )

        logging.info(
            "Loaded properties {:s} from {:s}".format(loaded_properties, hdf5_database)
        )

    def _load_molecule_data(self):
        """
        Load data stored by a MoleculeStream. This contains basic information on the system and is required to be
        present for the Loader.
        """
        # This is for molecule streams
        structures = self.database["molecules"]

        # General database info
        # TODO: Could be moved to global attrs if available
        self.n_replicas = structures.attrs["n_replicas"]
        self.n_molecules = structures.attrs["n_molecules"]
        self.n_atoms = structures.attrs["n_atoms"]
        self.entries = structures.attrs["entries"]
        self.time_step = structures.attrs["time_step"]

        # Write to main property dictionary
        self.properties[Structure.Z] = structures.attrs["atom_types"][0, ...]
        self.properties[Structure.R] = structures[
            self.skip_initial : self.entries, ..., :3
        ]
        self.properties["velocities"] = structures[
            self.skip_initial : self.entries, ..., 3:
        ]

    def _load_properties(self):
        """
        Load properties and their shape from the corresponding group in the hdf5 database.
        Properties are then reshaped to original form and stores in the self.properties dictionary.
        """
        # And for property stream
        properties = self.database["properties"]
        property_shape = json.loads(properties.attrs["shapes"])
        property_positions = json.loads(properties.attrs["positions"])

        # Reformat properties
        for prop in property_positions:
            prop_pos = slice(*property_positions[prop])
            self.properties[prop] = properties[
                self.skip_initial : self.entries, :, :, prop_pos
            ].reshape(
                (
                    self.entries - self.skip_initial,
                    self.n_replicas,
                    self.n_molecules,
                    *property_shape[prop],
                )
            )

    def get_property(self, property_name, mol_idx=0, replica_idx=None, atomistic=False):
        """
        Extract property from dataset.

        Args:
            property_name (str): Name of the property as contained in the self.properties dictionary.
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.
            atomistic (bool): Whether the property is atomistic (e.g. forces) or defined for the whole molecule
                              (e.g. energies, dipole moments). If set to True, the array is masked according to the
                               number of atoms for the requested molecule to counteract potential zero-padding.
                               (default=False)

        Returns:
            np.array: N_steps x [ property dimensions... ] array containing the requested property collected during the
                      simulation.
        """

        # Special case for atom types
        if property_name == Structure.Z:
            return self.properties[Structure.Z][mol_idx]

        # Check whether property is present
        if property_name not in self.properties:
            raise HDF5LoaderError(f"Property {property_name} not found in database.")

        # Mask by number of atoms if property is declared atomistic
        if atomistic:
            n_atoms = self.n_atoms[mol_idx]
            target_property = self.properties[property_name][
                :, :, mol_idx, :n_atoms, ...
            ]
        else:
            target_property = self.properties[property_name][:, :, mol_idx, ...]

        # Compute the centroid unless requested otherwise
        if replica_idx is None:
            target_property = np.mean(target_property, axis=1)
        else:
            target_property = target_property[:, replica_idx, ...]

        return target_property

    def get_velocities(self, mol_idx=0, replica_idx=None):
        """
        Auxiliary routine for getting the velocities of specific molecules and replicas.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x N_atoms x 3 array containing the atom velocities of the simulation in atomic units.
        """
        return self.get_property(
            "velocities", mol_idx=mol_idx, replica_idx=replica_idx, atomistic=True
        )

    def get_positions(self, mol_idx=0, replica_idx=None):
        """
        Auxiliary routine for getting the positions of specific molecules and replicas.

        Args:
            mol_idx (int): Index of the molecule to extract, by default uses the first molecule (mol_idx=0)
            replica_idx (int): Replica of the molecule to extract (e.g. for ring polymer molecular dynamics). If
                               replica_idx is set to None (default), the centroid is returned if multiple replicas are
                               present.

        Returns:
            np.array: N_steps x N_atoms x 3 array containing the atom positions of the simulation in atomic units.
        """
        return self.get_property(
            Structure.R, mol_idx=mol_idx, replica_idx=replica_idx, atomistic=True
        )

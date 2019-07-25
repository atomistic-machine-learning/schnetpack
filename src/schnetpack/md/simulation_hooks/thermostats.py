"""
This module contains various thermostats for regulating the temperature of the system during
molecular dynamics simulations. Apart from standard thermostats for convetional simulations,
a series of special thermostats developed for ring polymer molecular dynamics is also provided.
"""
import torch
import numpy as np
import scipy.linalg as linalg
import logging

from schnetpack.md.utils import (
    MDUnits,
    load_gle_matrices,
    NormalModeTransformer,
    YSWeights,
)
from schnetpack.md.integrators import RingPolymer
from schnetpack.md.simulation_hooks.basic_hooks import SimulationHook

__all__ = [
    "ThermostatHook",
    "BerendsenThermostat",
    "GLEThermostat",
    "PIGLETThermostat",
    "LangevinThermostat",
    "PILELocalThermostat",
    "PILEGlobalThermostat",
    "NHCThermostat",
    "NHCRingPolymerThermostat",
    "TRPMDThermostat",
]


class ThermostatError(Exception):
    """
    Exception for thermostat class.
    """

    pass


class ThermostatHook(SimulationHook):
    """
    Basic thermostat hook for simulator class. This class is initialized based on the simulator and system
    specifications during the first MD step. Thermostats are applied before and after each MD step.

    Args:
        temperature_bath (float): Temperature of the heat bath in Kelvin.
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module use dto transform between beads and normal
                                                                       model representation in ring polymer dynamics.
        detach (bool): Whether the computational graph should be detached after each simulation step. Default is true,
                       should be changed if differentiable MD is desired.
                       TODO: Make detach frequency instead
    """

    # TODO: Could be made a torch nn.Module

    def __init__(self, temperature_bath, nm_transformation=None, detach=True):
        self.temperature_bath = temperature_bath
        self.initialized = False
        self.device = None
        self.n_replicas = None
        self.nm_transformation = nm_transformation
        self.detach = detach

    def on_simulation_start(self, simulator):
        """
        Routine to initialize the thermostat based on the current state of the simulator. Reads the device to be uses,
        as well as the number of molecular replicas present in simulator.system. Furthermore, the normal mode
        transformer is initialized during ring polymer simulations. In addition, a flag is set so that the thermostat
        is not reinitialized upon continuation of the MD.

        Main function is the _init_thermostat routine, which takes the simulator as input and must be provided for every
        new thermostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        self.device = simulator.system.device
        self.n_replicas = simulator.system.n_replicas

        # Check if using normal modes is feasible and initialize
        if self.nm_transformation is not None:
            if type(simulator.integrator) is not RingPolymer:
                raise ThermostatError(
                    "Normal mode transformation should only"
                    "be used with ring polymer dynamics."
                )
            else:
                # If simulation is not restarted from a previous point, initialize.
                if simulator.effective_steps == 0:
                    self.nm_transformation = self.nm_transformation(
                        self.n_replicas, device=self.device
                    )

        if not self.initialized:
            self._init_thermostat(simulator)
            self.initialized = True

    def on_step_begin(self, simulator):
        """
        First application of the thermostat befor the first half step of the dynamics. Regulates temperature and applies
        a mask to the system momenta in order to avoid problems of e.g. thermal noise added to the zero padded tensors.
        The detach is carried out here.

        Main function is the _apply_thermostat routine, which takes the simulator as input and must be provided for
        every new thermostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Apply thermostat
        self._apply_thermostat(simulator)

        # Re-apply atom masks for differently sized molecules, as some
        # thermostats add random noise
        simulator.system.momenta = (
            simulator.system.momenta * simulator.system.atom_masks
        )

        # Detach if requested
        if self.detach:
            simulator.system.momenta = simulator.system.momenta.detach()

    def on_step_end(self, simulator):
        """
        First application of the thermostat befor the first half step of the dynamics. Regulates temperature and applies
        a mask to the system momenta in order to avoid problems of e.g. thermal noise added to the zero padded tensors.
        The detach is carried out here.

        Main function is the _apply_thermostat routine, which takes the simulator as input and must be provided for
        every new thermostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Apply thermostat
        self._apply_thermostat(simulator)

        # Re-apply atom masks for differently sized molecules, as some
        # thermostats add random noise
        simulator.system.momenta = (
            simulator.system.momenta * simulator.system.atom_masks
        )

        # Detach if requested
        if self.detach:
            simulator.system.momenta = simulator.system.momenta.detach()

    def _init_thermostat(self, simulator):
        """
        Dummy routine for initializing a thermostat based on the current simulator. Should be implemented for every
        new thermostat. Has access to the information contained in the simulator class, e.g. number of replicas, time
        step, masses of the atoms, etc.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        pass

    def _apply_thermostat(self, simulator):
        """
        Dummy routine for applying the thermostat to the system. Should use the implemented thermostat to update the
        momenta of the system contained in simulator.system.momenta. Is called twice each simulation time step.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        raise NotImplementedError


class BerendsenThermostat(ThermostatHook):
    """
    Berendsen velocity rescaling thermostat, as described in [#berendsen1]_. Simple thermostat for e.g. equilibrating
    the system, does not sample the canonical ensemble.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs

    References
    ----------
    .. [#berendsen1] Berendsen, Postma, van Gunsteren, DiNola, Haak:
       Molecular dynamics with coupling to an external bath.
       The Journal of Chemical Physics, 81 (8), 3684-3690. 1984.
    """

    def __init__(self, temperature_bath, time_constant):
        super(BerendsenThermostat, self).__init__(temperature_bath)

        self.time_constant = time_constant * MDUnits.fs2atu

    def _apply_thermostat(self, simulator):
        """
        Apply the Berendsen thermostat via rescaling the systems momenta based on the current instantaneous temperature
        and the bath temperature.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        scaling = 1.0 + simulator.integrator.time_step / self.time_constant * (
            self.temperature_bath / simulator.system.temperature - 1
        )
        simulator.system.momenta = (
            torch.sqrt(scaling[:, :, None, None]) * simulator.system.momenta
        )


class GLEThermostat(ThermostatHook):
    """
    Stochastic generalized Langevin colored noise thermostat by Ceriotti et. al. as described in [#gle_thermostat1]_.
    This thermostat requires specially parametrized matrices, which can be obtained online from:
    http://gle4md.org/index.html?page=matrix

    The additional degrees of freedom added to the system are defined via the matrix dimensions. This could in principle
    be used for ring polymer dynamics by providing a normal mode transformation.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        gle_file (str): File containing the GLE matrices
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module use dto transform between beads and normal
                                                                       model representation in ring polymer dynamics.

    References
    ----------
    .. [#gle_thermostat1] Ceriotti, Bussi, Parrinello:
       Colored-noise thermostats à la carte.
       Journal of Chemical Theory and Computation 6 (4), 1170-1180. 2010.
    """

    def __init__(self, temperature_bath, gle_file, nm_transformation=None):
        super(GLEThermostat, self).__init__(
            temperature_bath, nm_transformation=nm_transformation
        )

        self.gle_file = gle_file

        # To be initialized on beginning of the simulation, once system and
        # integrator are known
        self.c1 = None
        self.c2 = None
        self.thermostat_momenta = None
        self.thermostat_factor = None

    def _init_thermostat(self, simulator):
        """
        Initialize the GLE thermostat by reading in the the required matrices and setting up the initial random
        thermostat momenta and the mass factor.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Generate main matrices
        self.c1, self.c2 = self._init_gle_matrices(simulator)

        # Get particle masses
        self.thermostat_factor = torch.sqrt(simulator.system.masses)[..., None]

        # Get initial thermostat momenta
        self.thermostat_momenta = self._init_thermostat_momenta(simulator)

    def _init_gle_matrices(self, simulator):
        """
        Read all GLE matrices from a file and check, whether they have the right dimensions.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        a_matrix, c_matrix = load_gle_matrices(self.gle_file)

        if a_matrix is None:
            raise ThermostatError(
                "Error reading GLE matrices" " from {:s}".format(self.gle_file)
            )
        elif a_matrix.shape[0] > 1:
            raise ThermostatError(
                "More than one A matrix found. Could be " "PIGLET input."
            )
        else:
            # Remove leading dimension (for normal modes)
            a_matrix = a_matrix.squeeze()

        c1, c2 = self._init_single_gle_matrix(a_matrix, c_matrix, simulator)
        return c1, c2

    def _init_single_gle_matrix(self, a_matrix, c_matrix, simulator):
        """
        Based on the matrices found in the GLE file, initialize the GLE matrices required for a simulation with the
        thermostat. See [#stochastic_thermostats1]_ for more detail. The dimensions of all matrices are:
        degrees_of_freedom x degrees_of_freedom,
        where degrees_of_freedom are the degrees of freedom of the extended system.

        Args:
            a_matrix (np.array): Raw matrices containing friction friction acting on system (drift matrix).
            c_matrix (np.array): Raw matrices modulating the intensity of the random force (diffusion matrix).
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.

        Returns:
            torch.Tensor: Drift matrix for simulation.
            torch.Tensor: Diffusion matrix initialized for simulation.

        References
        ----------
        .. [#stochastic_thermostats1]_Ceriotti, Parrinello, Markland, Manolopoulos:
           Efficient stochastic thermostatting of path integral molecular dynamics.
           The Journal of Chemical Physics, 133 (12), 124104. 2010.
        """
        if c_matrix is None:
            c_matrix = np.eye(a_matrix.shape[-1]) * self.temperature_bath * MDUnits.kB
            # Check if normal GLE or GLE for ring polymers is needed:
            if type(simulator.integrator) is RingPolymer:
                logging.info(
                    "RingPolymer integrator detected, initializing " "C accordingly."
                )
                c_matrix *= simulator.system.n_replicas
        else:
            c_matrix = c_matrix.squeeze()
            logging.info(
                "C matrix for GLE loaded, provided temperature will " "be ignored."
            )

        # A does not need to be transposed, else c2 is imaginary
        c1 = linalg.expm(-0.5 * simulator.integrator.time_step * a_matrix)

        # c2 is symmetric
        c2 = linalg.sqrtm(c_matrix - np.dot(c1, np.dot(c_matrix, c1.T)))

        # To myself: original expression is c1 = exp(-dt/2 * a.T)
        # the C1 here is c1.T, since exp(-dt/2*a.T).T = exp(-dt/2*a)
        # The formula for c2 is c2 = sqrtm(1-c1.T*c1)
        # In our case, this becomes sqrtm(1-C1*C1.T)
        # For the propagation we have the original expression c1*p, where
        # p is a column vector (ndegrees x something)
        # In our case P is (something x ndegrees), hence p.T
        # The propagation then becomes P*C1 = p.T*c1.T = (c1*p).T
        # c2 is symmetric by construction, hence C2=c2
        c1 = torch.from_numpy(c1).to(self.device).float()
        c2 = torch.from_numpy(c2).to(self.device).float()
        return c1, c2

    def _init_thermostat_momenta(self, simulator, free_particle_limit=True):
        """
        Initialize the thermostat momenta tensor based on the system specifications. This tensor is then updated
        during the GLE dynamics.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
            free_particle_limit (bool): Initialize momenta according to free particle limit instead of a zero matrix
                                        (default=True).

        Returns:
            torch.Tensor: Initialized random momenta of the extended system with the dimension:
                          n_replicas x n_molecules x n_atoms x 3 x degrees_of_freedom
        """
        degrees_of_freedom = self.c1.shape[-1]
        if not free_particle_limit:
            initial_momenta = torch.zeros(
                *simulator.system.momenta.shape, degrees_of_freedom, device=self.device
            )
        else:
            initial_momenta = torch.randn(
                *simulator.system.momenta.shape, degrees_of_freedom, device=self.device
            )
            initial_momenta = torch.matmul(initial_momenta, self.c2)
        return initial_momenta

    def _apply_thermostat(self, simulator):
        """
        Perform the update of the system momenta according to the GLE thermostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Generate random noise
        thermostat_noise = torch.randn(
            self.thermostat_momenta.shape, device=self.device
        )

        # Get current momenta
        momenta = simulator.system.momenta

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.beads2normal(momenta)

        # Set current momenta
        self.thermostat_momenta[:, :, :, :, 0] = momenta

        # Apply thermostat
        self.thermostat_momenta = (
            torch.matmul(self.thermostat_momenta, self.c1)
            + torch.matmul(thermostat_noise, self.c2) * self.thermostat_factor
        )

        # Extract momenta
        momenta = self.thermostat_momenta[:, :, :, :, 0]

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.normal2beads(momenta)

        simulator.system.momenta = momenta

    @property
    def state_dict(self):
        state_dict = {
            "c1": self.c1,
            "c2": self.c2,
            "thermostat_factor": self.thermostat_factor,
            "thermostat_momenta": self.thermostat_momenta,
            "temperature_bath": self.temperature_bath,
            "n_replicas": self.n_replicas,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.c1 = state_dict["c1"]
        self.c2 = state_dict["c2"]
        self.thermostat_factor = state_dict["thermostat_factor"]
        self.thermostat_momenta = state_dict["thermostat_momenta"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.n_replicas = state_dict["n_replicas"]

        # Set initialized flag
        self.initialized = True


class PIGLETThermostat(GLEThermostat):
    """
    Efficient generalized Langevin equation stochastic thermostat for ring polymer dynamics simulations, see
    [#piglet_thermostat1]_ for a detailed description. In contrast to the standard GLE thermostat, every normal mode
    of the ring polymer is
    thermostated seperately.


    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        gle_file (str): File containing the GLE matrices
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module use dto transform between beads and normal
                                                                       model representation in ring polymer dynamics.

    References
    ----------
    .. [#piglet_thermostat1] Uhl, Marx, Ceriotti:
       Accelerated path integral methods for atomistic simulations at ultra-low temperatures.
       The Journal of chemical physics, 145(5), 054101. 2016.
    """

    def __init__(
        self, temperature_bath, gle_file, nm_transformation=NormalModeTransformer
    ):

        logging.info("Using PIGLET thermostat")
        super(PIGLETThermostat, self).__init__(
            temperature_bath, gle_file, nm_transformation=nm_transformation
        )

    def _init_gle_matrices(self, simulator):
        """
        Initialize the matrices necessary for the PIGLET thermostat. In contrast to the basic GLE thermostat, these
        have the dimension:
        n_replicas x degrees_of_freedom x degrees_of_freedom,
        where n_replicas is the number of beads in the ring polymer and degrees_of_freedom is the number of degrees of
        freedom introduced by GLE.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.

        Returns:
            torch.Tensor: Drift matrices for the PIGLET thermostat.
            torch.Tensor: Diffusion matrices.
        """
        a_matrix, c_matrix = load_gle_matrices(self.gle_file)

        if a_matrix is None:
            raise ThermostatError(
                "Error reading GLE matrices " "from {:s}".format(self.gle_file)
            )
        if a_matrix.shape[0] != self.n_replicas:
            raise ThermostatError(
                "Expected {:d} beads but "
                "found {:d}.".format(self.n_replicas, a_matrix.shape[0])
            )

        if not type(simulator.integrator) is RingPolymer:
            raise ThermostatError("PIGLET thermostat should only be used with " "RPMD.")

        all_c1 = []
        all_c2 = []

        # Generate main matrices
        for b in range(self.n_replicas):
            c1, c2 = self._init_single_gle_matrix(
                a_matrix[b], (c_matrix[b], None)[c_matrix is None], simulator
            )
            # Add extra dimension for use with torch.cat, correspond to normal
            # modes of ring polymer
            all_c1.append(c1[None, ...])
            all_c2.append(c2[None, ...])

        # Bring to correct shape for later matmul broadcasting
        c1 = torch.cat(all_c1)[:, None, None, :, :]
        c2 = torch.cat(all_c2)[:, None, None, :, :]
        return c1, c2


class LangevinThermostat(ThermostatHook):
    """
    Basic stochastic Langevin thermostat, see e.g. [#langevin_thermostat1]_ for more details.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module use dto transform between beads and normal
                                                                       model representation in ring polymer dynamics.

    References
    ----------
    .. [#langevin_thermostat1] Bussi, Parrinello:
       Accurate sampling using Langevin dynamics.
       Physical Review E, 75(5), 056707. 2007.
    """

    def __init__(self, temperature_bath, time_constant, nm_transformation=None):

        logging.info("Using Langevin thermostat")
        super(LangevinThermostat, self).__init__(
            temperature_bath, nm_transformation=nm_transformation
        )

        self.time_constant = time_constant * MDUnits.fs2atu

        self.thermostat_factor = None
        self.c1 = None
        self.c2 = None

    def _init_thermostat(self, simulator):
        """
        Initialize the Langevin coefficient matrices based on the system and simulator properties.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Initialize friction coefficients
        gamma = torch.ones(1, device=self.device) / self.time_constant

        # Initialize coefficient matrices
        c1 = torch.exp(-0.5 * simulator.integrator.time_step * gamma)
        c2 = torch.sqrt(1 - c1 ** 2)

        self.c1 = c1.to(self.device)[:, None, None, None]
        self.c2 = c2.to(self.device)[:, None, None, None]

        # Get mass and temperature factors
        self.thermostat_factor = torch.sqrt(
            simulator.system.masses * MDUnits.kB * self.temperature_bath
        )

    def _apply_thermostat(self, simulator):
        """
        Apply the stochastic Langevin thermostat to the systems momenta.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Get current momenta
        momenta = simulator.system.momenta

        # Apply transformation
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.beads2normal(momenta)

        # Generate random noise
        thermostat_noise = torch.randn(momenta.shape, device=self.device)

        # Apply thermostat
        momenta = (
            self.c1 * momenta + self.thermostat_factor * self.c2 * thermostat_noise
        )

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.normal2beads(momenta)

        simulator.system.momenta = momenta

    @property
    def state_dict(self):
        state_dict = {
            "c1": self.c1,
            "c2": self.c2,
            "thermostat_factor": self.thermostat_factor,
            "temperature_bath": self.temperature_bath,
            "n_replicas": self.n_replicas,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.c1 = state_dict["c1"]
        self.c2 = state_dict["c2"]
        self.thermostat_factor = state_dict["thermostat_factor"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.n_replicas = state_dict["n_replicas"]

        # Set initialized flag
        self.initialized = True


class PILELocalThermostat(LangevinThermostat):
    """
    Langevin thermostat for ring polymer molecular dynamics as introduced in [#stochastic_thermostats2]_.
    Applies specially initialized Langevin thermostats to the beads of the ring polymer in normal mode representation.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module use dto transform between beads and normal
                                                                       model representation in ring polymer dynamics.
        thermostat_centroid (bool): Whether a thermostat should be applied to the centroid of the ring polymer in
                                    normal mode representation (relevant e.g. for TRPMD, default is True)
        damping (float): If specified, damping factor is applied to the current momenta of the system (used in TRPMD,
                         default is no damping).

    References
    ----------
    .. [#stochastic_thermostats2] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133 (12), 124104. 2010.
    """

    def __init__(
        self,
        temperature_bath,
        time_constant,
        nm_transformation=NormalModeTransformer,
        thermostat_centroid=True,
        damping=None,
    ):
        super(PILELocalThermostat, self).__init__(
            temperature_bath, time_constant, nm_transformation=nm_transformation
        )
        self.thermostat_centroid = thermostat_centroid
        self.damping = damping

    def _init_thermostat(self, simulator):
        """
        Initialize the Langevin matrices based on the normal mode frequencies of the ring polymer. If the centroid is to
        be thermostatted, the suggested value of 1/time_constant is used.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        if type(simulator.integrator) is not RingPolymer:
            raise ThermostatError("PILE thermostats can only be used in RPMD")

        # Initialize friction coefficients
        gamma_normal = 2 * simulator.integrator.omega_normal

        # Use seperate coefficient for centroid mode (default, unless using thermostatted RPMD)
        if self.thermostat_centroid:
            gamma_normal[0] = 1.0 / self.time_constant

        # Apply TRPMD damping factor if provided
        if self.damping is not None:
            gamma_normal *= self.damping

        if self.nm_transformation is None:
            raise ThermostatError(
                "Normal mode transformation required for " "PILE thermostat"
            )

        # Initialize coefficient matrices
        c1 = torch.exp(-0.5 * simulator.integrator.time_step * gamma_normal)
        c2 = torch.sqrt(1 - c1 ** 2)

        self.c1 = c1.to(self.device)[:, None, None, None]
        self.c2 = c2.to(self.device)[:, None, None, None]

        # Get mass and temperature factors
        self.thermostat_factor = torch.sqrt(
            simulator.system.masses
            * MDUnits.kB
            * self.n_replicas
            * self.temperature_bath
        )

    @property
    def state_dict(self):
        state_dict = {
            "c1": self.c1,
            "c2": self.c2,
            "thermostat_factor": self.thermostat_factor,
            "temperature_bath": self.temperature_bath,
            "n_replicas": self.n_replicas,
            "damping": self.damping,
            "thermostat_centroid": self.thermostat_centroid,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.c1 = state_dict["c1"]
        self.c2 = state_dict["c2"]
        self.thermostat_factor = state_dict["thermostat_factor"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.n_replicas = state_dict["n_replicas"]
        self.damping = state_dict["damping"]
        self.thermostat_centroid = state_dict["thermostat_centroid"]

        # Set initialized flag
        self.initialized = True


class PILEGlobalThermostat(PILELocalThermostat):
    """
    Global variant of the ring polymer Langevin thermostat as suggested in [#stochastic_thermostats3]_. This thermostat
    applies a stochastic velocity rescaling thermostat [#stochastic_velocity_rescaling1]_ to the ring polymer centroid
    in normal mode representation.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module use dto transform between beads and normal
                                                                       model representation in ring polymer dynamics.

    References
    ----------
    .. [#stochastic_thermostats3] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133 (12), 124104. 2010.
    .. [#stochastic_velocity_rescaling1] Bussi, Donadio, Parrinello:
       Canonical sampling through velocity rescaling.
       The Journal of chemical physics, 126(1), 014101. 2007.
    """

    def __init__(
        self, temperature_bath, time_constant, nm_transformation=NormalModeTransformer
    ):
        logging.info("Using global PILE thermostat")
        super(PILEGlobalThermostat, self).__init__(
            temperature_bath, time_constant, nm_transformation=nm_transformation
        )

    def _apply_thermostat(self, simulator):
        """
        Apply the global PILE thermostat to the system momenta. This is essentially the same as for the basic Langevin
        thermostat, with exception of replacing the equations for the centroid (index 0 in first dimension) with the
        stochastic velocity rescaling equations.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Get current momenta
        momenta = simulator.system.momenta

        # Apply transformation
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.beads2normal(momenta)

        # Generate random noise
        thermostat_noise = torch.randn(momenta.shape, device=self.device)

        # Apply thermostat to centroid mode
        c1_centroid = self.c1[0]

        momenta_centroid = momenta[0]
        thermostat_noise_centroid = thermostat_noise[0]

        # Compute kinetic energy of centroid
        kinetic_energy_factor = torch.sum(
            momenta_centroid ** 2 / simulator.system.masses[0]
        ) / (self.temperature_bath * MDUnits.kB * self.n_replicas)

        centroid_factor = (1 - c1_centroid) / kinetic_energy_factor

        alpha_sq = (
            c1_centroid
            + torch.sum(thermostat_noise_centroid ** 2) * centroid_factor
            + 2
            * thermostat_noise_centroid[0, 0, 0]
            * torch.sqrt(c1_centroid * centroid_factor)
        )

        alpha_sign = torch.sign(
            thermostat_noise_centroid[0, 0, 0]
            + torch.sqrt(c1_centroid / centroid_factor)
        )

        alpha = torch.sqrt(alpha_sq) * alpha_sign

        # Finally apply thermostat...
        momenta[0] = alpha * momenta[0]

        # Apply thermostat for remaining normal modes
        momenta[1:] = (
            self.c1[1:] * momenta[1:]
            + self.thermostat_factor * self.c2[1:] * thermostat_noise[1:]
        )

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.normal2beads(momenta)

        simulator.system.momenta = momenta


class TRPMDThermostat(PILELocalThermostat):
    """
    Thermostatted ring polymer molecular dynamics thermostat variant of the local PILE thermostat as introduced in
    [#trpmd_thermostat1]_. Here, no thermostat is applied to the centroid and the dynamics of the system are damped via
    a given damping factor.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        damping (float): Damping factor of the thermostat.
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module use dto transform between beads and normal
                                                                       model representation in ring polymer dynamics.

    References
    ----------
    .. [#trpmd_thermostat1] Rossi, Ceriotti, Manolopoulos:
       How to remove the spurious resonances from ring polymer molecular dynamics.
       The Journal of Chemical Physics, 140(23), 234116. 2014.
    """

    def __init__(
        self, temperature_bath, damping, nm_transformation=NormalModeTransformer
    ):
        super(TRPMDThermostat, self).__init__(
            temperature_bath,
            1.0,
            nm_transformation=nm_transformation,
            thermostat_centroid=False,
            damping=damping,
        )


class NHCThermostat(ThermostatHook):
    """
    Nose-Hover chain thermostat, which links the system to a chain of deterministic Nose-Hoover thermostats first
    introduced in [#nhc_thermostat1]_ and described in great detail in [#nhc_thermostat2]_. Advantage of the NHC
    thermostat is, that it does not apply random perturbations to the system and is hence fully deterministic. However,
    this comes at an increased numerical cost compared to e.g. the stochastic thermostats described above.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
        chain_length (int): Number of Nose-Hoover thermostats applied in the chain.
        massive (bool): If set to true, an individual thermostat is applied to each degree of freedom in the system.
                        Can e.g. be used for thermostatting (default=False).
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module used to transform between beads and normal
                                                                   model representation in ring polymer dynamics.
        multi_step (int): Number of steps used for integrating the NH equations of motion (default=2)
        integration_order (int): Order of the Yoshida-Suzuki integrator used for propagating the thermostat (default=3).

    References
    ----------
    .. [#nhc_thermostat1] Tobias, Martyna, Klein:
       Molecular dynamics simulations of a protein in the canonical ensemble.
       The Journal of Physical Chemistry, 97(49), 12959-12966. 1993.
    .. [#nhc_thermostat2] Martyna, Tuckerman, Tobias, Klein:
       Explicit reversible integrators for extended systems dynamics.
       Molecular Physics, 87(5), 1117-1157. 1996.
    """

    def __init__(
        self,
        temperature_bath,
        time_constant,
        chain_length=3,
        massive=False,
        nm_transformation=None,
        multi_step=2,
        integration_order=3,
    ):
        super(NHCThermostat, self).__init__(
            temperature_bath, nm_transformation=nm_transformation
        )

        self.chain_length = chain_length
        self.massive = massive
        self.frequency = 1 / (time_constant * MDUnits.fs2atu)

        # Cpmpute kBT, since it will be used a lot
        self.kb_temperature = self.temperature_bath * MDUnits.kB

        # Propagation parameters
        self.multi_step = multi_step
        self.integration_order = integration_order
        self.time_step = None

        # Find out number of particles (depends on whether massive or not)
        self.degrees_of_freedom = None
        self.masses = None

        self.velocities = None
        self.positions = None
        self.forces = None

    def _init_thermostat(self, simulator):
        """
        Initialize the thermostat positions, forces, velocities and masses, as well as the number of degrees of freedom
        seen by each chain link.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Determine integration step via multi step and Yoshida Suzuki weights
        integration_weights = YSWeights(self.device).get_weights(self.integration_order)
        self.time_step = (
            simulator.integrator.time_step * integration_weights / self.multi_step
        )

        # Determine shape of tensors and internal degrees of freedom
        n_replicas, n_molecules, n_atoms, xyz = simulator.system.momenta.shape

        if self.massive:
            state_dimension = (n_replicas, n_molecules, n_atoms, xyz, self.chain_length)
            # Since momenta will be masked later, no need to set non-atoms to 0
            self.degrees_of_freedom = torch.ones(
                (n_replicas, n_molecules, n_atoms, xyz), device=self.device
            )
        else:
            state_dimension = (n_replicas, n_molecules, 1, 1, self.chain_length)
            self.degrees_of_freedom = (
                3 * simulator.system.n_atoms.float()[None, :, None, None]
            )

        # Set up masses
        self._init_masses(state_dimension, simulator)

        # Set up internal variables
        self.positions = torch.zeros(state_dimension, device=self.device)
        self.forces = torch.zeros(state_dimension, device=self.device)
        self.velocities = torch.zeros(state_dimension, device=self.device)

    def _init_masses(self, state_dimension, simulator):
        """
        Auxiliary routine for initializing the thermostat masses.

        Args:
            state_dimension (tuple): Size of the thermostat states. This is used to differentiate between the massive
                                     and the standard algorithm
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        self.masses = torch.ones(state_dimension, device=self.device)
        # Get masses of innermost thermostat
        self.masses[..., 0] = (
            self.degrees_of_freedom * self.kb_temperature / self.frequency ** 2
        )
        # Set masses of remaining thermostats
        self.masses[..., 1:] = self.kb_temperature / self.frequency ** 2

    def _propagate_thermostat(self, kinetic_energy):
        """
        Propagation step of the NHC thermostat. Please refer to [#nhc_thermostat2]_ for more detail on the algorithm.

        Args:
            kinetic_energy (torch.Tensor): Kinetic energy associated with the innermost NH thermostats.

        Returns:
            torch.Tensor: Scaling factor applied to the system momenta.

        References
        ----------
        .. [#nhc_thermostat2] Martyna, Tuckerman, Tobias, Klein:
           Explicit reversible integrators for extended systems dynamics.
           Molecular Physics, 87(5), 1117-1157. 1996.
        """
        # Compute forces on first thermostat
        self.forces[..., 0] = (
            kinetic_energy - self.degrees_of_freedom * self.kb_temperature
        ) / self.masses[..., 0]

        scaling_factor = 1.0
        for _ in range(self.multi_step):
            for idx_ys in range(self.integration_order):
                time_step = self.time_step[idx_ys]

                # Update velocities of outermost bath
                self.velocities[..., -1] += 0.25 * self.forces[..., -1] * time_step

                # Update the velocities moving through the beads of the chain
                for chain in range(self.chain_length - 2, -1, -1):
                    coeff = torch.exp(
                        -0.125 * time_step * self.velocities[..., chain + 1]
                    )
                    self.velocities[..., chain] = (
                        self.velocities[..., chain] * coeff ** 2
                        + 0.25 * self.forces[..., chain] * coeff * time_step
                    )

                # Accumulate velocity scaling
                scaling_factor *= torch.exp(-0.5 * time_step * self.velocities[..., 0])
                # Update forces of innermost thermostat
                self.forces[..., 0] = (
                    scaling_factor * scaling_factor * kinetic_energy
                    - self.degrees_of_freedom * self.kb_temperature
                ) / self.masses[..., 0]

                # Update thermostat positions
                # TODO: Only required if one is interested in the conserved
                #  quanity of the NHC.
                self.positions += 0.5 * self.velocities * time_step

                # Update the thermostat velocities
                for chain in range(self.chain_length - 1):
                    coeff = torch.exp(
                        -0.125 * time_step * self.velocities[..., chain + 1]
                    )
                    self.velocities[..., chain] = (
                        self.velocities[..., chain] * coeff ** 2
                        + 0.25 * self.forces[..., chain] * coeff * time_step
                    )
                    self.forces[..., chain + 1] = (
                        self.masses[..., chain] * self.velocities[..., chain] ** 2
                        - self.kb_temperature
                    ) / self.masses[..., chain + 1]

                # Update velocities of outermost thermostat
                self.velocities[..., -1] += 0.25 * self.forces[..., -1] * time_step

        return scaling_factor

    def _compute_kinetic_energy(self, momenta, masses):
        """
        Routine for computing the kinetic energy of the innermost NH thermostats based on the momenta and masses of the
        simulated systems.

        Args:
            momenta (torch.Tensor): Momenta of the simulated system.
            masses (torch.Tensor): Masses of the simulated system.

        Returns:
            torch.Tensor: Kinetic energy associated with the innermost NH thermostats. These are summed over the
                          corresponding degrees of freedom, depending on whether a massive NHC is used.

        """
        # Compute the kinetic energy (factor of 1/2 can be removed, as it
        # cancels with a times 2)
        # TODO: Is no problem, as NM transformation never mixes atom dimension
        #  which carries the masses.
        kinetic_energy = momenta ** 2 / masses
        if self.massive:
            return kinetic_energy
        else:
            return torch.sum(
                torch.sum(kinetic_energy, 3, keepdim=True), 2, keepdim=True
            )

    def _apply_thermostat(self, simulator):
        """
        Propagate the NHC thermostat, compute the corresponding scaling factor and apply it to the momenta of the
        system. If a normal mode transformer is provided, this is done in the normal model representation of the ring
        polymer.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Get current momenta
        momenta = simulator.system.momenta

        # Apply transformation
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.beads2normal(momenta)

        kinetic_energy = self._compute_kinetic_energy(momenta, simulator.system.masses)

        scaling_factor = self._propagate_thermostat(kinetic_energy)
        momenta = momenta * scaling_factor

        # Apply transformation if requested
        if self.nm_transformation is not None:
            momenta = self.nm_transformation.normal2beads(momenta)

        simulator.system.momenta = momenta

    @property
    def state_dict(self):
        state_dict = {
            "chain_length": self.chain_length,
            "massive": self.massive,
            "frequency": self.frequency,
            "kb_temperature": self.kb_temperature,
            "degrees_of_freedom": self.degrees_of_freedom,
            "masses": self.masses,
            "velocities": self.velocities,
            "forces": self.forces,
            "positions": self.positions,
            "time_step": self.time_step,
            "temperature_bath": self.temperature_bath,
            "n_replicas": self.n_replicas,
            "multi_step": self.multi_step,
            "integration_order": self.integration_order,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.chain_length = state_dict["chain_length"]
        self.massive = state_dict["massive"]
        self.frequency = state_dict["frequency"]
        self.kb_temperature = state_dict["kb_temperature"]
        self.degrees_of_freedom = state_dict["degrees_of_freedom"]
        self.masses = state_dict["masses"]
        self.velocities = state_dict["velocities"]
        self.forces = state_dict["forces"]
        self.positions = state_dict["positions"]
        self.time_step = state_dict["time_step"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.n_replicas = state_dict["n_replicas"]
        self.multi_step = state_dict["multi_step"]
        self.integration_order = state_dict["integration_order"]

        self.initialized = True


class NHCRingPolymerThermostat(NHCThermostat):
    """
    Nose-Hoover chain thermostat for ring polymer molecular dynamics simulations as e.g. described in
    [#stochastic_thermostats4]_. This is based on the massive setting of the standard NHC thermostat but operates in
    the normal mode representation and uses specially initialized thermostat masses.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
        chain_length (int): Number of Nose-Hoover thermostats applied in the chain.
        local (bool): If set to true, an individual thermostat is applied to each degree of freedom in the system.
                        Can e.g. be used for thermostatting (default=False).
        nm_transformation (schnetpack.md.utils.NormalModeTransformer): Module used to transform between beads and normal
                                                                   model representation in ring polymer dynamics.
        multi_step (int): Number of steps used for integrating the NH equations of motion (default=2)
        integration_order (int): Order of the Yoshida-Suzuki integrator used for propagating the thermostat (default=3).


    References
    ----------
    .. [#stochastic_thermostats4] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133 (12), 124104. 2010.
    """

    def __init__(
        self,
        temperature_bath,
        time_constant,
        chain_length=3,
        local=True,
        nm_transformation=NormalModeTransformer,
        multi_step=2,
        integration_order=3,
    ):
        super(NHCRingPolymerThermostat, self).__init__(
            temperature_bath,
            time_constant,
            chain_length=chain_length,
            massive=True,
            nm_transformation=nm_transformation,
            multi_step=multi_step,
            integration_order=integration_order,
        )
        self.local = local

    def _init_masses(self, state_dimension, simulator):
        """
        Initialize masses according to the normal mode frequencies of the ring polymer system.

        Args:
            state_dimension (tuple): Size of the thermostat states. This is used to differentiate between the massive
                                     and the standard algorithm
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Multiply factor by number of replicas
        self.kb_temperature = self.kb_temperature * self.n_replicas

        # Initialize masses with the frequencies of the ring polymer
        polymer_frequencies = simulator.integrator.omega_normal
        # 0.5 comes from Ceriotti paper, check
        polymer_frequencies[0] = 0.5 * self.frequency

        # Assume standard massive Nose-Hoover and initialize accordingly
        self.masses = torch.ones(state_dimension, device=self.device)
        self.masses *= (
            self.kb_temperature / polymer_frequencies[:, None, None, None, None] ** 2
        )

        # If a global thermostat is requested, we assign masses of 3N to
        # the first link in the chain on the centroid
        if not self.local:
            self.masses[0, :, :, :, 0] *= (
                3 * simulator.system.n_atoms.float()[:, None, None]
            )
            # Degrees of freedom also need to be adapted
            self.degrees_of_freedom[0, :, :, :] *= (
                3 * simulator.system.n_atoms.float()[:, None, None]
            )

    def _compute_kinetic_energy(self, momenta, masses):
        """
        Routine for computing the kinetic energies of the innermost NH thermostats based on the masses and momenta of
        the ring polymer in normal mode representation.

        Args:
            momenta (torch.Tensor): Normal mode momenta of the simulated system.
            masses (torch.Tensor): Masses of the simulated system.

        Returns:
            torch.Tensor: Kinetic energy of the innermost NH thermostats.
        """
        kinetic_energy = momenta ** 2 / masses

        # In case of a global NHC for RPMD, use the whole centroid kinetic
        # energy and broadcast it
        if not self.local:
            kinetic_energy_centroid = torch.sum(
                torch.sum(kinetic_energy[0, ...], 2, keepdim=True), 1, keepdim=True
            )
            kinetic_energy[0, ...] = kinetic_energy_centroid

        return kinetic_energy

    @property
    def state_dict(self):
        state_dict = {
            "chain_length": self.chain_length,
            "massive": self.massive,
            "frequency": self.frequency,
            "kb_temperature": self.kb_temperature,
            "degrees_of_freedom": self.degrees_of_freedom,
            "masses": self.masses,
            "velocities": self.velocities,
            "forces": self.forces,
            "positions": self.positions,
            "time_step": self.time_step,
            "temperature_bath": self.temperature_bath,
            "n_replicas": self.n_replicas,
            "multi_step": self.multi_step,
            "integration_order": self.integration_order,
            "local": self.local,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.chain_length = state_dict["chain_length"]
        self.massive = state_dict["massive"]
        self.frequency = state_dict["frequency"]
        self.kb_temperature = state_dict["kb_temperature"]
        self.degrees_of_freedom = state_dict["degrees_of_freedom"]
        self.masses = state_dict["masses"]
        self.velocities = state_dict["velocities"]
        self.forces = state_dict["forces"]
        self.positions = state_dict["positions"]
        self.time_step = state_dict["time_step"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.n_replicas = state_dict["n_replicas"]
        self.multi_step = state_dict["multi_step"]
        self.integration_order = state_dict["integration_order"]
        self.local = state_dict["local"]

        self.initialized = True

"""
This module contains pecialized thermostats for controlling temperature of the system during
ring polymer molecular dynamics simulations.
"""
from __future__ import annotations
import torch

from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md.simulator import Simulator, System

from schnetpack.md.simulation_hooks.thermostats import (
    LangevinThermostat,
    GLEThermostat,
    ThermostatError,
    NHCThermostat,
)
from schnetpack.md.utils import load_gle_matrices
from schnetpack import units as spk_units

__all__ = [
    "PILELocalThermostat",
    "PILEGlobalThermostat",
    "TRPMDThermostat",
    "RPMDGLEThermostat",
    "PIGLETThermostat",
    "NHCRingPolymerThermostat",
]


class PILELocalThermostat(LangevinThermostat):
    """
    Langevin thermostat for ring polymer molecular dynamics as introduced in [#stochastic_thermostats2]_.
    Applies specially initialized Langevin thermostats to the beads of the ring polymer in normal mode representation.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
        thermostat_centroid (bool): Whether a thermostat should be applied to the centroid of the ring polymer in
                                    normal mode representation (relevant e.g. for TRPMD, default is True)
        damping_factor (float): If specified, damping factor is applied to the current momenta of the system (used in TRPMD,
                         default is no damping).

    References
    ----------
    .. [#stochastic_thermostats2] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133 (12), 124104. 2010.
    """

    ring_polymer = True

    def __init__(
        self,
        temperature_bath: float,
        time_constant: float,
        thermostat_centroid: Optional[bool] = True,
        damping_factor: Optional[float] = 1.0,
    ):
        super(PILELocalThermostat, self).__init__(
            temperature_bath=temperature_bath, time_constant=time_constant
        )
        self.register_buffer("thermostat_centroid", torch.tensor(thermostat_centroid))
        self.register_buffer("damping_factor", torch.tensor(damping_factor))

    def _init_thermostat(self, simulator):
        """
        Initialize the Langevin matrices based on the normal mode frequencies of the ring polymer. If the centroid is to
        be thermostatted, the suggested value of 1/time_constant is used.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Initialize friction coefficients
        gamma_normal = 2 * simulator.integrator.omega_normal

        # Use seperate coefficient for centroid mode (default, unless using thermostatted RPMD)
        if self.thermostat_centroid:
            gamma_normal[0] = 1.0 / self.time_constant

        # Apply TRPMD damping factor if provided
        gamma_normal *= self.damping_factor

        # Initialize coefficient matrices
        c1 = torch.exp(-0.5 * simulator.integrator.time_step * gamma_normal)
        c2 = torch.sqrt(1 - c1**2)

        self.c1 = c1[:, None, None]
        self.c2 = c2[:, None, None]

        # Get mass and temperature factors
        self.thermostat_factor = torch.sqrt(
            simulator.system.masses
            * spk_units.kB
            * simulator.system.n_replicas
            * self.temperature_bath
        )

    def _apply_thermostat(self, simulator: Simulator):
        """
        Apply the PILE thermostat to the systems momenta in normal mode representation.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Get current momenta
        momenta = simulator.system.momenta_normal

        # Generate random noise
        thermostat_noise = torch.randn_like(momenta)

        # Apply thermostat
        simulator.system.momenta_normal = (
            self.c1 * momenta + self.thermostat_factor * self.c2 * thermostat_noise
        )


class PILEGlobalThermostat(PILELocalThermostat):
    """
    Global variant of the ring polymer Langevin thermostat as suggested in [#stochastic_thermostats3]_. This thermostat
    applies a stochastic velocity rescaling thermostat [#stochastic_velocity_rescaling1]_ to the ring polymer centroid
    in normal mode representation.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs

    References
    ----------
    .. [#stochastic_thermostats3] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133 (12), 124104. 2010.
    .. [#stochastic_velocity_rescaling1] Bussi, Donadio, Parrinello:
       Canonical sampling through velocity rescaling.
       The Journal of chemical physics, 126(1), 014101. 2007.
    """

    def __init__(self, temperature_bath: float, time_constant: float):
        super(PILEGlobalThermostat, self).__init__(
            temperature_bath=temperature_bath, time_constant=time_constant
        )

    def _apply_thermostat(self, simulator: Simulator):
        """
        Apply the global PILE thermostat to the system momenta. This is essentially the same as for the basic Langevin
        thermostat, with exception of replacing the equations for the centroid (index 0 in first dimension) with the
        stochastic velocity rescaling equations.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Get current momenta in normal mode representations
        momenta = simulator.system.momenta_normal

        # Generate random noise
        thermostat_noise = torch.randn_like(momenta)

        # Apply thermostat to centroid mode
        c1_centroid = self.c1[0][None, ...]

        momenta_centroid = momenta[0][None, ...]
        thermostat_noise_centroid = thermostat_noise[0][None, ...]

        # Compute kinetic energy of centroid
        kinetic_energy_centroid = simulator.system.sum_atoms(
            torch.sum(
                momenta_centroid**2 / simulator.system.masses, dim=2, keepdim=True
            )
        )
        kinetic_energy_factor = kinetic_energy_centroid / (
            self.temperature_bath * spk_units.kB * simulator.system.n_replicas
        )

        centroid_factor = (1 - c1_centroid) / kinetic_energy_factor

        alpha_sq = (
            c1_centroid
            + simulator.system.sum_atoms(
                torch.sum(thermostat_noise_centroid**2, dim=2, keepdim=True)
            )
            * centroid_factor
            + 2
            * thermostat_noise_centroid[0, 0, 0]
            * torch.sqrt(c1_centroid * centroid_factor)
        )

        alpha_sign = torch.sign(
            thermostat_noise_centroid[0, 0, 0]
            + torch.sqrt(c1_centroid / centroid_factor)
        )

        alpha = torch.sqrt(alpha_sq) * alpha_sign

        # Finally apply thermostat to centroid
        momenta[0] = simulator.system.expand_atoms(alpha)[0] * momenta[0]

        # Apply thermostat for remaining normal modes
        momenta[1:] = (
            self.c1[1:] * momenta[1:]
            + self.thermostat_factor * self.c2[1:] * thermostat_noise[1:]
        )

        simulator.system.momenta_normal = momenta


class TRPMDThermostat(PILELocalThermostat):
    """
    Thermostatted ring polymer molecular dynamics thermostat variant of the local PILE thermostat as introduced in
    [#trpmd_thermostat1]_. Here, no thermostat is applied to the centroid and the dynamics of the system are damped via
    a given damping factor.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        damping_factor (float): Damping factor of the thermostat.

    References
    ----------
    .. [#trpmd_thermostat1] Rossi, Ceriotti, Manolopoulos:
       How to remove the spurious resonances from ring polymer molecular dynamics.
       The Journal of Chemical Physics, 140(23), 234116. 2014.
    """

    def __init__(self, temperature_bath: float, damping_factor: float):
        super(TRPMDThermostat, self).__init__(
            temperature_bath=temperature_bath,
            time_constant=1.0,
            thermostat_centroid=False,
            damping_factor=damping_factor,
        )


class RPMDGLEThermostat(GLEThermostat):
    """
    Stochastic generalized Langevin colored noise thermostat for RPMD by Ceriotti et. al. as described in
    [#gle_thermostat1]_. This thermostat requires specially parametrized matrices, which can be obtained online from:
    http://gle4md.org/index.html?page=matrix

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        gle_file (str): File containing the GLE matrices
        free_particle_limit (bool): Initialize momenta according to free particle limit instead of a zero matrix
                                    (default=True).

    References
    ----------
    .. [#gle_thermostat1] Ceriotti, Bussi, Parrinello:
       Colored-noise thermostats à la carte.
       Journal of Chemical Theory and Computation 6 (4), 1170-1180. 2010.
    """

    # TODO: this does not seem to give the proper temperature average
    ring_polymer = True

    def __init__(
        self,
        temperature_bath: float,
        gle_file: str,
        free_particle_limit: Optional[bool] = True,
    ):
        super(RPMDGLEThermostat, self).__init__(
            temperature_bath=temperature_bath,
            gle_file=gle_file,
            free_particle_limit=free_particle_limit,
        )

    def _apply_thermostat(self, simulator):
        """
        Perform the update of the system momenta according to the GLE thermostat.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Generate random noise
        thermostat_noise = torch.randn_like(self.thermostat_momenta)

        # Get current momenta
        momenta = simulator.system.momenta_normal

        # Set current momenta
        self.thermostat_momenta[:, :, :, 0] = momenta

        # Apply thermostat
        self.thermostat_momenta = (
            torch.matmul(self.thermostat_momenta, self.c1)
            + torch.matmul(thermostat_noise, self.c2) * self.thermostat_factor
        )

        # Extract and set momenta
        simulator.system.momenta_normal = self.thermostat_momenta[:, :, :, 0]


class PIGLETThermostat(RPMDGLEThermostat):
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

    def __init__(self, temperature_bath: float, gle_file: str):
        super(PIGLETThermostat, self).__init__(
            temperature_bath=temperature_bath,
            gle_file=gle_file,
            free_particle_limit=True,
        )

    def _init_gle_matrices(self, simulator):
        """
        Initialize the matrices necessary for the PIGLET thermostat. In contrast to the basic GLE thermostat, these
        have the dimension:
        n_replicas x degrees_of_freedom x degrees_of_freedom,
        where n_replicas is the number of beads in the ring polymer and degrees_of_freedom is the number of degrees of
        freedom introduced by GLE.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.

        Returns:
            torch.Tensor: Drift matrices for the PIGLET thermostat.
            torch.Tensor: Diffusion matrices.
        """
        a_matrix, c_matrix = load_gle_matrices(self.gle_file)

        if a_matrix is None:
            raise ThermostatError(
                "Error reading GLE matrices " "from {:s}".format(self.gle_file)
            )
        if a_matrix.shape[0] != simulator.system.n_replicas:
            raise ThermostatError(
                "Expected {:d} beads but "
                "found {:d}.".format(simulator.system.n_replicas, a_matrix.shape[0])
            )

        all_c1 = []
        all_c2 = []

        # Generate main matrices
        for b in range(simulator.system.n_replicas):
            c1, c2 = self._init_single_gle_matrix(
                a_matrix[b], (c_matrix[b], None)[c_matrix is None], simulator
            )
            # Add extra dimension for use with torch.cat, correspond to normal
            # modes of ring polymer
            all_c1.append(c1[None, ...])
            all_c2.append(c2[None, ...])

        # Bring to correct shape for later matmul broadcasting
        c1 = torch.cat(all_c1)[:, None, :, :]
        c2 = torch.cat(all_c2)[:, None, :, :]
        return c1, c2


class NHCRingPolymerThermostat(NHCThermostat):
    """
    Nose-Hoover chain thermostat for ring polymer molecular dynamics simulations as e.g. described in
    [#stochastic_thermostats4]_. This is based on the massive setting of the standard NHC thermostat but operates in
    the normal mode representation and uses specially initialized thermostat masses.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
        local (bool): If set to true, an individual thermostat is applied to each degree of freedom in the system.
                        Can e.g. be used for thermostatting (default=False).
        chain_length (int): Number of Nose-Hoover thermostats applied in the chain.
        multi_step (int): Number of steps used for integrating the NH equations of motion (default=2)
        integration_order (int): Order of the Yoshida-Suzuki integrator used for propagating the thermostat (default=3).


    References
    ----------
    .. [#stochastic_thermostats4] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133 (12), 124104. 2010.
    """

    ring_polymer = True

    def __init__(
        self,
        temperature_bath: float,
        time_constant: float,
        local: Optional[bool] = True,
        chain_length: Optional[int] = 3,
        multi_step: Optional[int] = 2,
        integration_order: Optional[int] = 3,
    ):
        super(NHCRingPolymerThermostat, self).__init__(
            temperature_bath=temperature_bath,
            time_constant=time_constant,
            chain_length=chain_length,
            massive=True,
            multi_step=multi_step,
            integration_order=integration_order,
        )
        self.register_buffer("local", torch.tensor(local))

    def _init_masses(self, state_dimension: List[int], simulator: Simulator):
        """
        Initialize masses according to the normal mode frequencies of the ring polymer system.

        Args:
            state_dimension (tuple): Size of the thermostat states. This is used to differentiate between the massive
                                     and the standard algorithm
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Multiply factor by number of replicas
        self.kb_temperature = self.kb_temperature * simulator.system.n_replicas

        # Initialize masses with the frequencies of the ring polymer
        polymer_frequencies = simulator.integrator.omega_normal
        # 0.5 comes from Ceriotti paper, check
        polymer_frequencies[0] = 0.5 * self.frequency

        # Assume standard massive Nose-Hoover and initialize accordingly
        self.masses = torch.ones(
            state_dimension, device=simulator.device, dtype=simulator.dtype
        )

        self.masses *= (
            self.kb_temperature / polymer_frequencies[:, None, None, None] ** 2
        )

        # If a global thermostat is requested, we assign masses of 3N to
        # the first link in the chain on the centroid
        if not self.local:
            atoms_factor = (
                3.0
                * simulator.system.expand_atoms(
                    simulator.system.n_atoms[None, :, None]
                )[0]
            )
            self.masses[0, :, :, 0] *= atoms_factor
            # Degrees of freedom also need to be adapted
            self.degrees_of_freedom[0, :, :] *= atoms_factor

    def _compute_kinetic_energy(self, system: System):
        """
        Routine for computing the kinetic energies of the innermost NH thermostats based on the masses and momenta of
        the ring polymer in normal mode representation.

        Args:
            system (schnetpack.md.System): System object.

        Returns:
            torch.Tensor: Kinetic energy of the innermost NH thermostats.
        """
        kinetic_energy = system.momenta_normal**2 / system.masses

        # In case of a global NHC for RPMD, use the whole centroid kinetic
        # energy and broadcast it
        if not self.local:
            kinetic_energy_centroid = system.sum_atoms(
                torch.sum(kinetic_energy[0:1], dim=2, keepdim=True)
            )
            kinetic_energy[0, ...] = system.expand_atoms(kinetic_energy_centroid)[0]

        return kinetic_energy

    def _apply_thermostat(self, simulator):
        """
        Propagate the NHC thermostat, compute the corresponding scaling factor and apply it to the momenta of the
        system. If a normal mode transformer is provided, this is done in the normal model representation of the ring
        polymer.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Get current momenta
        momenta = simulator.system.momenta_normal

        kinetic_energy = self._compute_kinetic_energy(simulator.system)

        scaling_factor = self._propagate_thermostat(kinetic_energy)
        momenta = momenta * scaling_factor

        # self.compute_conserved(simulator.system)

        # Apply transformation if requested
        simulator.system.momenta_normal = momenta

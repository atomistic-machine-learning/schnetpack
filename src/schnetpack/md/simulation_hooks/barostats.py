"""
This module contains various thermostats for regulating the temperature of the system during
molecular dynamics simulations. Apart from standard thermostats for convetional simulations,
a series of special thermostats developed for ring polymer molecular dynamics is also provided.
"""
import torch
import numpy as np
import scipy.linalg as linalg
import logging

from schnetpack.md.utils import MDUnits, YSWeights, StableSinhDiv
from schnetpack.md.simulation_hooks import ThermostatHook, SimulationHook

__all__ = ["BarostatHook"]


# TODO:
#   -) state dicts
#   -) documentation
#   -) check dimensions


class BarostatHook(SimulationHook):
    """
    """

    def __init__(self, target_pressure, temperature_bath, detach=True):
        self.target_pressure = target_pressure
        self.temperature_bath = temperature_bath
        self.initialized = False
        self.device = None
        self.n_replicas = None
        self.n_molecules = None
        self.detach = detach
        self.time_step = None

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
        self.n_molecules = simulator.system.n_molecules
        self.time_step = simulator.integrator.time_step

        # TODO: Check if thermostat hook is used, which one is used
        #   -> is right integrator used? NPTIntegrator!!!!
        #   -> is a thermostat suitable for RPMD used??

        if not self.initialized:
            self._init_barostat(simulator)
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
        self._apply_barostat(simulator)

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
        self._apply_barostat(simulator)

        # Re-apply atom masks for differently sized molecules, as some
        # thermostats add random noise
        simulator.system.momenta = (
            simulator.system.momenta * simulator.system.atom_masks
        )

        # Detach if requested
        if self.detach:
            simulator.system.momenta = simulator.system.momenta.detach()

    def _init_barostat(self, simulator):
        """
        Dummy routine for initializing a thermostat based on the current simulator. Should be implemented for every
        new thermostat. Has access to the information contained in the simulator class, e.g. number of replicas, time
        step, masses of the atoms, etc.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        raise NotImplementedError

    def _apply_barostat(self, simulator):
        """
        Dummy routine for applying the thermostat to the system. Should use the implemented thermostat to update the
        momenta of the system contained in simulator.system.momenta. Is called twice each simulation time step.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        raise NotImplementedError

    def propagate_system(self, system):
        """
        Propagate the positions and cell of the system. Defaults to classic Verlet.

        ..math::
            q = q + \frac{p}{m} \delta t

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        system.positions = (
            system.positions + self.time_step * system.momenta / system.masses
        )
        system.positions = system.positions.detach()

    def propagate_barostat(self, system):
        """
        Used for RPMD

        Args:
            system:
        """
        pass


# TODO Integrator, PI baro, NHC iso, aniso, standalone NHC


class MTKBarostat(BarostatHook):
    def __init__(self, target_pressure, time_constant, temperature_bath, detach=True):
        super(MTKBarostat, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            detach=detach,
        )
        self.frequency = 1.0 / (time_constant * MDUnits.fs2atu)
        self.inv_kb_temperature = 1.0 / (self.temperature_bath * MDUnits.kB)

    def _init_barostat(self, simulator):
        # TODO: perform checks here?

        # Get scaling factors based on the number of atoms
        n_atoms = simulator.system.n_atoms[None, :, None, None].float()
        self.inv_sqrt_dof = 1.0 / torch.sqrt(3.0 * n_atoms + 1.0)
        self.inv_dof = 1.0 / n_atoms
        self.weight = self.frequency * self.inv_sqrt_dof

        # Thermostat variable, this will be used for scaling the positions and momenta
        self.zeta = torch.zeros(
            self.n_replicas, self.n_molecules, 1, 1, device=self.device
        )

    def _apply_barostat(self, simulator):
        # TODO: check dimensions and pressure units

        # 1) Update barostat variable
        # pressure_factor = 3.0 * simulator.system.volume * (
        #        simulator.system.compute_pressure(kinetic_component=True) - self.target_pressure
        # ) + 2.0 * self.inv_dof * simulator.system.kinetic_energy

        # The pressure factor is 3*V * (Pint - Pext) + 2*Ekin / N
        # Since Pint also contains the kinetic component it computes as
        #   Pint = 1/(3V) * 2 * Ekin + Pint'
        # This means, the expression can be rewritten as
        #   Pfact = 3*V*(Pint' - Pext) + 2*(1+1/N) * Ekin
        # Saving some computations
        pressure_factor = (
            3.0
            * simulator.system.volume
            * (simulator.system.compute_pressure() - self.target_pressure)
            + 2.0 * (1 + self.inv_dof) * simulator.system.kinetic_energy
        )

        self.zeta = (
            self.zeta
            + 0.5
            * self.time_step
            * self.weight
            * self.inv_kb_temperature
            * pressure_factor
        )

        # 2) Scale positions and cell
        scaling = torch.exp(self.time_step * self.weight * self.zeta)

        simulator.system.positions = simulator.system.positions * scaling
        simulator.system.cells = simulator.system.cells * scaling

        # 3) Scale momenta
        scaling = torch.exp(
            -self.time_step * self.weight * (1 + self.inv_dof) * self.zeta
        )
        simulator.system.momenta = simulator.system.momenta * scaling

        # 4) Second update of barostat variable based on new momenta and positions
        pressure_factor = (
            3.0
            * simulator.system.volume
            * (simulator.system.compute_pressure() - self.target_pressure)
            + 2.0 * (1 + self.inv_dof) * simulator.system.kinetic_energy
        )

        self.zeta = (
            self.zeta
            + 0.5
            * self.time_step
            * self.weight
            * self.inv_kb_temperature
            * pressure_factor
        )

        if self.detach:
            simulator.system.cells = simulator.system.cells.detach()
            simulator.system.positions = simulator.system.positions.detach()
            simulator.system.momenta = simulator.system.momenta.detach()
            self.zeta = self.zeta.detach()


class RPMDBarostat(BarostatHook):
    def __init__(self, target_pressure, time_constant, temperature_bath, detach=True):
        super(RPMDBarostat, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            detach=detach,
        )
        self.frequency = 1.0 / (time_constant * MDUnits.fs2atu)
        self.kb_temperature = temperature_bath * MDUnits.kB
        self.transformation = None
        self.propagator = None
        self.momenta = None
        self.sinhdx = StableSinhDiv()

    def _init_barostat(self, simulator):
        # Get normal mode transformer and propagator for position and cell update
        self.transformation = simulator.integrator.transformation
        self.propagator = simulator.integrator.propagator

        # Set up centroid momenta of cell (one for every molecule)
        self.cell_momenta = torch.zeros(
            self.n_molecules, device=simulator.system.device
        )
        self.mass = (
            3.0 * simulator.system.n_atoms / self.frequency ** 2 * self.kb_temperature
        )

        # Set up cell thermostat coefficients
        self.c1 = torch.exp(-0.5 * self.frequency * self.time_step)
        self.c2 = torch.sqrt(
            self.n_replicas * self.mass * self.kb_temperature * (1.0 - self.c1 ** 2)
        )

    def _apply_barostat(self, simulator):
        # Propagate cell momenta during half-step
        self.cell_momenta = self.c1 * self.cell_momenta + self.c2 * torch.randn_like(
            self.cell_momenta
        )
        self.cell_momenta = self.cell_momenta.detach()

    def propagate_system(self, system):
        # Transform to normal mode representation
        positions_normal = self.transformation.beads2normal(system.positions)
        momenta_normal = self.transformation.beads2normal(system.momenta)

        # Propagate centroid mode of the ring polymer
        reduced_momenta = (self.cell_momenta / self.mass)[:, None, None]
        scaling = torch.exp(-self.time_step * reduced_momenta)

        momenta_normal[0] = momenta_normal[0] * scaling
        # TODO: Check for stability of sinh
        positions_normal[0] = (
            positions_normal[0] / scaling
            + self.sinhdx.f(self.time_step * reduced_momenta)
            * (momenta_normal[0] / system.masses[0])
            * self.time_step
        )

        # Update cells
        system.cells = system.cells / scaling[None, ...]

        # Propagate the remaining modes of the ring polymer
        momenta_normal[1:] = (
            self.propagator[1:, 0, 0] * momenta_normal[1:]
            + self.propagator[1:, 0, 1] * positions_normal[1:] * system.masses[1:]
        )
        positions_normal[1:] = (
            self.propagator[1:, 1, 0] * momenta_normal[1:] / system.masses[1:]
            + self.propagator[1:, 1, 1] * positions_normal[1:]
        )

        # Transform back to bead representation
        system.positions = self.transformation.normal2beads(positions_normal)
        system.momenta = self.transformation.normal2beads(momenta_normal)

    def propagate_barostat(self, system):
        centroid_momenta = self.transformation.beads2normal(system.momenta)[0]
        centroid_forces = self.transformation.beads2normal(system.forces)[0]

        # Compute pressure component
        component_1 = (
            3.0
            * self.n_replicas
            * (
                system.volume
                * (
                    torch.mean(system.compute_pressure(kinetic_component=True), dim=0)
                    - self.target_pressure
                )
                + self.kb_temperature
            )
        )

        # Compute components based on forces and momenta
        force_by_mass = centroid_forces / system.masses[:, :, None]

        component_2 = torch.sum(force_by_mass * centroid_momenta, dim=[1, 2])
        component_3 = torch.sum(force_by_mass * centroid_forces / 3, dim=[1, 2])

        # Update cell momenta
        self.cell_momenta = (
            self.cell_momenta
            + 0.5 * self.time_step * component_1
            + self.time_step ** 2 * component_2
            + self.time_step ** 3 * component_3
        )

        self.cell_momenta = self.cell_momenta.detach()


# TODO Masterclass


class NHCBarostatIsotropic(BarostatHook):
    """
    Nose Hoover chain thermostat/barostat for isotropic cell fluctuations.

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
        target_pressure,
        temperature_bath,
        time_constant,
        time_constant_barostat=None,
        chain_length=3,
        multi_step=2,
        integration_order=3,
        detach=True,
    ):
        super(NHCBarostatIsotropic, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            detach=detach,
        )
        self.chain_length = chain_length
        self.frequency = 1 / (time_constant * MDUnits.fs2atu)

        if time_constant_barostat is None:
            self.barostat_frequency = self.frequency
        else:
            self.barostat_frequency = 1 / (time_constant_barostat * MDUnits.fs2atu)

        # Compute kBT, since it will be used a lot
        self.kb_temperature = self.temperature_bath * MDUnits.kB

        # Stable sinh(x)/x approximation
        self.sinhdx = StableSinhDiv()

        # Propagation parameters
        self.multi_step = multi_step
        self.integration_order = integration_order
        self.ys_time_step = None

        # Find out number of particles
        self.degrees_of_freedom = None

        # Thermostat variables for particles
        self.t_velocities = None
        self.t_forces = None
        self.t_masses = None

        # Thermostat variables for cell
        self.t_velocities_cell = None
        self.t_forces_cell = None
        self.t_masses_cell = None

        # Barostat variables
        self.b_velocities_cell = None
        self.b_forces_cell = None
        self.b_masses_cell = None

    def _init_barostat(self, simulator):
        """
        Initialize the thermostat positions, forces, velocities and masses, as well as the number of degrees of freedom
        seen by each chain link.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on the
                                                                 time step, system, etc.
        """
        # Determine integration step via multi step and Yoshida Suzuki weights
        integration_weights = YSWeights(self.device).get_weights(self.integration_order)
        self.ys_time_step = (
            simulator.integrator.time_step * integration_weights / self.multi_step
        )

        # Determine internal degrees of freedom
        self.degrees_of_freedom = 3 * simulator.system.n_atoms.float()[None, :]

        # Set up internal variables
        self._init_thermostat_variables()
        self._init_barostat_variables()

    def _init_thermostat_variables(self):

        # Set up thermostat masses
        self.t_masses = torch.zeros(
            self.n_replicas, self.n_molecules, self.chain_length, device=self.device
        )
        self.t_masses_cell = torch.zeros_like(self.t_masses)

        # Get masses of innermost thermostat
        self.t_masses[..., 0] = (
            self.degrees_of_freedom * self.kb_temperature / self.frequency ** 2
        )
        self.t_masses_cell[..., 0] = 9.0 * self.kb_temperature / self.frequency ** 2
        # Set masses of remaining thermostats
        self.t_masses[..., 1:] = self.kb_temperature / self.frequency ** 2
        self.t_masses_cell[..., 1:] = self.kb_temperature / self.frequency ** 2

        # Thermostat variables for particles
        self.t_velocities = torch.zeros_like(self.t_masses)
        self.t_forces = torch.zeros_like(self.t_masses)

        # Thermostat variables for cell
        self.t_velocities_cell = torch.zeros_like(self.t_masses_cell)
        self.t_forces_cell = torch.zeros_like(self.t_masses_cell)

    def _init_barostat_variables(self):
        # Set barostat masses
        self.b_masses_cell = torch.ones(
            self.n_replicas, self.n_molecules, device=self.device
        )
        self.b_masses_cell[:, :] = (
            (self.degrees_of_freedom + 3)
            * self.kb_temperature
            / self.barostat_frequency ** 2
        )

        # Remaining barostat variables
        self.b_velocities_cell = torch.zeros_like(self.b_masses_cell)
        self.b_forces_cell = torch.zeros_like(self.b_masses_cell)

    def _apply_barostat(self, simulator):

        self._init_kinetic_energy(simulator.system)
        kinetic_energy = self._compute_kinetic_energy(simulator.system)
        pressure, volume = self._compute_pressure(simulator.system)

        self._update_forces_barostat(kinetic_energy, pressure, volume)
        self._update_forces_thermostat(kinetic_energy)

        for _ in range(self.multi_step):
            for idx_ys in range(self.integration_order):
                time_step = self.ys_time_step[idx_ys]

                self._chain_forward(time_step)
                self._update_box_velocities(time_step)

                # Update the momenta of the particles (accumulate scaling)
                self._update_particle_momenta(time_step, simulator.system)

                # Recompute kinetic energy
                kinetic_energy = self._compute_kinetic_energy(simulator.system)

                # Update the box velocities
                self._update_forces_barostat(kinetic_energy, pressure, volume)
                self._update_box_velocities(time_step)

                # Update the thermostat force
                self._update_forces_thermostat(kinetic_energy)

                # Update velocities and forces of remaining thermostats
                self._chain_backward(time_step)

    def _init_kinetic_energy(self, system):
        self.scaling = 1.0
        # R x M
        self.kinetic_energy = 2.0 * system.kinetic_energy

    def _compute_kinetic_energy(self, system):
        return self.kinetic_energy

    def _compute_kinetic_energy_cell(self):
        return self.b_masses_cell * self.b_velocities_cell ** 2

    def _compute_pressure(self, system):
        # Get the pressure (R x M)
        pressure = system.compute_pressure(kinetic_component=False, tensor=False)[
            ..., 0
        ]
        # Get the volume (R x M)
        volume = system.volume[..., 0]
        return pressure, volume

    def _chain_forward(self, time_step):
        # Update velocities of outermost bath
        self.t_velocities[..., -1] += 0.25 * self.t_forces[..., -1] * time_step
        self.t_velocities_cell[..., -1] += (
            0.25 * self.t_forces_cell[..., -1] * time_step
        )

        # Update the velocities moving through the beads of the chain
        for chain in range(self.chain_length - 2, -1, -1):
            t_coeff = torch.exp(-0.125 * time_step * self.t_velocities[..., chain + 1])
            b_coeff = torch.exp(
                -0.125 * time_step * self.t_velocities_cell[..., chain + 1]
            )

            self.t_velocities[..., chain] = (
                self.velocities[..., chain] * t_coeff ** 2
                + 0.25 * self.t_forces[..., chain] * t_coeff * time_step
            )
            self.t_velocities_cell[..., chain] = (
                self.velocities[..., chain] * b_coeff ** 2
                + 0.25 * self.t_forces_cell[..., chain] * b_coeff * time_step
            )

    def _update_box_velocities(self, time_step):
        b_factor = torch.exp(-0.125 * time_step * self.t_velocities_cell)
        # TODO: Check where symmetrisation is necessary
        self.b_velocities_cell = (
            b_factor ** 2 * self.b_velocities_cell
            + 0.25 * time_step * self.t_forces_cell * b_factor
        )

    def _update_particle_momenta(self, time_step, system):
        scaling = torch.exp(
            -0.5
            * time_step
            * (
                self.t_velocities[..., 0]
                + self.b_velocities_cell * (1 + 1 / self.degrees_of_freedom)
            )
        )
        # TODO: Accumulate instead of doing it here?
        system.momenta *= scaling[:, :, None, None]
        self.kinetic_energy *= scaling ** 2

    def _chain_backward(self, time_step):
        # Update the thermostat velocities
        for chain in range(self.chain_length - 1):
            t_coeff = torch.exp(-0.125 * time_step * self.t_velocities[..., chain + 1])
            b_coeff = torch.exp(
                -0.125 * time_step * self.t_velocities_cell[..., chain + 1]
            )

            self.t_velocities[..., chain] = (
                self.velocities[..., chain] * t_coeff ** 2
                + 0.25 * self.t_forces[..., chain] * t_coeff * time_step
            )
            self.t_velocities_cell[..., chain] = (
                self.velocities[..., chain] * b_coeff ** 2
                + 0.25 * self.t_forces_cell[..., chain] * b_coeff * time_step
            )

            # Update forces through chain
            self.t_forces[..., chain + 1] = (
                self.t_masses[..., chain] * self.t_velocities[..., chain] ** 2
                - self.kb_temperature
            ) / self.t_masses[..., chain + 1]
            self.t_forces_cell[..., chain + 1] = (
                self.t_masses_cell[..., chain] * self.t_velocities_cell[..., chain] ** 2
                - self.kb_temperature
            ) / self.t_masses_cell[..., chain + 1]

        # Update velocities of outermost thermostat
        self.t_velocities[..., -1] += 0.25 * self.t_forces[..., -1] * time_step
        self.t_velocities_cell[..., -1] += (
            0.25 * self.t_forces_cell[..., -1] * time_step
        )

    def _update_forces_thermostat(self, kinetic_energy):
        # Compute forces on thermostat (R x M)
        self.t_forces[..., 0] = (
            kinetic_energy - self.degrees_of_freedom * self.kb_temperature
        ) / self.t_masses[..., 0]

        # Get kinetic energy of barostat (R x M)
        kinetic_energy_cell = self._compute_kinetic_energy_cell()
        # Compute forces on cell thermostat
        self.t_forces_cell[..., 0] = (
            kinetic_energy_cell - self.kb_temperature
        ) / self.t_masses_cell[..., 0]

    def _update_forces_barostat(self, kinetic_energy, pressure, volume):
        self.b_forces_cell = (
            (1.0 + 3.0 / self.degrees_of_freedom) * kinetic_energy
            + 3.0 * volume * (pressure - self.target_pressure) / self.b_masses_cell
        )

    def propagate_system(self, system):
        # Update the particle positions
        scaled_velocity = self.time_step * self.b_velocities_cell
        a_coeff = torch.exp(0.5 * scaled_velocity)[:, :, None, None]
        b_coeff = a_coeff * self.sinhdx.f(0.5 * scaled_velocity)[:, :, None, None]
        system.positions = (
            system.positions * a_coeff ** 2
            + system.momenta / system.masses * b_coeff * self.time_step
        )

        # Scale the cells
        cell_coeff = torch.exp(scaled_velocity)[:, :, None, None]
        system.cells = system.cells * cell_coeff

    @property
    def state_dict(self):
        # TODO: update this properly
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
            "time_step": self.ys_time_step,
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
        self.ys_time_step = state_dict["time_step"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.n_replicas = state_dict["n_replicas"]
        self.multi_step = state_dict["multi_step"]
        self.integration_order = state_dict["integration_order"]

        self.initialized = True


class NHCBarostatAnisotropic(NHCBarostatIsotropic):
    """
    Nose Hoover chain thermostat/barostat for isotropic cell fluctuations.

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
        target_pressure,
        temperature_bath,
        time_constant,
        time_constant_barostat=None,
        chain_length=3,
        multi_step=2,
        integration_order=3,
        detach=True,
    ):
        super(NHCBarostatAnisotropic, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            time_constant=time_constant,
            time_constant_barostat=time_constant_barostat,
            chain_length=chain_length,
            multi_step=multi_step,
            integration_order=integration_order,
            detach=detach,
        )

    def _init_barostat_variables(self):
        # Set barostat masses
        self.b_masses_cell = torch.ones(
            self.n_replicas, self.n_molecules, 1, 1, device=self.device
        )
        self.b_masses_cell[..., 0, 0] = (
            (self.degrees_of_freedom + 3)
            * self.kb_temperature
            / self.barostat_frequency ** 2
        ) / 3.0

        # Remaining barostat variables
        self.b_velocities_cell = torch.zeros(self.n_replicas, self.n_molecules, 3, 3)
        self.b_forces_cell = torch.zeros_like(self.b_velocities_cell)

        # Auxiliary identity matrix for broadcasting
        self.aux_eye = torch.eye(3, device=self.device)[None, None, :, :]

    def _init_kinetic_energy(self, system):
        pass

    def _compute_kinetic_energy(self, system):
        # Here we need the full tensor (R x M x 3 x 3)
        # Kinetic energy can be computed as 1/3 Tr[Etens]
        kinetic_energy_tensor = 2.0 * system.kinetic_energy_tensor
        return kinetic_energy_tensor

    def _compute_kinetic_energy_cell(self):
        b_cell_sq = torch.matmul(
            self.b_velocities_cell.transpose(2, 3), self.b_velocities_cell
        )
        # Einsum computes the trace
        return (
            self.b_masses_cell * torch.einsum("abii->ab", b_cell_sq)[:, :, None, None]
        )

    def _compute_pressure(self, system):
        # Get the pressure (R x M x 3 x 3)
        pressure = system.compute_pressure(kinetic_component=False, tensor=True)
        # Get the volume (R x M x 1 x 1)
        volume = system.volume[..., None]
        return pressure, volume

    def _compute_vtemp(self):
        vtemp = (
            self.b_velocities_cell
            + (
                torch.einsum("abii->ab", self.b_velocities_cell)
                / self.degrees_of_freedom
                + self.t_velocities[..., 0]
                + self.t_velocities_cell[..., 0]
            )[:, :, None, None]
            * self.aux_eye
        )
        return vtemp

    def _update_particle_momenta(self, time_step, system):
        # Compute auxiliary velocity tensor for propagation
        vtemp = self._compute_vtemp()

        # Compute eigenvectors and values for matrix exponential operator
        # eigval -> (R x M x 3)
        # eigvec -> (R x M x 3 x 3)
        eigval, eigvec = torch.symeig(vtemp, eigenvectors=True)
        operator = torch.exp(-0.5 * eigval * self.time_step)[:, :, None, :]

        # The following procedure computes the matrix exponential of vtemp and applies it to
        # the momenta.
        # p' = p * c
        momenta_tmp = torch.matmul(system.momenta, eigvec)
        # Multiply by operator
        momenta_tmp = momenta_tmp * operator
        # Transform back
        # p = p' * c.T
        system.momenta = torch.matmul(momenta_tmp, eigvec.transpose(2, 3))

    def _update_forces_thermostat(self, kinetic_energy):
        # Compute Ekin from tensor
        kinetic_energy = torch.einsum("abii->ab", kinetic_energy) / 3.0

        # Compute forces on thermostat (R x M)
        self.t_forces[..., 0] = (
            kinetic_energy - self.degrees_of_freedom * self.kb_temperature
        ) / self.t_masses[..., 0]

        # Get kinetic energy of barostat (R x M)
        kinetic_energy_cell = self._compute_kinetic_energy_cell()
        # Compute forces on cell thermostat
        self.t_forces_cell[..., 0] = (
            kinetic_energy_cell - 9 * self.kb_temperature
        ) / self.t_masses_cell[..., 0]

    def _update_forces_barostat(self, kinetic_energy, pressure, volume):
        kinetic_energy_scalar = (
            torch.einsum("abii->ab", kinetic_energy)[:, :, None, None] / 3.0
        )

        self.b_forces_cell = (
            1.0
            / self.degrees_of_freedom[:, :, None, None]
            * kinetic_energy_scalar
            * self.aux_eye
            + kinetic_energy
            + volume
            * (pressure - self.aux_eye * self.target_pressure)
            / self.b_masses_cell
        )

    def propagate_system(self, system):
        # Compute auxiliary velocity tensor for propagation
        vtemp = self._compute_vtemp()

        # Compute eigenvectors and values for matrix exponential operator
        # eigval -> (R x M x 3)
        # eigvec -> (R x M x 3 x 3)
        eigval, eigvec = torch.symeig(vtemp, eigenvectors=True)

        evaldt2 = 0.5 * eigval[:, :, None, :] * self.time_step

        # Compute exponential operator and sinh(x)/x operator (approximated)
        a_coeff = torch.exp(evaldt2)
        b_coeff = a_coeff * self.sinhdx.f(evaldt2)

        # Transform positons, velocities and cells via the eigenvectors
        positions_tmp = torch.matmul(system.positions, eigvec)
        velocities_tmp = torch.matmul(system.momenta / system.masses, eigvec)
        cells_tmp = torch.matmul(system.cells, eigvec)

        # Apply the propagator to the positions
        positions_tmp = (
            positions_tmp * a_coeff ** 2 + velocities_tmp * b_coeff * self.time_step
        )

        # Apply the propagator to the cells
        cells_tmp = cells_tmp * a_coeff ** 2

        # Transform everything back and update the system
        system.positions = torch.matmul(positions_tmp, eigvec.transpose(2, 3))
        system.cells = torch.matmul(cells_tmp, eigvec.transpose(2, 3))

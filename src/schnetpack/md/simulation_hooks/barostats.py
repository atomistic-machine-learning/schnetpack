"""
This module contains various barostats for controlling the pressure of the system during
molecular dynamics simulations.
"""

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md import Simulator, System

import torch

from schnetpack import units as spk_units

# from schnetpack.md import System, Simulator
from schnetpack.md.simulation_hooks import SimulationHook
from schnetpack.md.utils import StableSinhDiv, YSWeights

__all__ = ["BarostatHook", "NHCBarostatIsotropic", "NHCBarostatAnisotropic"]


class BarostatError(Exception):
    """
    Exception for barostat hooks.
    """

    pass


class BarostatHook(SimulationHook):
    """
    Basic barostat hook for simulator class. This class is initialized based on the simulator and system
    specifications during the first MD step. Barostats are applied before and after each MD step. In addition,
    they modify the update of positions and cells, which is why they have to be used with modified integrators.

    Args:
        target_pressure (float): Target pressure of the system (in bar).
        temperature_bath (float): Target temperature applied to the cell fluctuations (in K).
        time_constant (float): Time constant used for thermostatting if available (in fs).
    """

    ring_polymer = False
    temperature_control = False

    def __init__(
        self, target_pressure: float, temperature_bath: float, time_constant: float
    ):
        super(BarostatHook, self).__init__()
        # Convert pressure from bar to internal units
        self.register_buffer(
            "target_pressure", torch.tensor(target_pressure * spk_units.bar)
        )
        self.register_buffer("temperature_bath", torch.tensor(temperature_bath))
        self.register_buffer(
            "time_constant", torch.tensor(time_constant * spk_units.fs)
        )

        self.register_buffer("_initialized", torch.tensor(False))

        # This should be determined automatically and does not need to be stored in buffer
        self.time_step = None

    @property
    def initialized(self):
        """
        Auxiliary property for easy access to initialized flag used for restarts
        """
        return self._initialized.item()

    @initialized.setter
    def initialized(self, flag):
        """
        Make sure initialized is set to torch.tensor for storage in state_dict.
        """
        self._initialized = torch.tensor(flag)

    def on_simulation_start(self, simulator: Simulator):
        """
        Routine to initialize the barostat based on the current state of the simulator. Reads the device to be uses,
        as well as the number of molecular replicas present in simulator.system. A flag is set so that the barostat
        is not reinitialized upon continuation of the MD.

        Main function is the _init_barostat routine, which takes the simulator as input and must be provided for every
        new barostat.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                                the time step, system, etc.
        """
        self.time_step = simulator.integrator.time_step

        if not self.initialized:
            self._init_barostat(simulator)
            self.initialized = True

    def on_step_begin(self, simulator: Simulator):
        """
        First application of the barostat before the first half step of the dynamics. Must be provided for
        every new barostat. This is e.g. used to update the NHC thermostat chains on particles and cells in the NHC
        barostats

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
        """
        raise NotImplementedError

    def on_step_end(self, simulator: Simulator):
        """
        Second application of the barostat before the first half step of the dynamics. Must be provided for
        every new barostat. This is e.g. used to update the NHC thermostat chains on particles and cells in the NHC
        barostats

        Main function is the _apply_barostat routine, which takes the simulator as input and must be provided for
        every new barostat.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
        """
        raise NotImplementedError

    def _init_barostat(self, simulator: Simulator):
        """
        Dummy routine for initializing a barpstat based on the current simulator. Should be implemented for every
        new barostat. Has access to the information contained in the simulator class, e.g. number of replicas, time
        step, masses of the atoms, etc.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
        """
        raise NotImplementedError

    def propagate_main_step(self, system: System):
        """
        Propagate the system under the conditions imposed by the barostat. This has to be adapted to the specific
        barostat algorithm and should propagate the positons, cells and momenta (for RPMD). Defaults to classic Verlet.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        system.positions = (
            system.positions + self.time_step * system.momenta / system.masses
        )

    def propagate_half_step(self, system: System):
        """
        Propagate system under barostat conditions during half steps typically used for momenta. Should be adapted for
        each barostat (e.g. NHC propagates paricle and barostat momenta due to barostat actions. Defaults to classic
        Verlet.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        system.momenta = system.momenta + 0.5 * system.forces * self.time_step


class NHCBarostatIsotropic(BarostatHook):
    """
    Nose Hoover chain thermostat/barostat for isotropic cell fluctuations. This barostat already contains a built in
    thermostat, so no further temperature control is necessary. As suggested in [#nhc_barostat1]_, two separate chains
    are used to thermostat particle and cell momenta.

    Args:
        target_pressure (float): Target pressure in bar.
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Particle thermostat time constant in fs
        time_constant_cell (float): Cell thermostat time constant in fs. If None is given (default), the same time
                                    constant as for the thermostat component is used.
        time_constant_barostat (float): Barostat time constant in fs. If None is given (default), the same time constant
                                        as for the thermostat component is used.
        chain_length (int): Number of Nose-Hoover thermostats applied in the chain.
        multi_step (int): Number of steps used for integrating the NH equations of motion (default=2)
        integration_order (int): Order of the Yoshida-Suzuki integrator used for propagating the thermostat (default=3).
        massive (bool): Apply individual thermostat chains to all particle degrees of freedom (default=False).

    References
    ----------
    .. [#nhc_barostat1] Martyna, Tuckerman, Tobias, Klein:
       Explicit reversible integrators for extended systems dynamics.
       Molecular Physics, 87(5), 1117-1157. 1996.
    """

    temperature_control = True
    ring_polymer = False

    def __init__(
        self,
        target_pressure: float,
        temperature_bath: float,
        time_constant: float,
        time_constant_cell: Optional[float] = None,
        time_constant_barostat: Optional[float] = None,
        chain_length: Optional[int] = 4,
        multi_step: Optional[int] = 4,
        integration_order: Optional[int] = 7,
        massive: Optional[bool] = False,
    ):
        super(NHCBarostatIsotropic, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            time_constant=time_constant,
        )
        # Thermostat, cell thermostat and barostat frequencies
        self.register_buffer("frequency", 1.0 / self.time_constant)

        if time_constant_cell is None:
            cell_frequency = self.frequency
        else:
            cell_frequency = torch.tensor(1.0 / (time_constant_cell * spk_units.fs))

        if time_constant_barostat is None:
            barostat_frequency = self.frequency
        else:
            barostat_frequency = torch.tensor(
                1.0 / (time_constant_barostat * spk_units.fs)
            )

        self.register_buffer("cell_frequency", cell_frequency)
        self.register_buffer("barostat_frequency", barostat_frequency)

        # Cpmpute kBT, since it will be used a lot
        self.register_buffer("kb_temperature", self.temperature_bath * spk_units.kB)

        # Propagation parameters
        self.register_buffer("chain_length", torch.tensor(chain_length))
        self.register_buffer("massive", torch.tensor(massive))
        self.register_buffer("multi_step", torch.tensor(multi_step))
        self.register_buffer("integration_order", torch.tensor(integration_order))

        self.register_uninitialized_buffer("ys_time_step")

        # Find out number of particles (depends on whether massive or not)
        self.register_uninitialized_buffer("degrees_of_freedom")
        self.register_uninitialized_buffer("degrees_of_freedom_cell")
        self.register_uninitialized_buffer("degrees_of_freedom_particles")

        # Thermostat variables for particles
        self.register_uninitialized_buffer("t_velocities")
        self.register_uninitialized_buffer("t_positions")
        self.register_uninitialized_buffer("t_forces")
        self.register_uninitialized_buffer("t_masses")

        # Thermostat variables for cell
        self.register_uninitialized_buffer("t_velocities_cell")
        self.register_uninitialized_buffer("t_positions_cell")
        self.register_uninitialized_buffer("t_forces_cell")
        self.register_uninitialized_buffer("t_masses_cell")

        # Barostat variables
        self.register_uninitialized_buffer("b_velocities_cell")
        self.register_uninitialized_buffer("b_positions_cell")
        self.register_uninitialized_buffer("b_forces_cell")
        self.register_uninitialized_buffer("b_masses_cell")

        # Stable sinh(x)/x approximation
        self.sinhdx = StableSinhDiv()

    def _init_barostat(self, simulator: Simulator):
        """
        Initialize the thermostat positions, forces, velocities and masses, as well as the number of degrees of freedom
        seen by each chain link. In the same manner, all quantities required for the barostat are initialized.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                    the time step, system, etc.
        """
        # Determine integration step via multi step and Yoshida Suzuki weights
        integration_weights = (
            YSWeights()
            .get_weights(self.integration_order.item())
            .to(simulator.device, simulator.dtype)
        )

        self.ys_time_step = (
            simulator.integrator.time_step * integration_weights / self.multi_step
        )

        # Determine internal degrees of freedom for barostat
        self.degrees_of_freedom = (
            3.0 * simulator.system.n_atoms.to(simulator.dtype)[None, :, None]
        )

        # Determine degrees of freedom for particle thermostats (depends on massive)
        n_replicas = simulator.system.n_replicas
        n_molecules = simulator.system.n_molecules
        n_atoms_total = simulator.system.total_n_atoms

        if self.massive:
            state_dimension = (n_replicas, n_atoms_total, 3, self.chain_length)
            self.degrees_of_freedom_particles = torch.ones(
                (n_replicas, n_atoms_total, 3),
                device=simulator.device,
                dtype=simulator.dtype,
            )
        else:
            state_dimension = (n_replicas, n_molecules, 1, self.chain_length)
            self.degrees_of_freedom_particles = self.degrees_of_freedom

        # Set up internal variables
        self._init_barostat_variables(simulator)
        self._init_thermostat_variables(state_dimension, simulator)

    def _init_barostat_variables(self, simulator: Simulator):
        """
        Initialize all quantities required for the barostat component.
        """
        # Set barostat masses
        self.b_masses_cell = torch.ones(
            (simulator.system.n_replicas, simulator.system.n_molecules, 1),
            device=simulator.device,
            dtype=simulator.dtype,
        )
        self.b_masses_cell = (
            (self.degrees_of_freedom + 3)
            * self.kb_temperature
            / self.barostat_frequency**2
        )
        # Remaining barostat variables
        self.b_velocities_cell = torch.zeros_like(self.b_masses_cell)
        self.b_forces_cell = torch.zeros_like(self.b_masses_cell)

        # Set cell degrees of freedom (1 for isotropic, 9 for full, 9-3 for full with symmetric pressure (no rotations)
        self.degrees_of_freedom_cell = torch.tensor(
            1, dtype=simulator.dtype, device=simulator.device
        )

    def _init_thermostat_variables(
        self, state_dimension: Tuple[int, int, int, int], simulator: Simulator
    ):
        """
        Initialize all quantities required for the two thermostat chains on the particles and cells.
        Args:
            state_dimension (tuple): Size of the thermostat states. This is used to differentiate between the massive
                                     and the standard algorithm
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Set up thermostat masses
        self.t_masses = torch.zeros(
            state_dimension, device=simulator.device, dtype=simulator.dtype
        )

        self.t_masses_cell = torch.zeros(
            (
                simulator.system.n_replicas,
                simulator.system.n_molecules,
                1,
                self.chain_length,
            ),
            device=simulator.device,
            dtype=simulator.dtype,
        )

        # Get masses of innermost thermostat
        self.t_masses[..., 0] = (
            self.degrees_of_freedom_particles * self.kb_temperature / self.frequency**2
        )

        # Get masses of cell
        self.t_masses_cell[..., 0] = (
            self.degrees_of_freedom_cell * self.kb_temperature / self.cell_frequency**2
        )

        # Set masses of remaining thermostats
        self.t_masses[..., 1:] = self.kb_temperature / self.frequency**2
        self.t_masses_cell[..., 1:] = self.kb_temperature / self.cell_frequency**2

        # Thermostat variables for particles
        self.t_velocities = torch.zeros_like(self.t_masses)
        self.t_forces = torch.zeros_like(self.t_masses)

        # Thermostat variables for cell
        self.t_velocities_cell = torch.zeros_like(self.t_masses_cell)
        self.t_forces_cell = torch.zeros_like(self.t_masses_cell)

        # Positions for conservation
        # self.t_positions = torch.zeros_like(self.t_masses, device=self.device)
        # self.t_positions_cell = torch.zeros_like(self.t_masses_cell, device=self.device)

    def on_step_begin(self, simulator: Simulator):
        """
        Propagate the thermostat chains on particles and cell and update cell velocities using the barostat forces.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        self._update_thermostat(simulator)
        self._update_barostat(simulator)

    def on_step_end(self, simulator: Simulator):
        """
        Propagate the thermostat chains on particles and cell and update cell velocities using the barostat forces.
        Order is reversed in order to preserve symmetric splitting of the overall propagator.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        self._update_barostat(simulator)
        self._update_thermostat(simulator)

    def _update_thermostat(self, simulator: Simulator):
        """
        Apply the thermostat chains to the system and cell momenta. This is done by computing forces of the innermost
        thermostats, propagating the chain forward, updating the box velocities, particle momenta and associated
        energies. Based on this, the kinetic energies and forces can be updated, which are the propagated backward
        through the chain.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                                the time step, system, etc.
        """
        # Get kinetic energies for cell and particles
        kinetic_energy_particles = self._compute_kinetic_energy_particles(
            simulator.system
        )
        kinetic_energy_cell = self._compute_kinetic_energy_cell()

        # Update the innermost  thermostat forces using the current kinetic energies
        self._update_inner_t_forces(kinetic_energy_particles)
        self._update_inner_t_forces_cell(kinetic_energy_cell)

        # Initialize scaling factor which will be accumulated
        scaling_thermostat_particles = torch.ones_like(self.t_velocities[..., 0])
        scaling_thermostat_cell = torch.ones_like(self.t_velocities_cell[..., 0])

        # Multistep and YS procedure for propagating the thermostat operators
        for _ in range(self.multi_step):
            for idx_ys in range(self.integration_order):
                # Determine integration time step
                time_step = self.ys_time_step[idx_ys]

                # Propagate the chain inward (forces are not updated, the innermost forces are already initialized)
                self._chain_forward(time_step)

                # Accumulate scaling
                scaling_thermostat_particles *= torch.exp(
                    -0.5 * time_step * self.t_velocities[..., 0]
                )
                scaling_thermostat_cell *= torch.exp(
                    -0.5 * time_step * self.t_velocities_cell[..., 0]
                )

                # Recompute forces using scaling to update the kinetic energies
                self._update_inner_t_forces(
                    kinetic_energy_particles * scaling_thermostat_particles**2
                )
                self._update_inner_t_forces_cell(
                    kinetic_energy_cell * scaling_thermostat_cell**2
                )

                # Update velocities and forces of remaining thermostats
                self._chain_backward(time_step)

        # Update the momenta of the particles
        if not self.massive:
            scaling_thermostat_particles = simulator.system.expand_atoms(
                scaling_thermostat_particles
            )
        simulator.system.momenta = (
            simulator.system.momenta * scaling_thermostat_particles
        )

        # Update cell momenta
        self._scale_cell(scaling_thermostat_cell)

    def _scale_cell(self, scaling_thermostat_cell: torch.tensor):
        """
        Auxiliary routine for scaling the cell, since the scaling factor will be missing one dimension in the
        anisotropic barostat

        Args:
            scaling_thermostat_cell (torch.tensor): Accumulated scaling for cell velocities
        """
        self.b_velocities_cell = self.b_velocities_cell * scaling_thermostat_cell

    def _compute_kinetic_energy_particles(self, system: System):
        """
        Compute the current kinetic energy for the particles, depending on whether massive or normal thermostatting is
        used.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        Returns:
            torch.tensor: Current kinetic energy of the particles.
        """
        if self.massive:
            kinetic_energy_particles = system.momenta**2 / system.masses
        else:
            kinetic_energy_particles = 2.0 * system.kinetic_energy

        return kinetic_energy_particles

    def _compute_kinetic_energy_cell(self):
        """
        Compute the kinetic energy of the cells.

        Returns:
            torch.tensor: Kinetic energy associated with the cells.
        """
        return self.b_masses_cell * self.b_velocities_cell**2

    def _update_inner_t_forces(self, kinetic_energy_particles: torch.tensor):
        """
        Update the forces acting on the innermost chain of the particle thermostat.

        Args:
            kinetic_energy_particles (torch.tensor): kinetic energy of the particles
        """
        self.t_forces[..., 0] = (
            kinetic_energy_particles
            - self.degrees_of_freedom_particles * self.kb_temperature
        ) / self.t_masses[..., 0]

    def _update_inner_t_forces_cell(self, kinetic_energy_cell: torch.tensor):
        """
        Update the forces acting on the innermost chain of the cell thermostat.

        Args:
            kinetic_energy_cell (torch.tensor): kinetic energy of the cell
        """
        self.t_forces_cell[..., 0] = (
            kinetic_energy_cell - self.degrees_of_freedom_cell * self.kb_temperature
        ) / self.t_masses_cell[..., 0]

    def _chain_forward(self, time_step: float):
        """
        Forward propagation of the two Nose-Hoover chains attached to particles and cells.
        Force updates are not required here, as long as the innermost force is precomputed, since forces are effectively
        taken from the previous step and everything gets gets overwritten by the force update in the backward chain.

        Args:
            time_step (float): Current timestep considering YS and multi-timestep integration.
        """
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
                self.t_velocities[..., chain] * t_coeff**2
                + 0.25 * self.t_forces[..., chain] * t_coeff * time_step
            )
            self.t_velocities_cell[..., chain] = (
                self.t_velocities_cell[..., chain] * b_coeff**2
                + 0.25 * self.t_forces_cell[..., chain] * b_coeff * time_step
            )

    def _chain_backward(self, time_step: float):
        """
        Backward propagation of the two Nose-Hoover chains attached to particles and cells. In addition, the repsective
        thermostat forces are updated.

        Args:
            time_step (float): Current timestep considering YS and multi-timestep integration.
        """
        # Update the thermostat velocities
        for chain in range(self.chain_length - 1):
            t_coeff = torch.exp(-0.125 * time_step * self.t_velocities[..., chain + 1])
            b_coeff = torch.exp(
                -0.125 * time_step * self.t_velocities_cell[..., chain + 1]
            )

            self.t_velocities[..., chain] = (
                self.t_velocities[..., chain] * t_coeff**2
                + 0.25 * self.t_forces[..., chain] * t_coeff * time_step
            )
            self.t_velocities_cell[..., chain] = (
                self.t_velocities_cell[..., chain] * b_coeff**2
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

    def _update_barostat(self, simulator: Simulator):
        # Get new barostat forces
        self._update_b_forces(simulator.system)

        # Update the cell velocities
        self.b_velocities_cell = (
            self.b_velocities_cell + 0.5 * self.time_step * self.b_forces_cell
        )

    def _update_b_forces(self, system: System):
        """
        Update the barostat forces using kinetic energy, current pressure and volume of the system.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Get the pressure (R x M x 1)
        pressure = system.compute_pressure(kinetic_component=False, tensor=False)
        # Get the volume (R x M x 1)
        volume = system.volume
        # Get the kinetic energy
        kinetic_energy = 2.0 * system.kinetic_energy

        self.b_forces_cell = (
            (1.0 + 3.0 / self.degrees_of_freedom) * kinetic_energy
            + 3.0 * volume * (pressure - self.target_pressure)
        ) / self.b_masses_cell

    def propagate_main_step(self, system: System):
        """
        Main routine for propagating the system positions and cells. Since this is modified, no conventional velocity
        verlet integrator can be used.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        scaled_velocity = self.time_step * self.b_velocities_cell
        # Compute exponential coefficient
        a_coeff = torch.exp(0.5 * scaled_velocity)
        # Compute sinh(x)/x term
        b_coeff = a_coeff * self.sinhdx.f(0.5 * scaled_velocity)

        # Update the particle positions
        system.positions = (
            system.positions * system.expand_atoms(a_coeff**2)
            + system.momenta
            / system.masses
            * system.expand_atoms(b_coeff)
            * self.time_step
        )

        # Scale the cells (propagation is in logarithmic space)
        cell_coeff = torch.exp(scaled_velocity)[..., None]
        system.cells = system.cells * cell_coeff

    def propagate_half_step(self, system: System):
        """
        Main routine for propagating the system momenta. Since this is modified, no conventional velocity
        verlet integrator can be used.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Compute basic argument
        scaled_velocity = (
            0.25
            * self.time_step
            * self.b_velocities_cell
            * (1.0 + 3.0 / self.degrees_of_freedom)
        )
        # Compute exponential coefficient
        a_coeff = torch.exp(-scaled_velocity)
        # Compute sinh(x)/x term
        b_coeff = a_coeff * self.sinhdx.f(scaled_velocity)

        # Update the momenta (using half timestep)
        system.momenta = (
            system.momenta * system.expand_atoms(a_coeff**2)
            + system.forces * system.expand_atoms(b_coeff) * self.time_step * 0.5
        )

    # def compute_conserved(self, system):
    #    """
    #    Computed the conserved quantity. For debug purposes only.
    #    """
    #    conserved = (
    #            system.kinetic_energy[..., None, None]
    #            + system.energies[..., None, None]
    #            + 0.5 * torch.sum(self.t_velocities ** 2 * self.t_masses, 2)
    #            + 0.5 * torch.sum(self.t_velocities_cell ** 2 * self.t_masses_cell, 2)
    #            + 0.5 * self.b_velocities_cell ** 2 * self.b_masses_cell
    #            + self.degrees_of_freedom * self.kb_temperature * self.t_positions[..., 0]
    #            + self.kb_temperature * self.t_positions_cell[..., 0]
    #            + self.kb_temperature * torch.sum(self.t_positions[..., 1:], 2)
    #            + self.kb_temperature * torch.sum(self.t_positions_cell[..., 1:], 2)
    #            + self.target_pressure * system.volume
    #    )
    #    return conserved


class NHCBarostatAnisotropic(NHCBarostatIsotropic):
    """
    Nose Hoover chain thermostat/barostat for anisotropic cell fluctuations. This barostat already contains a built in
    thermostat, so no further temperature control is necessary. As suggested in [#nhc_barostat1]_, two separate chains
    are used to thermostat particle and cell momenta.

    Args:
        target_pressure (float): Target pressure in bar.
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Particle thermostat time constant in fs
        time_constant_cell (float): Cell thermostat time constant in fs. If None is given (default), the same time
                                    constant as for the thermostat component is used.
        time_constant_barostat (float): Barostat time constant in fs. If None is given (default), the same time constant
                                        as for the thermostat component is used.
        chain_length (int): Number of Nose-Hoover thermostats applied in the chain.
        multi_step (int): Number of steps used for integrating the NH equations of motion (default=2)
        integration_order (int): Order of the Yoshida-Suzuki integrator used for propagating the thermostat (default=3).
        massive (bool): Apply individual thermostat chains to all particle degrees of freedom (default=False).

    References
    ----------
    .. [#nhc_barostat1] Martyna, Tuckerman, Tobias, Klein:
       Explicit reversible integrators for extended systems dynamics.
       Molecular Physics, 87(5), 1117-1157. 1996.
    """

    temperature_control = True
    ring_polymer = False

    def __init__(
        self,
        target_pressure: float,
        temperature_bath: float,
        time_constant: float,
        time_constant_cell: Optional[float] = None,
        time_constant_barostat: Optional[float] = None,
        chain_length: Optional[int] = 4,
        multi_step: Optional[int] = 4,
        integration_order: Optional[int] = 7,
        massive: Optional[bool] = False,
    ):
        super(NHCBarostatAnisotropic, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            time_constant=time_constant,
            time_constant_cell=time_constant_cell,
            time_constant_barostat=time_constant_barostat,
            chain_length=chain_length,
            multi_step=multi_step,
            integration_order=integration_order,
            massive=massive,
        )

    def _init_barostat_variables(self, simulator: Simulator):
        """
        Initialize all quantities required for the barostat component.
        """
        # Set barostat masses
        self.b_masses_cell = torch.ones(
            (simulator.system.n_replicas, simulator.system.n_molecules, 1),
            device=simulator.device,
            dtype=simulator.dtype,
        )
        # Modified due to full cell
        self.b_masses_cell = (
            (self.degrees_of_freedom + 3)
            * self.kb_temperature
            / self.barostat_frequency**2
            / 3.0
        )
        # Remaining barostat variables (forces and velocities are now 3 x 3)
        self.b_velocities_cell = torch.zeros(
            (simulator.system.n_replicas, simulator.system.n_molecules, 3, 3),
            device=simulator.device,
            dtype=simulator.dtype,
        )
        self.b_forces_cell = torch.zeros_like(self.b_velocities_cell)

        # Auxiliary identity matrix for broadcasting
        self.register_buffer(
            "aux_eye",
            torch.eye(3, device=simulator.device, dtype=simulator.dtype)[
                None, None, :, :
            ],
        )

        # Set cell degrees of freedom (1 for isotropic, 9 for full, 9-3 for full with symmetric pressure (no rotations)
        self.degrees_of_freedom_cell = torch.tensor(
            6, dtype=simulator.dtype, device=simulator.device
        )

    def _scale_cell(self, scaling_thermostat_cell: torch.tensor):
        """
        Auxiliary routine for scaling the cell, here the scaling factor needs one additional dimension compared to the
        isotropic case.

        Args:
            scaling_thermostat_cell (torch.tensor): Accumulated scaling for cell velocities
        """
        self.b_velocities_cell = (
            self.b_velocities_cell * scaling_thermostat_cell[..., None]
        )

    def _compute_kinetic_energy_cell(self):
        """
        Compute the kinetic energy of the cells.

        Returns:
            torch.tensor: Kinetic energy associated with the cells.
        """
        b_velocities_cell_sq = torch.sum(
            self.b_velocities_cell**2, dim=(2, 3), keepdim=True
        ).squeeze(-1)
        return self.b_masses_cell * b_velocities_cell_sq

    def _update_b_forces(self, system: System):
        """
        Update the barostat forces using kinetic energy, current pressure and volume of the system.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Get the pressure (R x M x 1)
        pressure = system.compute_pressure(kinetic_component=True, tensor=True)
        # Get the volume (R x M x 1)
        volume = system.volume
        # Get the kinetic energy
        kinetic_energy = 2.0 * system.kinetic_energy

        self.b_forces_cell = (
            volume[..., None] * (pressure - self.aux_eye * self.target_pressure)
            + kinetic_energy[..., None]
            / self.degrees_of_freedom[..., None]
            * self.aux_eye
        ) / self.b_masses_cell[..., None]

    def propagate_main_step(self, system: System):
        """
        Main routine for propagating the system positions and cells. Since this is modified, no conventional velocity
        verlet integrator can be used.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        eigval_b_velocities, eigvec_b_velocities = torch.symeig(
            self.b_velocities_cell, eigenvectors=True
        )

        scaled_velocity = eigval_b_velocities * self.time_step
        coeff_a = torch.exp(0.5 * scaled_velocity)[..., None] * self.aux_eye
        coeff_b = coeff_a * self.sinhdx.f(0.5 * scaled_velocity)[..., None]

        # Construct matrix operators and update positions using positions and momenta
        operator_a = torch.matmul(
            eigvec_b_velocities,
            torch.matmul(coeff_a**2, eigvec_b_velocities.transpose(2, 3)),
        )
        operator_b = torch.matmul(
            eigvec_b_velocities,
            torch.matmul(coeff_b, eigvec_b_velocities.transpose(2, 3)),
        )

        update_positions = torch.sum(
            system.expand_atoms(operator_a) * system.positions[..., None], dim=2
        )
        update_momenta = torch.sum(
            system.expand_atoms(operator_b)
            * (system.momenta / system.masses)[..., None],
            dim=2,
        )

        system.positions = update_positions + update_momenta * self.time_step

        # Update cells using first operator
        system.cells = torch.matmul(system.cells, operator_a)

    def propagate_half_step(self, system: System):
        """
        Main routine for propagating the system momenta. Since this is modified, no conventional velocity
        verlet integrator can be used.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        eigval_b_velocities, eigvec_b_velocities = torch.symeig(
            self.b_velocities_cell, eigenvectors=True
        )
        # Trace of matrix is sum of eigenvalues
        trace_b_velocities = torch.sum(eigval_b_velocities, dim=2, keepdim=True)

        scaled_velocity = (
            (eigval_b_velocities + trace_b_velocities / self.degrees_of_freedom)
            * self.time_step
            * 0.5
        )

        coeff_a = torch.exp(-0.5 * scaled_velocity)[..., None] * self.aux_eye
        coeff_b = coeff_a * self.sinhdx.f(0.5 * scaled_velocity)[..., None]

        # Construct matrix operators and update positions using positions and momenta
        operator_a = torch.matmul(
            eigvec_b_velocities,
            torch.matmul(coeff_a**2, eigvec_b_velocities.transpose(2, 3)),
        )
        operator_b = torch.matmul(
            eigvec_b_velocities,
            torch.matmul(coeff_b, eigvec_b_velocities.transpose(2, 3)),
        )

        update_momenta = torch.sum(
            system.expand_atoms(operator_a) * system.momenta[..., None], dim=2
        )
        update_forces = torch.sum(
            system.expand_atoms(operator_b) * system.forces[..., None], dim=2
        )

        system.momenta = update_momenta + 0.5 * self.time_step * update_forces

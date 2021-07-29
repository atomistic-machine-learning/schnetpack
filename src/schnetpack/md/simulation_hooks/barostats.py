"""
This module contains various thermostat for regulating the temperature of the system during
molecular dynamics simulations. Apart from standard thermostat for convetional simulations,
a series of special thermostat developed for ring polymer molecular dynamics is also provided.
"""
import torch

from typing import Optional, Tuple
from ase import units as ase_units

from schnetpack import units as spk_units
from schnetpack.md import System, Simulator
from schnetpack.md.simulation_hooks import SimulationHook
from schnetpack.md.utils import StableSinhDiv, YSWeights

__all__ = ["BarostatHook", "NHCBarostatIsotropic"]


class BarostatError(Exception):
    """
    Exception for barostat hooks.
    """

    pass


class BarostatHook(SimulationHook):
    ring_polymer = False
    temperature_control = False
    """
    Basic barostat hook for simulator class. This class is initialized based on the simulator and system
    specifications during the first MD step. Barostats are applied before and after each MD step. In addition,
    they modify the update of positions and cells, which is why they have to be used with modified integrators.

    Args:
        target_pressure (float): Target pressure of the system (in bar).
        temperature_bath (float): Target temperature applied to the cell fluctuations (in K).
        time_constant (float): Time constant used for thermostatting if available (in fs).
    """

    def __init__(
        self, target_pressure: float, temperature_bath: float, time_constant: float
    ):
        super(BarostatHook, self).__init__()
        # Convert pressure from bar to internal units
        self.register_buffer(
            "target_pressure",
            torch.tensor(
                target_pressure
                * spk_units.convert_units(1e5 * ase_units.Pascal, spk_units.pressure)
            ),
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
        First application of the barostat before the first half step of the dynamics. Applies a mask to the system
        momenta in order to avoid problems of e.g. noise added to the zero padded tensors. The detach is carried out
        here.

        Main function is the _apply_barostat routine, which takes the simulator as input and must be provided for
        every new barostat.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
        """
        # Apply thermostat
        self._apply_barostat(simulator)

    def on_step_end(self, simulator: Simulator):
        """
        Second application of the barostat after the second half step of the dynamics. Applies a mask to the system
        momenta in order to avoid problems of e.g. noise added to the zero padded tensors. The detach is carried out
        here.

        Main function is the _apply_barostat routine, which takes the simulator as input and must be provided for
        every new barostat.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
        """
        # Apply thermostat
        self._apply_barostat(simulator)

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

    def _apply_barostat(self, simulator: Simulator):
        """
        Dummy routine for applying the barostat to the system and propagating the barostat. Is called twice each
        simulation time step.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
        """
        raise NotImplementedError

    def propagate_system(self, system: System):
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

    def propagate_barostat_half_step(self, system: System):
        """
        Routine only required for RPMD which propagates the barostat attached to the centroid cell. Is called during
        each half-step before system momenta are propagated.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        print("????????")
        pass


class NHCBarostatIsotropic(BarostatHook):
    temperature_control = True
    ring_polymer = False
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
        detach (bool): Whether the computational graph should be detached after each simulation step. Default is true,
                       should be changed if differentiable MD is desired.

    References
    ----------
    .. [#nhc_barostat1] Martyna, Tuckerman, Tobias, Klein:
       Explicit reversible integrators for extended systems dynamics.
       Molecular Physics, 87(5), 1117-1157. 1996.
    """

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
        self.register_buffer("ys_time_step", None)

        # Find out number of particles (depends on whether massive or not)
        self.register_buffer("degrees_of_freedom", None)
        self.register_buffer("degrees_of_freedom_particles", None)

        # Thermostat variables for particles
        self.register_buffer("t_velocities", None)
        self.register_buffer("t_positions", None)
        self.register_buffer("t_forces", None)
        self.register_buffer("t_masses", None)

        # Thermostat variables for cell
        self.register_buffer("t_velocities_cell", None)
        self.register_buffer("t_positions_cell", None)
        self.register_buffer("t_forces_cell", None)
        self.register_buffer("t_masses_cell", None)

        # Barostat variables
        self.register_buffer("b_velocities_cell", None)
        self.register_buffer("b_positions_cell", None)
        self.register_buffer("b_forces_cell", None)
        self.register_buffer("b_masses_cell", None)

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
            self.degrees_of_freedom_particles = self.degrees_of_freedom  # [..., None]

        # Set up internal variables
        self._init_thermostat_variables(state_dimension, simulator)
        self._init_barostat_variables(simulator)

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
            self.degrees_of_freedom_particles
            * self.kb_temperature
            / self.frequency ** 2
        )

        # Get masses of cell
        self.t_masses_cell[..., 0] = (
            9.0 * self.kb_temperature / self.cell_frequency ** 2
        )

        # Set masses of remaining thermostats
        self.t_masses[..., 1:] = self.kb_temperature / self.frequency ** 2
        self.t_masses_cell[..., 1:] = self.kb_temperature / self.cell_frequency ** 2

        # Thermostat variables for particles
        self.t_velocities = torch.zeros_like(self.t_masses)
        self.t_forces = torch.zeros_like(self.t_masses)

        # Thermostat variables for cell
        self.t_velocities_cell = torch.zeros_like(self.t_masses_cell)
        self.t_forces_cell = torch.zeros_like(self.t_masses_cell)

        # Positions for conservation
        # self.t_positions = torch.zeros_like(self.t_masses, device=self.device)
        # self.t_positions_cell = torch.zeros_like(self.t_masses_cell, device=self.device)

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
            / self.barostat_frequency ** 2
        )
        # Remaining barostat variables
        self.b_velocities_cell = torch.zeros_like(self.b_masses_cell)
        self.b_forces_cell = torch.zeros_like(self.b_masses_cell)

    def _apply_barostat(self, simulator: Simulator):
        """
        Apply the thermostat chains to the system and cell momenta. This is done by computing forces of the innermost
        thermostats, propagating the chain forward, updating the box velocities, particle momenta and associated
        energies. Based on this, the kinetic energies and forces can be updated, which are the propagated backward
        through the chain.

        Args:
            simulator (schnetpack.md.simulator.Simulator): Main simulator class containing information on
                                                                the time step, system, etc.
        """
        self._init_kinetic_energy(simulator.system)
        (
            kinetic_energy_for_thermostat,
            kinetic_energy_for_barostat,
        ) = self._compute_kinetic_energy(simulator.system)
        pressure, volume = self._compute_pressure(simulator.system)

        self._update_forces_barostat(kinetic_energy_for_barostat, pressure, volume)
        self._update_forces_thermostat(kinetic_energy_for_thermostat)

        for _ in range(self.multi_step):
            for idx_ys in range(self.integration_order):
                time_step = self.ys_time_step[idx_ys]

                self._chain_forward(time_step)
                self._update_box_velocities(time_step)

                # Update the momenta of the particles (accumulate scaling)
                self._update_particle_momenta(time_step, simulator.system)

                # Update thermostat positions (Only for debugging via conserved quantity)
                # self.t_positions = self.t_positions + 0.5 * time_step * self.t_velocities
                # self.t_positions_cell = self.t_positions_cell + 0.5 * time_step * self.t_velocities_cell

                # Recompute kinetic energy
                (
                    kinetic_energy_for_thermostat,
                    kinetic_energy_for_barostat,
                ) = self._compute_kinetic_energy(simulator.system)

                # Update the box velocities
                self._update_forces_barostat(
                    kinetic_energy_for_barostat, pressure, volume
                )
                self._update_box_velocities(time_step)

                # Update the thermostat force
                self._update_forces_thermostat(kinetic_energy_for_thermostat)

                # Update velocities and forces of remaining thermostats
                self._chain_backward(time_step)

    def _init_kinetic_energy(self, system: System):
        """
        Auxiliary routine for initializing the kinetic energy and accumulating the scaling factor.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        self.scaling = 1.0
        # R x (M*A) x 3
        self.kinetic_energy = system.momenta ** 2 / system.masses

    def _compute_kinetic_energy(self, system: System):
        """
        Compute the current kinetic energy. Here an internal surrogate is used, which is scaled continuously.
        Since barostat and thermostat updates require different kinetic energy conventions in case of massive
        updates, two separate tensors are computed.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        Returns:
            torch.tensor: Current kinetic energy of the particles.
        """
        kinetic_energy_for_barostat = system.sum_atoms(
            torch.sum(self.kinetic_energy, dim=2, keepdim=True)
        )

        if self.massive:
            kinetic_energy_for_thermostat = self.kinetic_energy
        else:
            # TODO: this this need to be brought from M to MN?
            kinetic_energy_for_thermostat = kinetic_energy_for_barostat

        return kinetic_energy_for_thermostat, kinetic_energy_for_barostat

    def _compute_kinetic_energy_cell(self):
        """
        Compute the kinetic energy of the cells.

        Returns:
            torch.tensor: Kinetic energy associated with the cells.
        """
        return self.b_masses_cell * self.b_velocities_cell ** 2

    def _compute_pressure(self, system: System):
        """
        Routine for computing the current pressure and volume associated with the simulated systems. The stress tensor
        is used for pressure computation.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.

        Returns:
            (torch.Tensor, torch.Tensor): Duple containing the pressures and volumes.
        """
        # Get the pressure (R x M x 1)
        pressure = system.compute_pressure(kinetic_component=False, tensor=False)
        # Get the volume (R x M x 1)
        volume = system.volume
        return pressure, volume

    def _chain_forward(self, time_step: float):
        """
        Forward propagation of the two Nose-Hoover chains attached to particles and cells.

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
                self.t_velocities[..., chain] * t_coeff ** 2
                + 0.25 * self.t_forces[..., chain] * t_coeff * time_step
            )
            self.t_velocities_cell[..., chain] = (
                self.t_velocities_cell[..., chain] * b_coeff ** 2
                + 0.25 * self.t_forces_cell[..., chain] * b_coeff * time_step
            )

    def _update_box_velocities(self, time_step: float):
        """
        Update the velocities of the additional degree of freedom associated with the simulation cells.

        Args:
            time_step (float): Current timestep considering YS and multi-timestep integration.
        """
        b_factor = torch.exp(-0.125 * time_step * self.t_velocities_cell[..., 0])
        self.b_velocities_cell = (
            b_factor ** 2 * self.b_velocities_cell
            + 0.25 * time_step * self.b_forces_cell * b_factor
        )

    def _update_particle_momenta(self, time_step: float, system: System):
        """
        Update the momenta of the particles, as well as the internal kinetic energy surrogate.

        Args:
            time_step (float): Current timestep considering YS and multi-timestep integration.
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        cell_component = self.b_velocities_cell * (1 + 3.0 / self.degrees_of_freedom)

        if self.massive:
            cell_component = system.expand_atoms(cell_component)

        scaling = torch.exp(
            -0.5 * time_step * (self.t_velocities[..., 0] + cell_component)
        )

        # Expand to mn dimension and scale
        if not self.massive:
            scaling = system.expand_atoms(scaling)

        system.momenta = system.momenta * scaling

        # Update kinetic energy
        self.kinetic_energy *= scaling ** 2

    def _chain_backward(self, time_step):
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
                self.t_velocities[..., chain] * t_coeff ** 2
                + 0.25 * self.t_forces[..., chain] * t_coeff * time_step
            )
            self.t_velocities_cell[..., chain] = (
                self.t_velocities_cell[..., chain] * b_coeff ** 2
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

    def _update_forces_thermostat(self, kinetic_energy: torch.tensor):
        """
        Update the forces acting on the two innermost thermostats coupled to the particle and cell momenta.

        Args:
            kinetic_energy (torch.Tensor): Tensor containing the current kinetic energies of the systems.
        """
        # Compute forces on thermostat (R x M)
        self.t_forces[..., 0] = (
            kinetic_energy - self.degrees_of_freedom_particles * self.kb_temperature
        ) / self.t_masses[..., 0]

        # Get kinetic energy of barostat (R x M)
        kinetic_energy_cell = self._compute_kinetic_energy_cell()
        # Compute forces on cell thermostat
        self.t_forces_cell[..., 0] = (
            kinetic_energy_cell - self.kb_temperature
        ) / self.t_masses_cell[..., 0]

    def _update_forces_barostat(
        self, kinetic_energy: torch.tensor, pressure: torch.tensor, volume: torch.tensor
    ):
        """
        Update the forces acting on the barostat coupled to the cell.

        Args:
            kinetic_energy (torch.Tensor): Tensor containing the current kinetic energies of the systems.
            pressure (torch.Tensor): Current pressure of each system.
            volume (torch.Tensor): Current volume of each system.
        """
        self.b_forces_cell = (
            (1.0 + 3.0 / self.degrees_of_freedom) * kinetic_energy
            + 3.0 * volume * (pressure - self.target_pressure)
        ) / self.b_masses_cell

    def propagate_system(self, system: System):
        """
        Main routine for propagating the system positions and cells. Since this is modifed, no conventional velocity
        verlet integrator can be used.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Update the particle positions
        # R x M x 1
        print("I AM HERE AND TERRIFIED")
        scaled_velocity = self.time_step * self.b_velocities_cell
        a_coeff = torch.exp(0.5 * scaled_velocity)
        b_coeff = a_coeff * self.sinhdx.f(0.5 * scaled_velocity)

        # print(a_coeff.shape)
        # print(b_coeff.shape)
        print(a_coeff, "AC")
        print(b_coeff, "BC")

        # Bring to molecule x n_atom dimension and scale
        system.positions = (
            system.positions * system.expand_atoms(a_coeff) ** 2
            + system.momenta
            / system.masses
            * system.expand_atoms(b_coeff)
            * self.time_step
        )

        # Scale the cells
        cell_coeff = torch.exp(scaled_velocity)[..., None]
        print(system.cells)
        system.cells = system.cells * cell_coeff
        print(system.cells)

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

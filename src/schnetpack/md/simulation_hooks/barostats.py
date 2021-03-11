"""
This module contains various thermostats for regulating the temperature of the system during
molecular dynamics simulations. Apart from standard thermostats for convetional simulations,
a series of special thermostats developed for ring polymer molecular dynamics is also provided.
"""
import torch

from schnetpack.md.utils import MDUnits, YSWeights, StableSinhDiv
from schnetpack.md.simulation_hooks import SimulationHook

__all__ = [
    "BarostatHook",
    "NHCBarostatIsotropic",
    "NHCBarostatAnisotropic",
    "RPMDBarostat",
]


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
        temperature_bath (float): Target temperature applied to the cell fluctuations.
        detach (bool): Whether the computational graph should be detached after each simulation step. Default is true,
                       should be changed if differentiable MD is desired.
    """

    def __init__(self, target_pressure, temperature_bath, detach=True):
        self.target_pressure = target_pressure * MDUnits.bar2internal
        self.temperature_bath = temperature_bath
        self.initialized = False
        self.device = None
        self.n_replicas = None
        self.n_molecules = None
        self.n_atoms = None
        self.detach = detach
        self.time_step = None

    def on_simulation_start(self, simulator):
        """
        Routine to initialize the barostat based on the current state of the simulator. Reads the device to be uses,
        as well as the number of molecular replicas present in simulator.system. A flag is set so that the barostat
        is not reinitialized upon continuation of the MD.

        Main function is the _init_barostat routine, which takes the simulator as input and must be provided for every
        new barostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                                the time step, system, etc.
        """
        self.device = simulator.system.device
        self.n_replicas = simulator.system.n_replicas
        self.n_molecules = simulator.system.n_molecules
        self.n_atoms = simulator.system.max_n_atoms
        self.time_step = simulator.integrator.time_step

        if not self.initialized:
            self._init_barostat(simulator)
            self.initialized = True

    def on_step_begin(self, simulator):
        """
        First application of the barostat before the first half step of the dynamics. Applies a mask to the system
        momenta in order to avoid problems of e.g. noise added to the zero padded tensors. The detach is carried out
        here.

        Main function is the _apply_barostat routine, which takes the simulator as input and must be provided for
        every new barostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
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
        Second application of the barostat after the second half step of the dynamics. Applies a mask to the system
        momenta in order to avoid problems of e.g. noise added to the zero padded tensors. The detach is carried out
        here.

        Main function is the _apply_barostat routine, which takes the simulator as input and must be provided for
        every new barostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
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
        Dummy routine for initializing a barpstat based on the current simulator. Should be implemented for every
        new barostat. Has access to the information contained in the simulator class, e.g. number of replicas, time
        step, masses of the atoms, etc.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
        """
        raise NotImplementedError

    def _apply_barostat(self, simulator):
        """
        Dummy routine for applying the barostat to the system and propagating the barostat. Is called twice each
        simulation time step.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                the time step, system, etc.
        """
        raise NotImplementedError

    def propagate_system(self, system):
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

    def propagate_barostat_half_step(self, system):
        """
        Routine only required for RPMD which propagates the barostat attached to the centroid cell. Is called during
        each half-step before system momenta are propagated.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        pass


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
        target_pressure,
        temperature_bath,
        time_constant,
        time_constant_cell=None,
        time_constant_barostat=None,
        chain_length=4,
        multi_step=4,
        integration_order=7,
        massive=False,
        detach=True,
    ):
        super(NHCBarostatIsotropic, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            detach=detach,
        )
        self.chain_length = chain_length
        self.frequency = 1 / (time_constant * MDUnits.fs2internal)

        if time_constant_cell is None:
            self.cell_frequency = self.frequency
        else:
            self.cell_frequency = 1 / (time_constant_cell * MDUnits.fs2internal)

        if time_constant_barostat is None:
            self.barostat_frequency = self.frequency
        else:
            self.barostat_frequency = 1 / (time_constant_barostat * MDUnits.fs2internal)

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
        self.degrees_of_freedom_particles = None

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

        # Flag for massive theromstating
        self.massive = massive

    def _init_barostat(self, simulator):
        """
        Initialize the thermostat positions, forces, velocities and masses, as well as the number of degrees of freedom
        seen by each chain link. In the same manner, all quantities required for the barostat are initialized.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                    the time step, system, etc.
        """
        # Determine integration step via multi step and Yoshida Suzuki weights
        integration_weights = YSWeights(self.device).get_weights(self.integration_order)
        self.ys_time_step = (
            simulator.integrator.time_step * integration_weights / self.multi_step
        )

        # Determine internal degrees of freedom
        self.degrees_of_freedom = 3 * simulator.system.n_atoms.float()[None, :]

        # Determine degrees of freedom for particle thermostats (depends on massive)
        n_replicas, n_molecules, n_atoms, xyz = simulator.system.momenta.shape
        if self.massive:
            # Since momenta will be masked later, no need to set non-atoms to 0
            self.degrees_of_freedom_particles = torch.ones(
                (n_replicas, n_molecules, n_atoms, xyz), device=self.device
            )
        else:
            self.degrees_of_freedom_particles = self.degrees_of_freedom[None, None]

        # Set up internal variables
        self._init_thermostat_variables()
        self._init_barostat_variables()

    def _init_thermostat_variables(self):
        """
        Initialize all quantities required for the two thermostat chains on the particles and cells.
        """
        # Set up thermostat masses
        if self.massive:
            self.t_masses = torch.zeros(
                self.n_replicas,
                self.n_molecules,
                self.n_atoms,
                3,
                self.chain_length,
                device=self.device,
            )
        else:
            self.t_masses = torch.zeros(
                self.n_replicas,
                self.n_molecules,
                1,
                1,
                self.chain_length,
                device=self.device,
            )

        self.t_masses_cell = torch.zeros(
            self.n_replicas, self.n_molecules, self.chain_length, device=self.device
        )

        # Get masses of innermost thermostat
        self.t_masses[..., 0] = (
            self.degrees_of_freedom_particles
            * self.kb_temperature
            / self.frequency ** 2
        )
        self.t_masses_cell[..., 0] = (
            9.0 * self.kb_temperature / self.cell_frequency ** 2
        )
        # Set masses of remaining thermostats
        self.t_masses[..., 1:] = self.kb_temperature / self.frequency ** 2
        self.t_masses_cell[..., 1:] = self.kb_temperature / self.cell_frequency ** 2

        # Thermostat variables for particles
        self.t_velocities = torch.zeros_like(self.t_masses, device=self.device)
        self.t_forces = torch.zeros_like(self.t_masses, device=self.device)

        # Thermostat variables for cell
        self.t_velocities_cell = torch.zeros_like(
            self.t_masses_cell, device=self.device
        )
        self.t_forces_cell = torch.zeros_like(self.t_masses_cell, device=self.device)

        # Positions for conservation
        # self.t_positions = torch.zeros_like(self.t_masses, device=self.device)
        # self.t_positions_cell = torch.zeros_like(self.t_masses_cell, device=self.device)

    def _init_barostat_variables(self):
        """
        Initialize all quantities required for the barostat component.
        """
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
        self.b_velocities_cell = torch.zeros_like(
            self.b_masses_cell, device=self.device
        )
        self.b_forces_cell = torch.zeros_like(self.b_masses_cell, device=self.device)

    def _apply_barostat(self, simulator):
        """
        Apply the thermostat chains to the system and cell momenta. This is done by computing forces of the innermost
        thermostats, propagating the chain forward, updating the box velocities, particle momenta and associated
        energies. Based on this, the kinetic energies and forces can be updated, which are the propagated backward
        through the chain.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                                the time step, system, etc.
        """
        self._init_kinetic_energy(simulator.system)
        (
            kinetic_energy_for_thermostat,
            kinetic_energy_for_barostat,
        ) = self._compute_kinetic_energy(simulator.system)
        # kinetic_energy = self._compute_kinetic_energy(simulator.system)
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

        if self.detach:
            # self.t_positions = self.t_positions.detach()
            # self.t_positions_cell = self.t_positions_cell.detach()
            self.t_velocities = self.t_velocities.detach()
            self.t_velocities_cell = self.t_velocities_cell.detach()
            self.b_velocities_cell = self.b_velocities_cell.detach()

    def _init_kinetic_energy(self, system):
        """
        Auxiliary routine for initializing the kinetic energy and accumulating the scaling factor.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        self.scaling = 1.0
        # R x M x A x 3
        self.kinetic_energy = system.momenta ** 2 / system.masses * system.atom_masks

    def _compute_kinetic_energy(self, system):
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

        kinetic_energy_for_barostat = torch.sum(self.kinetic_energy, dim=[2, 3])

        if self.massive:
            kinetic_energy_for_thermostat = self.kinetic_energy
        else:
            kinetic_energy_for_thermostat = kinetic_energy_for_barostat[
                :, :, None, None
            ]

        return kinetic_energy_for_thermostat, kinetic_energy_for_barostat

    def _compute_kinetic_energy_cell(self):
        """
        Compute the kinetic energy of the cells.

        Returns:
            torch.tensor: Kinetic energy associated with the cells.
        """
        return self.b_masses_cell * self.b_velocities_cell ** 2

    def _compute_pressure(self, system):
        """
        Routine for computing the current pressure and volume associated with the simulated systems. The stress tensor
        is used for pressure computation.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.

        Returns:
            (torch.Tensor, torch.Tensor): Duple containing the pressures and volumes.
        """
        # Get the pressure (R x M)
        pressure = system.compute_pressure(kinetic_component=False, tensor=False)
        # Get the volume (R x M)
        volume = system.volume
        return pressure, volume

    def _chain_forward(self, time_step):
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

    def _update_box_velocities(self, time_step):
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

    def _update_particle_momenta(self, time_step, system):
        """
        Update the momenta of the particles, as well as the internal kinetic energy surrogate.

        Args:
            time_step (float): Current timestep considering YS and multi-timestep integration.
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        scaling = torch.exp(
            -0.5
            * time_step
            * (
                self.t_velocities[..., 0]
                + self.b_velocities_cell * (1 + 3 / self.degrees_of_freedom)
            )
        )
        system.momenta *= scaling * system.atom_masks
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

    def _update_forces_thermostat(self, kinetic_energy):
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

    def _update_forces_barostat(self, kinetic_energy, pressure, volume):
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

    def propagate_system(self, system):
        """
        Main routine for propagating the system positions and cells. Since this is modifed, no conventional velocity
        verlet integrator can be used.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
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

    @property
    def state_dict(self):
        state_dict = {
            "chain_length": self.chain_length,
            "frequency": self.frequency,
            "cell_frequency": self.cell_frequency,
            "barostat_frequency": self.barostat_frequency,
            "kb_temperature": self.kb_temperature,
            "degrees_of_freedom": self.degrees_of_freedom,
            "t_masses": self.t_masses,
            "t_masses_cell": self.t_masses_cell,
            "b_masses_cell": self.b_masses_cell,
            "t_velocities": self.t_velocities,
            "t_velocities_cell": self.t_velocities_cell,
            "b_velocities_cell": self.b_velocities_cell,
            "t_forces": self.t_forces,
            "t_forces_cell": self.t_forces_cell,
            "b_forces_cell": self.b_forces_cell,
            "time_step": self.ys_time_step,
            "temperature_bath": self.temperature_bath,
            "target_pressure": self.target_pressure,
            "n_replicas": self.n_replicas,
            "multi_step": self.multi_step,
            "integration_order": self.integration_order,
            "massive": self.massive,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.chain_length = state_dict["chain_length"]
        self.frequency = state_dict["frequency"]
        self.cell_frequency = state_dict["cell_frequency"]
        self.barostat_frequency = state_dict["barostat_frequency"]
        self.kb_temperature = state_dict["kb_temperature"]
        self.degrees_of_freedom = state_dict["degrees_of_freedom"]
        self.t_masses = state_dict["t_masses"]
        self.t_masses_cell = state_dict["t_masses_cell"]
        self.b_masses_cell = state_dict["b_masses_cell"]
        self.t_velocities = state_dict["t_velocities"]
        self.t_velocities_cell = state_dict["t_velocities_cell"]
        self.b_velocities_cell = state_dict["b_velocities_cell"]
        self.t_forces = state_dict["t_forces"]
        self.t_forces_cell = state_dict["t_forces_cell"]
        self.b_forces_cell = state_dict["b_forces_cell"]
        self.ys_time_step = state_dict["time_step"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.target_pressure = state_dict["target_pressure"]
        self.n_replicas = state_dict["n_replicas"]
        self.multi_step = state_dict["multi_step"]
        self.integration_order = state_dict["integration_order"]
        self.massive = state_dict["massive"]

        self.initialized = True


class NHCBarostatAnisotropic(NHCBarostatIsotropic):
    """
    Parrinello--Rahman thermostat/barostat based on Nose-Hoover chains for fully anisotropic cell fluctuations. For a
    in-depth description of the algorithm, see[#nhc_barostat2]_. This barostat already contains a built in thermostat,
    so no further temperature control is necessary. As suggested in [#nhc_barostat2]_, two separate chains are used to
    thermostat particle and cell momenta. In order to propagate the system, operators in the form of matrix exponentials
    need to be applied, which requires several eigendecompositions of 3x3 matrices, making the barostat computationally
    intensive. In addition, accurate predictors of the stress tensor are required, since deformations of the cells are
    now possible.

    Args:
        target_pressure (float): Target pressure in bar.
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs
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
    .. [#nhc_barostat2] Martyna, Tuckerman, Tobias, Klein:
       Explicit reversible integrators for extended systems dynamics.
       Molecular Physics, 87(5), 1117-1157. 1996.
    """

    def __init__(
        self,
        target_pressure,
        temperature_bath,
        time_constant,
        time_constant_cell=None,
        time_constant_barostat=None,
        chain_length=4,
        multi_step=4,
        integration_order=7,
        massive=False,
        detach=True,
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
            detach=detach,
        )

    def _init_barostat_variables(self):
        """
        Initialize all quantities required for the barostat component.
        """
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
        self.b_velocities_cell = torch.zeros(
            self.n_replicas, self.n_molecules, 3, 3, device=self.device
        )
        self.b_forces_cell = torch.zeros_like(
            self.b_velocities_cell, device=self.device
        )

        # Auxiliary identity matrix for broadcasting
        self.aux_eye = torch.eye(3, device=self.device)[None, None, :, :]

    def _init_kinetic_energy(self, system):
        """
        This routine is no longer required, since it is no longer possible to accumulate the barostat
        action.

        Args:
           system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        pass

    def _compute_kinetic_energy(self, system):
        """
        Compute the current kinetic energy tensor.
        Since barostat and thermostat updates require different kinetic energy conventions in case of massive
        updates, two separate tensors are computed.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        Returns:
            torch.tensor: Current kinetic energy tensor of the particles.
        """
        # Here we need the full tensor (R x M x 3 x 3)
        # Kinetic energy can be computed as Tr[Etens]
        kinetic_energy_tensor = 2.0 * system.kinetic_energy_tensor

        if self.massive:
            kinetic_energy_for_thermostat = system.momenta ** 2 / system.masses
        else:
            kinetic_energy_for_thermostat = torch.einsum(
                "abii->ab", kinetic_energy_tensor
            )[:, :, None, None]

        return kinetic_energy_for_thermostat, kinetic_energy_tensor

    def _compute_kinetic_energy_cell(self):
        """
        Compute the kinetic energy of the cells.

        Returns:
            torch.tensor: Kinetic energy associated with the cells.
        """
        b_cell_sq = torch.matmul(
            self.b_velocities_cell.transpose(2, 3), self.b_velocities_cell
        )
        # Einsum computes the trace
        return (
            self.b_masses_cell * torch.einsum("abii->ab", b_cell_sq)[:, :, None, None]
        )

    def _compute_pressure(self, system):
        """
        Routine for computing the current pressure tensor and volume associated with the simulated systems.
        The pressure tensor is equivalent to the negative stress tensor.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.

        Returns:
            (torch.Tensor, torch.Tensor): Duple containing the pressure tensors and volumes.
        """
        # Get the pressure (R x M x 3 x 3)
        pressure = system.compute_pressure(kinetic_component=False, tensor=True)
        # Get the volume (R x M x 1 x 1)
        volume = system.volume[..., None]
        return pressure, volume

    def _update_particle_momenta(self, time_step, system):
        """
        Update the momenta of the particles. In contrast to the isotropic case, an eigenvalue problem needs to be
        solved.

        Args:
            time_step (float): Current timestep considering YS and multi-timestep integration.
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Compute auxiliary velocity tensor for propagation
        # vtemp = (
        #         self.b_velocities_cell
        #         + (
        #                 torch.einsum("abii->ab", self.b_velocities_cell)[:, :, None, None]
        #                 / self.degrees_of_freedom
        #                 + self.t_velocities[..., 0]
        #         ) * self.aux_eye
        # )

        # Compute eigenvectors and values for matrix exponential operator
        # eigval -> (R x M x 3)
        # eigvec -> (R x M x 3 x 3)
        # eigval, eigvec = torch.symeig(vtemp, eigenvectors=True)
        # operator = torch.exp(-0.5 * eigval * self.time_step)[:, :, None, :]

        # Since the original matrix consist of matrix + diagonal term, the operator can be split
        # in order to apply massive particle thermostats
        # TODO: check sinh in momenta updates as in Tuckerman book
        vtemp = (
            self.b_velocities_cell
            + (
                torch.einsum("abii->ab", self.b_velocities_cell)
                / self.degrees_of_freedom
            )[:, :, None, None]
            * self.aux_eye
        )
        eigval, eigvec = torch.symeig(vtemp, eigenvectors=True)
        operator1 = torch.exp(-0.5 * eigval * time_step)[:, :, None, :]
        operator2 = torch.exp(-0.5 * self.t_velocities[..., 0] * time_step)

        # The following procedure computes the matrix exponential of vtemp and applies it to
        # the momenta.
        # p' = p * c
        momenta_tmp = torch.matmul(system.momenta / system.masses, eigvec)
        # Multiply by operator
        momenta_tmp = momenta_tmp * operator1
        # Transform back
        # p = p' * c.T
        system.momenta = (
            torch.matmul(momenta_tmp, eigvec.transpose(2, 3))
            * operator2
            * system.masses
            * system.atom_masks
        )

    def _update_forces_thermostat(self, kinetic_energy):
        """
        Update the forces acting on the two innermost thermostats coupled to the particle and cell momenta.
        The standard kinetic energy is computed as the trace of the kinetic energy tensor.

        Args:
            kinetic_energy (torch.Tensor): Tensor containing the current kinetic energies of the systems.
        """
        # Compute Ekin from tensor
        # kinetic_energy = torch.einsum("abii->ab", kinetic_energy)
        # Compute forces on thermostat (R x M)
        self.t_forces[..., 0] = (
            kinetic_energy - self.degrees_of_freedom_particles * self.kb_temperature
        ) / self.t_masses[..., 0]

        # Get kinetic energy of barostat (R x M)
        kinetic_energy_cell = self._compute_kinetic_energy_cell()
        # Compute forces on cell thermostat
        self.t_forces_cell[..., 0] = (
            kinetic_energy_cell - 9.0 * self.kb_temperature
        ) / self.t_masses_cell[..., 0]

    def _update_forces_barostat(self, kinetic_energy, pressure, volume):
        """
        Update the forces acting on the barostat coupled to the cell.
        The standard kinetic energy is computed as the trace of the kinetic energy tensor.

        Args:
            kinetic_energy (torch.Tensor): Tensor containing the current kinetic energies of the systems.
            pressure (torch.Tensor): Current pressure of each system.
            volume (torch.Tensor): Current volume of each system.
        """
        kinetic_energy_scalar = torch.einsum("abii->ab", kinetic_energy)[
            :, :, None, None
        ]
        self.b_forces_cell = (
            kinetic_energy_scalar
            / self.degrees_of_freedom[:, :, None, None]
            * self.aux_eye
            + kinetic_energy
            + volume * (pressure - self.aux_eye * self.target_pressure)
        ) / self.b_masses_cell

    def propagate_system(self, system):
        """
        Main routine for propagating the system positions and cells. Compared to the standard velocity verlet, this
        routine is heavily modified and makes use of eigendecomposition.
        verlet integrator can be used.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Compute eigenvectors and values for matrix exponential operator
        # eigval -> (R x M x 3)
        # eigvec -> (R x M x 3 x 3)
        eigval, eigvec = torch.symeig(self.b_velocities_cell, eigenvectors=True)

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

    # def compute_conserved(self, system):
    #    """
    #    Computed the conserved quantity. For debug purposes only.
    #    """
    #    conserved = (
    #            system.kinetic_energy[..., None, None]
    #            + system.energies[..., None, None]
    #            + 0.5 * torch.sum(self.t_velocities ** 2 * self.t_masses, 2)
    #            + 0.5 * torch.sum(self.t_velocities_cell ** 2 * self.t_masses_cell, 2)
    #            + 0.5 * self._compute_kinetic_energy_cell()
    #            + self.degrees_of_freedom * self.kb_temperature * self.t_positions[..., 0]
    #            + 9.0 * self.kb_temperature * self.t_positions_cell[..., 0]
    #            + self.kb_temperature * torch.sum(self.t_positions[..., 1:], 2)
    #            + self.kb_temperature * torch.sum(self.t_positions_cell[..., 1:], 2)
    #            + self.target_pressure * system.volume
    #    )
    #    return conserved

    @property
    def state_dict(self):
        state_dict = {
            "chain_length": self.chain_length,
            "frequency": self.frequency,
            "cell_frequency": self.cell_frequency,
            "barostat_frequency": self.barostat_frequency,
            "kb_temperature": self.kb_temperature,
            "degrees_of_freedom": self.degrees_of_freedom,
            "t_masses": self.t_masses,
            "t_masses_cell": self.t_masses_cell,
            "b_masses_cell": self.b_masses_cell,
            "t_velocities": self.t_velocities,
            "t_velocities_cell": self.t_velocities_cell,
            "b_velocities_cell": self.b_velocities_cell,
            "t_forces": self.t_forces,
            "t_forces_cell": self.t_forces_cell,
            "b_forces_cell": self.b_forces_cell,
            "time_step": self.ys_time_step,
            "temperature_bath": self.temperature_bath,
            "target_pressure": self.target_pressure,
            "n_replicas": self.n_replicas,
            "multi_step": self.multi_step,
            "integration_order": self.integration_order,
            "aux_eye": self.aux_eye,
            "massive": self.massive,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.chain_length = state_dict["chain_length"]
        self.frequency = state_dict["frequency"]
        self.cell_frequency = state_dict["cell_frequency"]
        self.barostat_frequency = state_dict["barostat_frequency"]
        self.kb_temperature = state_dict["kb_temperature"]
        self.degrees_of_freedom = state_dict["degrees_of_freedom"]
        self.t_masses = state_dict["t_masses"]
        self.t_masses_cell = state_dict["t_masses_cell"]
        self.b_masses_cell = state_dict["b_masses_cell"]
        self.t_velocities = state_dict["t_velocities"]
        self.t_velocities_cell = state_dict["t_velocities_cell"]
        self.b_velocities_cell = state_dict["b_velocities_cell"]
        self.t_forces = state_dict["t_forces"]
        self.t_forces_cell = state_dict["t_forces_cell"]
        self.b_forces_cell = state_dict["b_forces_cell"]
        self.ys_time_step = state_dict["time_step"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.target_pressure = state_dict["target_pressure"]
        self.n_replicas = state_dict["n_replicas"]
        self.multi_step = state_dict["multi_step"]
        self.integration_order = state_dict["integration_order"]
        self.aux_eye = state_dict["aux_eye"]
        self.massive = state_dict["massive"]

        self.initialized = True


class RPMDBarostat(BarostatHook):
    """
    Barostat for ring polymer molecular dynamics simulations. This barostat is based on the algorithm introduced in
    [#rpmd_barostat1]_ and uses a PILE thermostat for the cell. It can be combined with any RPMD thermostat for
    temperature control. The barostat only acts on the centroid mode of the ring polymer and is designed for isotropic
    cell fluctuations.

    Args:
        target_pressure (float): Target pressure of the system (in bar).
        temperature_bath (float): Target temperature applied to the cell fluctuations.
        detach (bool): Whether the computational graph should be detached after each simulation step. Default is true,
                       should be changed if differentiable MD is desired.

    References
    ----------
    .. [#rpmd_barostat1] Kapil, et al.
                         i-PI 2.0: A universal force engine for advanced molecular simulations.
                         Computer Physics Communications, 236, 214-223. 2019.
    """

    def __init__(self, target_pressure, temperature_bath, time_constant, detach=True):
        super(RPMDBarostat, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            detach=detach,
        )
        self.frequency = 1.0 / (time_constant * MDUnits.fs2internal)
        self.kb_temperature = temperature_bath * MDUnits.kB
        self.transformation = None
        self.propagator = None
        self.cell_momenta = None
        self.sinhdx = StableSinhDiv()

    def _init_barostat(self, simulator):
        """
        Initialize the thermostat coefficients and barostat quantities.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                    the time step, system, etc.
        """
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
        self.c1 = torch.exp(
            -0.5 * torch.ones(1, device=self.device) * self.frequency * self.time_step
        )
        self.c2 = torch.sqrt(
            self.n_replicas * self.mass * self.kb_temperature * (1.0 - self.c1 ** 2)
        )

    def _apply_barostat(self, simulator):
        """
        Apply the thermostat. This simply propagates the cell momenta under the influence of a PILE thermostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                                the time step, system, etc.
        """
        # Propagate cell momenta during half-step
        self.cell_momenta = self.c1 * self.cell_momenta + self.c2 * torch.randn_like(
            self.cell_momenta
        )
        if self.detach:
            self.cell_momenta = self.cell_momenta.detach()

    def propagate_system(self, system):
        """
        Main routine for propagating the ring polymer and the cells. The barostat acts only on the centroid, while the
        remaining replicas are propagated in the conventional way.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Transform to normal mode representation
        positions_normal = self.transformation.beads2normal(system.positions)
        momenta_normal = self.transformation.beads2normal(system.momenta)

        # Propagate centroid mode of the ring polymer
        reduced_momenta = (self.cell_momenta / self.mass)[:, None, None]
        scaling = torch.exp(-self.time_step * reduced_momenta)

        momenta_normal[0] = momenta_normal[0] * scaling
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
            + self.propagator[1:, 0, 1] * positions_normal[1:] * system.masses
        )
        positions_normal[1:] = (
            self.propagator[1:, 1, 0] * momenta_normal[1:] / system.masses
            + self.propagator[1:, 1, 1] * positions_normal[1:]
        )

        # Transform back to bead representation
        system.positions = self.transformation.normal2beads(positions_normal)
        system.momenta = self.transformation.normal2beads(momenta_normal)

    def propagate_barostat_half_step(self, system):
        """
        Propagate the momenta of the thermostat attached to the barostat during each half-step.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        centroid_momenta = self.transformation.beads2normal(system.momenta)[0]
        centroid_forces = self.transformation.beads2normal(system.forces)[0]

        # Compute pressure component (volume[0] can be used, since the volume is scaled equally for all replicas)
        component_1 = (
            3.0
            * self.n_replicas
            * (
                system.volume[0]
                * (
                    torch.mean(system.compute_pressure(kinetic_component=True), dim=0)
                    - self.target_pressure
                )
                + self.kb_temperature
            )
        )

        # Compute components based on forces and momenta
        force_by_mass = centroid_forces / system.masses[0]

        component_2 = torch.sum(force_by_mass * centroid_momenta, dim=[1, 2])
        component_3 = torch.sum(force_by_mass * centroid_forces / 3, dim=[1, 2])

        # Update cell momenta
        self.cell_momenta = (
            self.cell_momenta
            + (0.5 * self.time_step) * component_1
            + (0.5 * self.time_step) ** 2 * component_2
            + (0.5 * self.time_step) ** 3 * component_3
        )

        if self.detach:
            self.cell_momenta = self.cell_momenta.detach()

    @property
    def state_dict(self):
        state_dict = {
            "frequency": self.frequency,
            "kb_temperature": self.kb_temperature,
            "transformation": self.transformation,
            "propagator": self.propagator,
            "cell_momenta": self.cell_momenta,
            "mass": self.mass,
            "c1": self.c1,
            "c2": self.c2,
            "temperature_bath": self.temperature_bath,
            "target_pressure": self.target_pressure,
            "n_replicas": self.n_replicas,
        }
        return state_dict

    @state_dict.setter
    def state_dict(self, state_dict):
        self.frequency = state_dict["frequency"]
        self.kb_temperature = state_dict["kb_temperature"]
        self.transformation = state_dict["transformation"]
        self.propagator = state_dict["propagator"]
        self.cell_momenta = state_dict["cell_momenta"]
        self.mass = state_dict["mass"]
        self.c1 = state_dict["c1"]
        self.c2 = state_dict["c2"]
        self.temperature_bath = state_dict["temperature_bath"]
        self.target_pressure = state_dict["target_pressure"]
        self.n_replicas = state_dict["n_replicas"]

        self.initialized = True

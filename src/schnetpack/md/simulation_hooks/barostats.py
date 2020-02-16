"""
This module contains various thermostats for regulating the temperature of the system during
molecular dynamics simulations. Apart from standard thermostats for convetional simulations,
a series of special thermostats developed for ring polymer molecular dynamics is also provided.
"""
import torch
import numpy as np
import scipy.linalg as linalg
import logging

from schnetpack.md.utils import MDUnits, YSWeights
from schnetpack.md.simulation_hooks import ThermostatHook, SimulationHook

__all__ = ["BarostatHook"]


class BarostatHook(SimulationHook):
    """
    """

    # TODO: Could be made a torch nn.Module

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
        n_atoms = simulator.system.n_atoms[None, :, None, None]
        self.inv_sqrt_dof = 1.0 / torch.sqrt(3 * n_atoms + 1.0)
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
        # Get pressure difference
        pressure_difference = (
            3.0
            * system.volume
            * (system.compute_pressure(kinetic_component=True) - self.target_pressure)
        )

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
            + torch.sinh(self.time_step * reduced_momenta)
            / reduced_momenta
            * momenta_normal[0]
            / system.masses[0]
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


class NHCBarostatIsotropic(ThermostatHook):
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
        temperature_bath,
        time_constant,
        chain_length=3,
        massive=False,
        multi_step=2,
        integration_order=3,
    ):
        super(NHCBarostatIsotropic, self).__init__(
            temperature_bath, nm_transformation=None
        )
        # TODO: Get rid of massive
        # TODO: separate chains for cell and particles

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

    def _propagate_positions(self):
        raise NotImplementedError

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

        kinetic_energy = self._compute_kinetic_energy(momenta, simulator.system.masses)

        scaling_factor = self._propagate_thermostat(kinetic_energy)
        momenta = momenta * scaling_factor

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

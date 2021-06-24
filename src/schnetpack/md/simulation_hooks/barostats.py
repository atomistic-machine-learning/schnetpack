"""
This module contains various thermostats for regulating the temperature of the system during
molecular dynamics simulations. Apart from standard thermostats for convetional simulations,
a series of special thermostats developed for ring polymer molecular dynamics is also provided.
"""
import torch

from schnetpack import units as spk_units
from ase import units as ase_units
from schnetpack.md.simulation_hooks import SimulationHook

__all__ = ["BarostatHook"]


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
        temperature_bath (float): Target temperature applied to the cell fluctuations.
    """

    def __init__(self, target_pressure: float, temperature_bath: float):
        super(BarostatHook, self).__init__()
        # Convert pressure from bar to internal units
        self.target_pressure = target_pressure * spk_units.convert_units(
            1e5 * ase_units.Pascal, spk_units.pressure
        )
        self.temperature_bath = temperature_bath

        self.initialized = False

        self.n_replicas = None
        self.n_molecules = None
        self.n_atoms = None
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

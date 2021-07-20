"""
This module contains various thermostats for regulating the temperature of the system during
molecular dynamics simulations. Apart from standard thermostats for convetional simulations,
a series of special thermostats developed for ring polymer molecular dynamics is also provided.
"""
import torch

from schnetpack import units as spk_units
from schnetpack.md.simulation_hooks.basic_hooks import SimulationHook
from schnetpack.md.simulator import Simulator

__all__ = ["ThermostatHook", "BerendsenThermostat"]


class ThermostatError(Exception):
    """
    Exception for thermostat class.
    """

    pass


class ThermostatHook(SimulationHook):
    ring_polymer = False
    """
    Basic thermostat hook for simulator class. This class is initialized based on the simulator and system
    specifications during the first MD step. Thermostats are applied before and after each MD step.

    Args:
        temperature_bath (float): Temperature of the heat bath in Kelvin.
    """

    def __init__(self, temperature_bath: float):
        super(ThermostatHook, self).__init__()
        self.register_buffer("temperature_bath", torch.tensor([temperature_bath]))
        self.initialized = False

    def on_simulation_start(self, simulator: Simulator):
        """
        Routine to initialize the thermostat based on the current state of the simulator. Reads the device to be used.
        In addition, a flag is set so that the thermostat is not reinitialized upon continuation of the MD.

        Main function is the `_init_thermostat` routine, which takes the simulator as input and must be provided for every
        new thermostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                                         the time step, system, etc.
        """
        if not self.initialized:
            self._init_thermostat(simulator)
            self.initialized = True

    def on_step_begin(self, simulator: Simulator):
        """
        First application of the thermostat before the first half step of the dynamics. Regulates temperature.

        Main function is the `_apply_thermostat` routine, which takes the simulator as input and must be provided for
        every new thermostat.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Apply thermostat
        self._apply_thermostat(simulator)

    def on_step_end(self, simulator: Simulator):
        """
        Application of the thermostat after the second half step of the dynamics. Regulates temperature.

        Main function is the `_apply_thermostat` routine, which takes the simulator as input and must be provided for
        every new thermostat.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Apply thermostat
        self._apply_thermostat(simulator)

    def _init_thermostat(self, simulator: Simulator):
        """
        Dummy routine for initializing a thermostat based on the current simulator. Should be implemented for every
        new thermostat. Has access to the information contained in the simulator class, e.g. number of replicas, time
        step, masses of the atoms, etc.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        pass

    def _apply_thermostat(self, simulator: Simulator):
        """
        Dummy routine for applying the thermostat to the system. Should use the implemented thermostat to update the
        momenta of the system contained in `simulator.system.momenta`. Is called twice each simulation time step.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        raise NotImplementedError


class BerendsenThermostat(ThermostatHook):
    ring_polymer = False
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

    def __init__(self, temperature_bath: float, time_constant: float):
        super(BerendsenThermostat, self).__init__(temperature_bath)
        # Convert from fs to internal time units
        self.register_buffer(
            "time_constant", torch.tensor([time_constant * spk_units.fs])
        )

    def _apply_thermostat(self, simulator):
        """
        Apply the Berendsen thermostat via rescaling the systems momenta based on the current instantaneous temperature
        and the bath temperature.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        scaling = torch.sqrt(
            1.0
            + simulator.integrator.time_step
            / self.time_constant
            * (self.temperature_bath / simulator.system.temperature - 1)
        )
        simulator.system.momenta = (
            simulator.system.expand_atoms(scaling) * simulator.system.momenta
        )


# TODO: decide on multiple thermostat temperatures after implementing RPMD thermos

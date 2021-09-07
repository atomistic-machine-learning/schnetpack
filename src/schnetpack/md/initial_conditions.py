"""
Module for setting up the initial conditions of the molecules in :obj:`schnetpack.md.System`.
This entails sampling the momenta from random distributions corresponding to certain temperatures.
"""
import torch
from schnetpack.md import System
from schnetpack import units as spk_units
from typing import Union, List

__all__ = ["Initializer", "MaxwellBoltzmannInit", "UniformInit"]


class InitializerError(Exception):
    pass


class Initializer:
    """
    Basic initializer class template. Initializes the systems momenta to correspond to a certain temperature.

    Args:
        temperature (float):  Target initialization temperature in Kelvin.
        remove_translation (bool): Remove the translational components of the momenta after initialization. Will stop
                                   molecular drift for NVE simulations and NVT simulations with deterministic
                                   thermostat (default=False).
        remove_rotation (bool): Remove the rotational components of the momenta after initialization. Will reduce
                                molecular rotation for NVE simulations and NVT simulations with deterministic
                                thermostat (default=False).
        wrap_positions (bool): Wrap atom positions back to box when using periodic boundary conditions.
    """

    def __init__(
        self,
        temperature: Union[float, List[float]],
        remove_center_of_mass: bool = True,
        remove_translation: bool = True,
        remove_rotation: bool = False,
        wrap_positions: bool = False,
    ):
        if not isinstance(temperature, list):
            temperature = [temperature]

        self.temperature: torch.Tensor = torch.tensor(temperature)
        self.remove_com = remove_center_of_mass
        self.remove_translation = remove_translation
        self.remove_rotation = remove_rotation
        self.wrap_positions = wrap_positions

    def initialize_system(self, system: System):
        """
        Initialize the system according to the instructions given in _setup_momenta.

        Args:
            system (object): System class containing all molecules and their replicas.
        """
        if self.temperature.shape[0] != 1:
            if self.temperature.shape[0] != system.n_molecules:
                raise InitializerError(
                    "Initializer requires either a single temperature or one per molecule."
                )

        if self.remove_com:
            system.remove_center_of_mass()

        if self.wrap_positions:
            system.wrap_positions()

        self._setup_momenta(system)

        if self.remove_translation:
            system.remove_translation()

        if self.remove_rotation:
            system.remove_com_rotation()

    def _setup_momenta(self, system: System):
        """
        Main routine for initializing system momenta based on the molecules defined in system and the provided
        temperature. To be implemented.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their replicas.
        """
        raise NotImplementedError


class UniformInit(Initializer):
    """
    Initializes the system momenta according to a uniform distribution scaled to the given temperature.

    Args:
        temperature (float): Target temperature in Kelvin.
        remove_translation (bool): Remove the translational components of the momenta after initialization. Will stop
                                   molecular drift for NVE simulations and NVT simulations with deterministic
                                   thermostat (default=False).
        remove_rotation (bool): Remove the rotational components of the momenta after initialization. Will reduce
                                molecular rotation for NVE simulations and NVT simulations with deterministic
                                thermostat (default=False).
        wrap_positions (bool): Wrap atom positions back to box when using periodic boundary conditions.
    """

    def __init__(
        self,
        temperature: Union[float, List[float]],
        remove_center_of_mass: bool = True,
        remove_translation: bool = True,
        remove_rotation: bool = False,
        wrap_positions: bool = False,
    ):
        super(UniformInit, self).__init__(
            temperature,
            remove_center_of_mass=remove_center_of_mass,
            remove_translation=remove_translation,
            remove_rotation=remove_rotation,
            wrap_positions=wrap_positions,
        )

    def _setup_momenta(self, system: System):
        """
        Initialize the momenta, by drawing from a random normal distribution and rescaling the momenta to the desired
        temperature afterwards. In addition, the system is centered at its center of mass.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their replicas.
        """
        # Set initial system momenta and apply atom masks
        system.momenta = torch.randn_like(system.momenta) * system.masses

        # Scale velocities to desired temperature
        scaling = torch.sqrt(
            self.temperature.to(system.momenta.device)[None, :, None]
            / system.temperature
        )
        system.momenta *= system.expand_atoms(scaling)


class MaxwellBoltzmannInit(Initializer):
    """
    Initializes the system momenta according to a Maxwell--Boltzmann distribution at the given temperature.

    Args:
        temperature (float): Target temperature in Kelvin.
        remove_translation (bool): Remove the translational components of the momenta after initialization. Will stop
                                   molecular drift for NVE simulations and NVT simulations with deterministic
                                   thermostat (default=False).
        remove_rotation (bool): Remove the rotational components of the momenta after initialization. Will reduce
                                molecular rotation for NVE simulations and NVT simulations with deterministic
                                thermostat (default=False).
        wrap_positions (bool): Wrap atom positions back to box when using periodic boundary conditions.
    """

    def __init__(
        self,
        temperature: Union[float, List[float]],
        remove_center_of_mass: bool = True,
        remove_translation: bool = True,
        remove_rotation: bool = False,
        wrap_positions: bool = False,
    ):
        super(MaxwellBoltzmannInit, self).__init__(
            temperature,
            remove_center_of_mass=remove_center_of_mass,
            remove_translation=remove_translation,
            remove_rotation=remove_rotation,
            wrap_positions=wrap_positions,
        )

    def _setup_momenta(self, system: System):
        """
        Initialize the momenta, by drawing from a random normal distribution and rescaling them according to
        Maxwell--Boltzmann statistics.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their replicas.
        """
        if self.temperature.shape[0] == 1:
            temp = self.temperature
        else:
            temp = system.expand_atoms(self.temperature)

        temp = temp.to(system.device)

        # Compute width of Maxwell-Boltzmann distributions for momenta
        stddev = torch.sqrt(system.masses * spk_units.kB * temp)

        # Set initial system momenta
        system.momenta = stddev * torch.randn_like(system.momenta)

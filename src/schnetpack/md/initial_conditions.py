"""
Module for setting up the initial conditions of the molecules in :obj:`schnetpack.md.System`.
This entails sampling the momenta from random distributions corresponding to certain temperatures.
"""
import torch


__all__ = ["Initializer", "MaxwellBoltzmannInit"]


class Initializer:
    """
    Basic initializer class template. Initializes the systems momenta to correspond to a certain temperature.

    Args:
        temperature (float):  Target initialization temperature in Kelvin.
    """

    def __init__(self, temperature):
        self.temperature = temperature

    def initialize_system(self, system):
        """
        Initialize the system according to the instructions given in _setup_momenta.

        Args:
            system (object): System class containing all molecules and their replicas.
        """
        self._setup_momenta(system)

    def _setup_momenta(self, system):
        """
        Main routine for initializing system momenta based on the molecules defined in system and the provided
        temperature. To be implemented.

        Args:
            system (object): System class containing all molecules and their replicas.
        """
        raise NotImplementedError


class MaxwellBoltzmannInit(Initializer):
    """
    Initializes the system momenta according to a Maxwell--Boltzmann distribution at the given temperature.

    Args:
        temperature (float): Target temperature in Kelvin.
        remove_translation (bool): Remove the translational components of the momenta after initialization. Will stop
                                   molecular drift for NVE simulations and NVT simulations with deterministic
                                   thermostats (default=False).
        remove_rotation (bool): Remove the rotational components of the momenta after initialization. Will reduce
                                molecular rotation for NVE simulations and NVT simulations with deterministic
                                thermostats (default=False).
    """

    def __init__(self, temperature, remove_translation=False, remove_rotation=False):
        super(MaxwellBoltzmannInit, self).__init__(temperature)
        self.remove_translation = remove_translation
        self.remove_rotation = remove_rotation

    def _setup_momenta(self, system):
        """
        Initialize the momenta, by drawing from a random normal distribution and rescaling the momenta to the desired
        temperature afterwards. In addition, the system is centered at its center of mass.

        Args:
            system (object): System class containing all molecules and their replicas.
        """
        # Move center of mass to origin
        system.remove_com()

        # Initialize velocities
        velocities = torch.randn(system.momenta.shape, device=system.device)

        # Set initial system momenta and apply atom masks
        system.momenta = velocities * system.masses * system.atom_masks

        # Remove translational motion if requested
        if self.remove_translation:
            system.remove_com_translation()

        # Remove rotational motion if requested
        if self.remove_rotation:
            system.remove_com_rotation()

        # Scale velocities to desired temperature
        scaling = torch.sqrt(self.temperature / system.temperature)
        system.momenta *= scaling[:, :, None, None]

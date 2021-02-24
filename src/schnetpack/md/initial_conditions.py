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

    def initialize_system(self, system, mask=None):
        """
        Initialize the system according to the instructions given in _setup_momenta.

        Args:
            system (object): System class containing all molecules and their replicas.
            mask (array): Mask where 1 indicates that velocities of system will be changed.
        """
        self._setup_momenta(system, mask=mask)

    def _setup_momenta(self, system, mask=None):
        """
        Main routine for initializing system momenta based on the molecules defined in system and the provided
        temperature. To be implemented.

        Args:
            system (object): System class containing all molecules and their replicas.
            mask (array): Mask where 1 indicates that velocities of system will be changed.
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

    def _setup_momenta(self, system, mask=None):
        """
        Initialize the momenta, by drawing from a random normal distribution and rescaling the momenta to the desired
        temperature afterwards. In addition, the system is centered at its center of mass.

        Args:
            system (object): System class containing all molecules and their replicas.
        """
        # Move center of mass to origin
        system.remove_com()

        # Set initial system momenta and apply atom masks
        momenta = (
            torch.randn(system.momenta.shape, device=system.device)
            * system.masses
            * system.atom_masks
        )

        if mask is None:
            system.momenta = momenta
        else:
            system.momenta[mask == 1] = momenta[mask == 1]

        # Remove translational motion if requested
        if self.remove_translation:
            system.remove_com_translation()

        # Remove rotational motion if requested
        if self.remove_rotation:
            system.remove_com_rotation()

        # Scale velocities to desired temperature
        scaling = torch.sqrt(self.temperature / system.temperature)
        if mask is None:
            system.momenta *= scaling[:, :, None, None]
        else:
            system.momenta[mask == 1, ...] *= scaling[mask == 1][:, None, None]

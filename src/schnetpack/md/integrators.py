"""
Integrators are used to propagate the simulated system in time. SchNetPack
provides two basic types of integrators. The Velocity Verlet integrator is a standard
integrator for a purely classical simulations of the nuclei. The ring polymer molecular dynamics
integrator simulates multiple replicas of the system coupled by harmonic springs and recovers
a certain extent of nuclear quantum effects (e.g. tunneling).
"""

import torch
import torch.nn as nn
import numpy as np

import schnetpack as spk
from schnetpack.md import System
from schnetpack.md.simulation_hooks import BarostatHook

from ase import units as ase_units
from schnetpack import units as spk_units

__all__ = ["VelocityVerlet", "RingPolymer", "NPTVelocityVerlet", "NPTRingPolymer"]


class Integrator(nn.Module):
    """
    Basic integrator class template. Uses the typical scheme of propagating
    system momenta in two half steps and system positions in one main step.
    The half steps are defined by default and only the _main_step function
    needs to be specified. Uses atomic time units internally.

    If required, the torch graphs generated by this routine can be detached
    every step via the detach flag.

    Args:
        time_step (float): Integration time step in femto seconds.
    """

    ring_polymer = False
    pressure_control = False

    def __init__(self, time_step: float):
        super(Integrator, self).__init__()
        # Convert fs to internal time units.
        self.time_step = time_step * spk.units.convert_units(
            ase_units.fs, spk_units.time
        )

    def main_step(self, system: System):
        """
        Main integration step wrapper routine to make a default detach
        behavior possible. Calls upon _main_step to perform the actual
        propagation of the system.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        self._main_step(system)

    def half_step(self, system: System):
        """
        Half steps propagating the system momenta according to:

        ..math::
            p = p + \frac{1}{2} F \delta t

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        system.momenta = system.momenta + 0.5 * system.forces * self.time_step

    def _main_step(self, system: System):
        """
        Main integration step to be implemented in derived routines.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        raise NotImplementedError


class VelocityVerlet(Integrator):
    """
    Standard velocity Verlet integrator for non ring-polymer simulations.

    Args:
        time_step (float): Integration time step in femto seconds.
    """

    ring_polymer = False
    pressure_control = False

    def __init__(self, time_step: float):
        super(VelocityVerlet, self).__init__(time_step)

    def _main_step(self, system: System):
        """
        Propagate the positions of the system according to:

        ..math::
            q = q + \frac{p}{m} \delta t

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        system.positions = (
            system.positions + self.time_step * system.momenta / system.masses
        )


class RingPolymer(Integrator):
    """
    Integrator for ring polymer molecular dynamics, as e.g. described in
    [#rpmd1]_

    During the main step, ring polymer positions and momenta are transformed
    from bead to normal mode representation, propagated deterministically and
    then transformed back. Needs the number of beads and the ring polymer
    temperature in order to initialize the propagator matrix. The integrator
    reverts to standard velocity Verlet integration if only one bead is used.

    Uses atomic units of time internally.

    Args:
        time_step (float): Time step in femto seconds.
        n_beads (int): Number of beads in the ring polymer.
        temperature (float): Ring polymer temperature in Kelvin.

    References
    ----------
    .. [#rpmd1] Ceriotti, Parrinello, Markland, Manolopoulos:
       Efficient stochastic thermostatting of path integral molecular dynamics.
       The Journal of Chemical Physics, 133, 124105. 2010.
    """

    ring_polymer = True
    pressure_control = False

    def __init__(self, time_step: float, n_beads: int, temperature: float):
        super(RingPolymer, self).__init__(time_step)

        self.n_beads = n_beads

        # Compute the ring polymer frequency
        self.omega = spk_units.kB * n_beads * temperature / spk_units.hbar

        # Initialize the propagator matrices and normal mode frequencies
        omega_normal, propagator = self._init_propagator()
        self.register_buffer("omega_normal", omega_normal)
        self.register_buffer("propagator", propagator)

    def _init_propagator(self):
        """
        Computes the ring polymer normal mode frequencies and constructs propagator in normal mode representation
        as for example given in [#rpmd2]_.

        Returns:
            torch.Tensor: ring polymer frequencies in normal mode representation
            torch.Tensor: Propagator with the dimension n_beads x 2 x 2,
                          where the last two dimensions mix the systems
                          momenta and positions in normal mode representation.

        References
        ----------
        .. [#rpmd2] Ceriotti, Parrinello, Markland, Manolopoulos:
           Efficient stochastic thermostatting of path integral molecular
           dynamics.
           The Journal of Chemical Physics, 133, 124105. 2010.
        """

        # Set up omega_normal, the ring polymer frequencies in normal mode
        omega_normal = (
            2.0
            * self.omega
            * torch.sin(torch.arange(self.n_beads).float() * np.pi / self.n_beads)
        )

        # Compute basic terms
        omega_dt = omega_normal * self.time_step
        cos_dt = torch.cos(omega_dt)
        sin_dt = torch.sin(omega_dt)

        # Initialize the propagator
        propagator = torch.zeros(self.n_beads, 2, 2)

        # Define the propagator elements, the central normal mode is treated
        # special
        propagator[:, 0, 0] = cos_dt
        propagator[:, 1, 1] = cos_dt
        propagator[:, 0, 1] = -sin_dt * omega_normal
        propagator[1:, 1, 0] = sin_dt[1:] / omega_normal[1:]

        # Centroid normal mode is special as reverts to standard velocity
        # Verlet for one bead.
        propagator[0, 1, 0] = self.time_step

        # Expand dimensions to avoid broadcasting in main_step
        propagator = propagator[..., None, None]

        return omega_normal, propagator

    def _main_step(self, system: System):
        """
        Main propagation step for ring polymer dynamics. First transforms
        positions and momenta to their normal mode representations,
        then applies the propagator defined above (mixing momenta and
        positions accordingly) and performs a backtransformation to the bead
        momenta and positions afterwards, which are used to update the
        current system state.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Transform to normal mode representation
        positions_normal = system.positions_normal
        momenta_normal = system.momenta_normal

        # Propagate ring polymer
        system.momenta_normal = (
            self.propagator[:, 0, 0] * momenta_normal
            + self.propagator[:, 0, 1] * positions_normal * system.masses
        )
        system.positions_normal = (
            self.propagator[:, 1, 0] * momenta_normal / system.masses
            + self.propagator[:, 1, 1] * positions_normal
        )


class NPTVelocityVerlet(VelocityVerlet):
    """
    Verlet integrator for constant pressure dynamics (NPT). Since barostats modify the position update,
    a routine defined in the respectve barostat class is called every main step.

    Args:
        time_step (float): Integration time step in femto seconds.
        barostat (schnetpack.md.simulation_hooks.BarostatHook): Barostat used for constant pressure dynamics.
    """

    ring_polymer = False
    pressure_control = True

    def __init__(self, time_step: float, barostat: BarostatHook):
        super(NPTVelocityVerlet, self).__init__(time_step)
        self.barostat = barostat

    def half_step(self, system: System):
        """
        Half steps propagating the system and barostat momenta.

        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        self.barostat.propagate_half_step(system)

    def _main_step(self, system: System):
        """
        Main integrator step, where the barostat routine is used to propagate the system positions and cells.
        """
        self.barostat.propagate_main_step(system)


class NPTRingPolymer(RingPolymer):
    """
    Ring polymer integrator for constant pressure dynamics (NPT). Here, the barostat modifies the main and the
    half steps.

    Args:
        time_step (float): Time step in femto seconds.
        n_beads (int): Number of beads in the ring polymer.
        temperature (float): Ring polymer temperature in Kelvin.
        barostat (schnetpack.md.simulation_hooks.BarostatHook): Barostat used for constant pressure dynamics.
    """

    ring_polymer = True
    pressure_control = True

    def __init__(
        self, time_step: float, n_beads: int, temperature: float, barostat: BarostatHook
    ):
        super(NPTRingPolymer, self).__init__(time_step, n_beads, temperature)
        self.barostat = barostat

    def half_step(self, system: System):
        """
        Half steps propagating the system and barostat momenta.

        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        self.barostat.propagate_half_step(system)

    def _main_step(self, system: System):
        """
        Perform the main update using the barostat routine.

        Args:
            system (object): System class containing all molecules and their
                             replicas.
        """
        self.barostat.propagate_main_step(system)

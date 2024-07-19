"""
This module contains barostats for controlling the pressure of the system during
ring polymer molecular dynamics simulations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schnetpack.md import Simulator, System

import torch
from schnetpack.md.simulation_hooks import BarostatHook
from schnetpack import units as spk_units
from schnetpack.md.utils import StableSinhDiv

__all__ = ["PILEBarostat"]


class PILEBarostat(BarostatHook):
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

    temperature_control = False
    ring_polymer = True

    def __init__(
        self, target_pressure: float, temperature_bath: float, time_constant: float
    ):
        super(PILEBarostat, self).__init__(
            target_pressure=target_pressure,
            temperature_bath=temperature_bath,
            time_constant=time_constant,
        )
        self.register_buffer("frequency", 1.0 / self.time_constant)

        # Compute kBT, since it will be used a lot
        self.register_buffer("kb_temperature", self.temperature_bath * spk_units.kB)

        self.register_uninitialized_buffer("propagator")
        self.register_uninitialized_buffer("cell_momenta")

        self.register_uninitialized_buffer("c1")
        self.register_uninitialized_buffer("c2")

        self.sinhdx = StableSinhDiv()

    def _init_barostat(self, simulator):
        """
        Initialize the thermostat coefficients and barostat quantities.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                    the time step, system, etc.
        """
        # Get normal mode transformer and propagator for position and cell update
        self.propagator = simulator.integrator.propagator

        # Set up centroid momenta of cell (one for every molecule)
        self.cell_momenta = torch.zeros(
            simulator.system.n_molecules, device=simulator.device, dtype=simulator.dtype
        )
        self.mass = (
            3.0 * simulator.system.n_atoms / self.frequency**2 * self.kb_temperature
        ).to(simulator.dtype)

        # Set up cell thermostat coefficients
        self.c1 = torch.exp(
            -0.5
            * torch.ones(1, device=simulator.device, dtype=simulator.dtype)
            * self.frequency
            * self.time_step
        )
        self.c2 = torch.sqrt(
            simulator.system.n_replicas
            * self.mass
            * self.kb_temperature
            * (1.0 - self.c1**2)
        )

    def on_step_begin(self, simulator: Simulator):
        self._update_barostat()

    def on_step_end(self, simulator: Simulator):
        self._update_barostat()

    def _update_barostat(self):
        """
        Apply the thermostat. This simply propagates the cell momenta under the influence of a PILE thermostat.
        """
        # Propagate cell momenta during half-step
        self.cell_momenta = self.c1 * self.cell_momenta + self.c2 * torch.randn_like(
            self.cell_momenta
        )

    def propagate_main_step(self, system: System):
        """
        Main routine for propagating the ring polymer and the cells. The barostat acts only on the centroid, while the
        remaining replicas are propagated in the conventional way.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        # Get coefficients
        reduced_momenta = (self.cell_momenta / self.mass)[None, :, None]
        coeff_a = torch.exp(-self.time_step * reduced_momenta)
        coeff_b = self.sinhdx.f(self.time_step * reduced_momenta)

        coeff_a_atomic = system.expand_atoms(coeff_a)
        coeff_b_atomic = system.expand_atoms(coeff_b)

        # Transform to normal mode representation
        positions_normal = system.positions_normal
        momenta_normal = system.momenta_normal

        # Propagate centroid mode of the ring polymer
        momenta_normal[0:1] = momenta_normal[0:1] * coeff_a_atomic
        positions_normal[0:1] = (
            positions_normal[0:1] / coeff_a_atomic
            + coeff_b_atomic
            * (momenta_normal[0:1] / system.masses[0:1])
            * self.time_step
        )

        # Update cells
        system.cells = system.cells / coeff_a[..., None]

        # Create copies to avoid overwriting
        positions_normal_tmp = positions_normal.clone()
        momenta_normal_tmp = momenta_normal.clone()

        # Propagate the remaining modes of the ring polymer
        momenta_normal[1:] = (
            self.propagator[1:, 0, 0] * momenta_normal_tmp[1:]
            + self.propagator[1:, 0, 1] * positions_normal_tmp[1:] * system.masses
        )
        positions_normal[1:] = (
            self.propagator[1:, 1, 0] * momenta_normal_tmp[1:] / system.masses
            + self.propagator[1:, 1, 1] * positions_normal_tmp[1:]
        )

        # Transform back to bead representation
        system.positions_normal = positions_normal
        system.momenta_normal = momenta_normal

    def propagate_half_step(self, system: System):
        """
        Propagate the momenta of the thermostat attached to the barostat during each half-step, as well as the particle
        momenta.

        Args:
            system (schnetpack.md.System): System class containing all molecules and their
                             replicas.
        """
        centroid_momenta = system.momenta_normal[0]
        centroid_forces = system.nm_transform.beads2normal(system.forces)[0]

        # Compute pressure component (volume[0] can be used, since the volume is scaled equally for all replicas)
        component_1 = (
            3.0
            * system.n_replicas
            * (
                torch.mean(system.volume, dim=0)
                * (
                    system.compute_centroid_pressure(kinetic_component=True)[0]
                    - self.target_pressure
                )
                + self.kb_temperature
            )
        )

        # Compute components based on forces and momenta
        force_by_mass = centroid_forces / system.masses[0]

        component_2 = system.sum_atoms(
            torch.sum(force_by_mass * centroid_momenta, dim=1)[None, ...]
        )[0]
        component_3 = system.sum_atoms(
            torch.sum(force_by_mass * centroid_forces / 3.0, dim=1)[None, ...]
        )[0]

        # Update cell momenta
        self.cell_momenta += (
            +(0.5 * self.time_step) * component_1[:, 0]
            + (0.5 * self.time_step) ** 2 * component_2
            + (0.5 * self.time_step) ** 3 * component_3
        )

        # Update the system momenta
        system.momenta = system.momenta + 0.5 * system.forces * self.time_step

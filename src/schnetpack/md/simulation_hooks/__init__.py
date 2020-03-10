from .basic_hooks import *
from .thermostats import *
from .barostats import *
from .sampling import *
from .logging_hooks import *

# class MTKBarostat(BarostatHook):
#    def __init__(self, target_pressure, time_constant, temperature_bath, detach=True):
#        super(MTKBarostat, self).__init__(
#           target_pressure=target_pressure,
#          temperature_bath=temperature_bath,
#          detach=detach,
#      )
#      self.frequency = 1.0 / (time_constant * MDUnits.fs2atu)
#      self.inv_kb_temperature = 1.0 / (self.temperature_bath * MDUnits.kB)

#    def _init_barostat(self, simulator):
# TODO: perform checks here?

#        # Get scaling factors based on the number of atoms
#        n_atoms = simulator.system.n_atoms[None, :, None, None].float()
#        self.inv_sqrt_dof = 1.0 / torch.sqrt(3.0 * n_atoms + 1.0)
#        self.inv_dof = 1.0 / n_atoms
#        self.weight = self.frequency * self.inv_sqrt_dof
#
#        # Thermostat variable, this will be used for scaling the positions and momenta
#        self.zeta = torch.zeros(
#            self.n_replicas, self.n_molecules, 1, 1, device=self.device
#        )
#
#    def _apply_barostat(self, simulator):
#        # TODO: check dimensions and pressure units
#
#        # 1) Update barostat variable
#        # pressure_factor = 3.0 * simulator.system.volume * (
#        #        simulator.system.compute_pressure(kinetic_component=True) - self.target_pressure
#        # ) + 2.0 * self.inv_dof * simulator.system.kinetic_energy
#
#        # The pressure factor is 3*V * (Pint - Pext) + 2*Ekin / N
#        # Since Pint also contains the kinetic component it computes as
#        #   Pint = 1/(3V) * 2 * Ekin + Pint'
#        # This means, the expression can be rewritten as
#        #   Pfact = 3*V*(Pint' - Pext) + 2*(1+1/N) * Ekin
#        # Saving some computations
#        pressure_factor = (
#                3.0
#                * simulator.system.volume
#                * (simulator.system.compute_pressure() - self.target_pressure)
#                + 2.0 * (1 + self.inv_dof) * simulator.system.kinetic_energy
#        )
#
#        self.zeta = (
#                self.zeta
#                + 0.5
#                * self.time_step
#                * self.weight
#                * self.inv_kb_temperature
#                * pressure_factor
#        )
#
#        # 2) Scale positions and cell
#        scaling = torch.exp(self.time_step * self.weight * self.zeta)
#
#        simulator.system.positions = simulator.system.positions * scaling
#        simulator.system.cells = simulator.system.cells * scaling
#
#        # 3) Scale momenta
#        scaling = torch.exp(
#            -self.time_step * self.weight * (1 + self.inv_dof) * self.zeta
#        )
#        simulator.system.momenta = simulator.system.momenta * scaling
#
#        # 4) Second update of barostat variable based on new momenta and positions
#        pressure_factor = (
#                3.0
#                * simulator.system.volume
#                * (simulator.system.compute_pressure() - self.target_pressure)
#                + 2.0 * (1 + self.inv_dof) * simulator.system.kinetic_energy
#        )
#
#        self.zeta = (
#                self.zeta
#                + 0.5
#                * self.time_step
#                * self.weight
#                * self.inv_kb_temperature
#                * pressure_factor
#       )
#
#        if self.detach:
#            simulator.system.cells = simulator.system.cells.detach()
#            simulator.system.positions = simulator.system.positions.detach()
#            simulator.system.momenta = simulator.system.momenta.detach()
#            self.zeta = self.zeta.detach()

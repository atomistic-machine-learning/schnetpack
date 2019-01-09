import os
from sacred import Ingredient

from schnetpack.simulate.simulator import Simulator
from schnetpack.simulate.hooks import *
from schnetpack.simulate.thermostats import *
from schnetpack.md.integrators import *
from schnetpack.md.calculators import SchnetPackCalculator


simulator_ingredient = Ingredient('simulator')


@simulator_ingredient.config
def config():
    system = None
    integrator = 'velocity_verlet'
    calculator = None
    simulator_hooks = []
    step = 0
    thermostat = 'berendsen'


@simulator_ingredient.named_config
def berendsen_thermostat():
    thermostat = 'berendsen'
    bath_temperature = 50.
    transfer_time = 1.


@simulator_ingredient.named_config
def gle_thermostat():
    bath_temperature = 50.
    gle_file = './some_file.txt'
    nm_transformation = None


@simulator_ingredient.named_config
def velocity_verlet():
    integrator = 'velocity_verlet'
    time_step = 1


@simulator_ingredient.named_config
def ring_polymer():
    integrator = 'ring_polymer'
    n_beads = 10
    time_step = 1
    temperature = 50.
    transformation = NormalModeTransformer
    device = 'cuda'


@simulator_ingredient.capture
def build_simulator(system, integrator, calculator_object, simulator_hooks,
                    step, thermostat):
    hook_objects = build_simulator_hooks(simulator_hooks, thermostat)
    integrator_object = build_integrator(integrator)
    return Simulator(system=system, integrator=integrator_object,
                     calculator=calculator_object, simulator_hooks=hook_objects,
                     step=step)


@simulator_ingredient.capture
def get_velocity_verlet(time_step):
    return VelocityVerlet(time_step=time_step)


@simulator_ingredient.capture
def get_ring_polymer(n_beads, time_step, temperature,
                     transformation=NormalModeTransformer, device='cuda'):
    return RingPolymer(n_beads=n_beads, time_step=time_step,
                       temperature=temperature, transformation=transformation,
                       device=device)


@simulator_ingredient.capture
def build_integrator(integrator):
    if integrator == 'velocity_verlet':
        return get_velocity_verlet()
    elif integrator == 'ring_polymer':
        return get_ring_polymer()
    else:
        raise NotImplementedError


@simulator_ingredient.capture
def build_simulator_hooks(simulator_hooks, thermostat):
    hook_objects = [build_thermostat(thermostat)] if thermostat else []
    hook_objects += build_logging_hooks(simulator_hooks)
    return hook_objects


@simulator_ingredient.capture
def build_logging_hooks(simulator_hooks):
    hook_objects = []
    for hook in simulator_hooks:
        if hook == 'test':
            hook_objects.append('')
        elif hook == 'test2':
            hook_objects.append('')
        else:
            raise NotImplementedError


@simulator_ingredient.capture
def get_berendsen_thermostat(bath_temperature, transfer_time):
    return BerendsenThermostat(temperature_bath=bath_temperature,
                               transfer_time=transfer_time)


@simulator_ingredient.capture
def get_gle_thermostat(bath_temperature, gle_file, nm_transformation):
    return GLEThermostat(bath_temperature=bath_temperature,
                         gle_file=gle_file, nm_transformation=nm_transformation)


@simulator_ingredient.capture
def build_thermostat(thermostat):
    if thermostat == 'berendsen':
        return get_berendsen_thermostat()
    elif thermostat == 'gle':
        return get_gle_thermostat()
    else:
        return NotImplementedError

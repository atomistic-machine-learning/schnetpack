from sacred import Ingredient

from schnetpack.simulate.thermostats import *


thermostat_ingredient = Ingredient('thermostat')


@thermostat_ingredient.config
def config():
    """configuration for the thermostat ingredient"""
    thermostat = None


@thermostat_ingredient.named_config
def berendsen():
    """configuration for the berendsen thermostat"""
    thermostat = 'berendsen'
    bath_temperature = 300.
    transfer_time = 1.


@thermostat_ingredient.named_config
def gle():
    """configuration for the GLE thermostat"""
    thermostat = 'gle'
    bath_temperature = 300.
    gle_file = './some_file.txt'
    nm_transformation = None


@thermostat_ingredient.named_config
def piglet():
    """configuration for the piglet thermostat"""
    thermostat = 'piglet'
    bath_temperature = 300.
    gle_file = './some_file.txt'
    nm_transformation = None


@thermostat_ingredient.named_config
def langevin():
    """configuration for the langevin thermostat"""
    thermostat = 'langevin'
    bath_temperature = 300.
    time_constant = 2.


@thermostat_ingredient.named_config
def pile_local():
    """configuration for the pile-local thermostat"""
    thermostat = 'pile_local'
    bath_temperature = 300.
    time_constant = 2.


@thermostat_ingredient.named_config
def pile_global():
    """configuration for the pile-global thermostat"""
    thermostat = 'pile_global'
    bath_temperature = 300.
    time_constant = 2.


@thermostat_ingredient.named_config
def nhc():
    """configuration for the nhc thermostat"""
    thermostat = 'nhc'
    bath_temperature = 300.
    time_constant = 2.
    chain_length = 3
    massive = False
    nm_transformation = None
    multi_step = 2
    integration_order = 3


@thermostat_ingredient.named_config
def nhc_ring_polymer():
    """configuration for the nhc-ring-polymer thermostat"""
    thermostat = 'nhc_ring_polymer'
    bath_temperature = 300.
    time_constant = 2.
    chain_length = 3
    local = True
    nm_transformation = None
    multi_step = 2
    integration_order = 3


@thermostat_ingredient.capture
def get_langevin_thermostat(bath_temperature, time_constant):
    return LangevinThermostat(bath_temperature, time_constant)


@thermostat_ingredient.capture
def get_piglet_thermostat(bath_temperature, gle_file):
    return PIGLETThermostat(bath_temperature, gle_file)


@thermostat_ingredient.capture
def get_berendsen_thermostat(bath_temperature, transfer_time):
    return BerendsenThermostat(temperature_bath=bath_temperature,
                               transfer_time=transfer_time)


@thermostat_ingredient.capture
def get_gle_thermostat(bath_temperature, gle_file):
    return GLEThermostat(bath_temperature=bath_temperature,
                         gle_file=gle_file)


@thermostat_ingredient.capture
def get_pile_local_thermostat(bath_temperature, time_constant):
    return PILELocalThermostat(temperature_bath=bath_temperature,
                               time_constant=time_constant)


@thermostat_ingredient.capture
def get_pile_global_thermostat(bath_temperature, time_constant):
    return PILEGlobalThermostat(temperature_bath=bath_temperature,
                                time_constant=time_constant)


@thermostat_ingredient.capture
def get_nhc_thermostat(bath_temperature, time_constant, chain_length,
                       massive, nm_transformation, multi_step,
                       integration_order):
    return NHCThermostat(bath_temperature, time_constant, chain_length, massive,
                         nm_transformation, multi_step, integration_order)


@thermostat_ingredient.capture
def get_nhc_ring_polymer_thermostat(bath_temperature, time_constant,
                                    chain_length, local, nm_transformation,
                                    multi_step, integration_order):
    return NHCThermostat(bath_temperature, time_constant, chain_length, local,
                         nm_transformation, multi_step, integration_order)


@thermostat_ingredient.capture
def build_thermostat(thermostat):
    if thermostat == 'berendsen':
        return get_berendsen_thermostat()
    elif thermostat == 'gle':
        return get_gle_thermostat()
    elif thermostat == 'piglet':
        return get_piglet_thermostat()
    elif thermostat == 'langevin':
        return get_langevin_thermostat()
    elif thermostat == 'pile_local':
        return get_pile_local_thermostat()
    elif thermostat == 'pile_global':
        return get_pile_global_thermostat()
    elif thermostat == 'nhc':
        return get_nhc_thermostat()
    elif thermostat == 'nhc_ring_polymer':
        return get_nhc_ring_polymer_thermostat()
    else:
        raise NotImplementedError

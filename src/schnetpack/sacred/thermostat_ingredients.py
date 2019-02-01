from sacred import Ingredient

from schnetpack.simulate.thermostats import *

thermostat_ingredient = Ingredient('thermostat')


@thermostat_ingredient.capture
def thermostat_info(_log, name=None, temperature=None, addition=None):
    string = ''
    if name is None:
        string += 'No thermostat is used.'
    else:
        if temperature is not None:
            string += 'Using {:s} thermostat at {:.2f} K'.format(name, temperature)
        elif addition is not None:
            if type(addition) == str:
                string += ' and parameters loaded from {:s}.'.format(addition)
            else:
                string += ' with a time constant of {:.1f} fs'.format(addition)
    _log.info(string)


@thermostat_ingredient.config
def config():
    """configuration for the thermostat ingredient"""
    thermostat = None


@thermostat_ingredient.named_config
def berendsen():
    """configuration for the berendsen thermostat"""
    thermostat = 'berendsen'
    bath_temperature = 300.
    time_constant = 100.


@thermostat_ingredient.named_config
def gle():
    """configuration for the GLE thermostat"""
    thermostat = 'gle'
    bath_temperature = 300.
    gle_file = './some_file.txt'


@thermostat_ingredient.named_config
def piglet():
    """configuration for the piglet thermostat"""
    thermostat = 'piglet'
    bath_temperature = 300.
    gle_file = './some_file.txt'


@thermostat_ingredient.named_config
def langevin():
    """configuration for the langevin thermostat"""
    thermostat = 'langevin'
    bath_temperature = 300.
    time_constant = 100.


@thermostat_ingredient.named_config
def pile_local():
    """configuration for the pile-local thermostat"""
    thermostat = 'pile_local'
    bath_temperature = 300.
    time_constant = 100.


@thermostat_ingredient.named_config
def pile_global():
    """configuration for the pile-global thermostat"""
    thermostat = 'pile_global'
    bath_temperature = 300.
    time_constant = 100.


@thermostat_ingredient.named_config
def nhc():
    """configuration for the nhc thermostat"""
    thermostat = 'nhc'
    bath_temperature = 300.
    time_constant = 100.
    chain_length = 3
    multi_step = 2
    integration_order = 3


@thermostat_ingredient.named_config
def nhc_massive():
    """configuration for the nhc thermostat"""
    thermostat = 'nhc_massive'
    bath_temperature = 300.
    time_constant = 100.
    chain_length = 3
    multi_step = 2
    integration_order = 3


@thermostat_ingredient.named_config
def nhc_ring_polymer():
    """configuration for the nhc-ring-polymer thermostat"""
    thermostat = 'nhc_ring_polymer'
    bath_temperature = 300.
    time_constant = 100.
    chain_length = 3
    multi_step = 2
    integration_order = 3


@thermostat_ingredient.named_config
def nhc_ring_polymer_global():
    """configuration for the nhc-ring-polymer thermostat"""
    thermostat = 'nhc_ring_polymer_global'
    bath_temperature = 300.
    time_constant = 100.
    chain_length = 3
    multi_step = 2
    integration_order = 3


@thermostat_ingredient.capture
def get_no_thermostat():
    thermostat_info(name=None)
    return None


@thermostat_ingredient.capture
def get_langevin_thermostat(bath_temperature, time_constant):
    thermostat_info(name='Langevin', temperature=bath_temperature,
                    addition=time_constant)
    return LangevinThermostat(bath_temperature, time_constant)


@thermostat_ingredient.capture
def get_piglet_thermostat(bath_temperature, gle_file):
    thermostat_info(name='PiGLET', temperature=bath_temperature,
                    addition=gle_file)
    return PIGLETThermostat(bath_temperature, gle_file)


@thermostat_ingredient.capture
def get_berendsen_thermostat(bath_temperature, time_constant):
    thermostat_info(name='Berendsen', temperature=bath_temperature,
                    addition=time_constant)
    return BerendsenThermostat(temperature_bath=bath_temperature,
                               time_constant=time_constant)


@thermostat_ingredient.capture
def get_gle_thermostat(bath_temperature, gle_file):
    thermostat_info(name='GLE', temperature=bath_temperature,
                    addition=gle_file)
    return GLEThermostat(bath_temperature=bath_temperature,
                         gle_file=gle_file)


@thermostat_ingredient.capture
def get_pile_local_thermostat(bath_temperature, time_constant):
    thermostat_info(name='local PILE', temperature=bath_temperature,
                    addition=time_constant)
    return PILELocalThermostat(temperature_bath=bath_temperature,
                               time_constant=time_constant)


@thermostat_ingredient.capture
def get_pile_global_thermostat(bath_temperature, time_constant):
    thermostat_info(name='global PILE', temperature=bath_temperature,
                    addition=time_constant)
    return PILEGlobalThermostat(temperature_bath=bath_temperature,
                                time_constant=time_constant)


@thermostat_ingredient.capture
def get_nhc_thermostat(bath_temperature, time_constant, chain_length,
                       multi_step, integration_order, massive=False):
    name = f'{("", "massive ")[massive]}NHC'
    thermostat_info(name=name, temperature=bath_temperature,
                    addition=time_constant)
    return NHCThermostat(bath_temperature, time_constant,
                         chain_length=chain_length,
                         integration_order=integration_order,
                         multi_step=multi_step,
                         massive=massive)


@thermostat_ingredient.capture
def get_nhc_ring_polymer_thermostat(bath_temperature, time_constant,
                                    chain_length, multi_step,
                                    integration_order, local=True):
    name = f'{("global", "local")[local]} PI-NHC'
    thermostat_info(name=name, temperature=bath_temperature,
                    addition=time_constant)
    return NHCRingPolymerThermostat(bath_temperature, time_constant,
                                    chain_length=chain_length,
                                    integration_order=integration_order,
                                    multi_step=multi_step,
                                    local=local)


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
        return get_nhc_thermostat(massive=False)
    elif thermostat == 'nhc_massive':
        return get_nhc_thermostat(massive=True)
    elif thermostat == 'nhc_ring_polymer':
        return get_nhc_ring_polymer_thermostat(local=True)
    elif thermostat == 'nhc_ring_polymer_global':
        return get_nhc_ring_polymer_thermostat(local=False)
    elif thermostat is None:
        return get_no_thermostat()
    else:
        raise NotImplementedError

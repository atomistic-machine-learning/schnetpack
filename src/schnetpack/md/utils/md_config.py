from omegaconf import DictConfig

import schnetpack.md.calculators

config_aliases = {
    "uniform": "schnetpack.md.UniformInit",
    "maxwell-boltzmann": "schnetpack.md.MaxwellBoltzmannInit",
    "schnetpack": "schnetpack.md.calculators.SchnetPackCalculator",
    "ase": "schnetpack.md.neighborlist_md.ASENeighborListMD",
    "torch": "schnetpack.md.neighborlist_md.TorchNeighborListMD",
    "berendsen": "schnetpack.md.simulation_hooks.BerendsenThermostat",
}


class AliasError(Exception):
    pass


def get_alias(name):
    if name in config_aliases:
        return config_aliases[name]
    else:
        return name


def antialias_config(config: DictConfig):
    if "_target_" in config:
        config._target_ = get_alias(config._target_)
    return config

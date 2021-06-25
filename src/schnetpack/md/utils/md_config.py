import logging
from typing import Union, Dict
from omegaconf import DictConfig
from schnetpack.utils import str2class

log = logging.getLogger(__name__)

config_aliases = {
    "uniform": "schnetpack.md.UniformInit",
    "maxwell-boltzmann": "schnetpack.md.MaxwellBoltzmannInit",
    "schnetpack": "schnetpack.md.calculators.SchnetPackCalculator",
    "ase": "schnetpack.md.neighborlist_md.ASENeighborListMD",
    "torch": "schnetpack.md.neighborlist_md.TorchNeighborListMD",
    "berendsen": "schnetpack.md.simulation_hooks.BerendsenThermostat",
    "verlet": "schnetpack.md.integrators.VelocityVerlet",
    "verlet_npt": "schnetpack.md.integrators.NPTVelocityVerlet",
    "rpmd": "schnetpack.md.integrators.RingPolymer",
    "rpmd_npt": "schnetpack.md.integrators.NPTRingPolymer",
    "checkpoint": "schnetpack.md.simulation_hooks.Checkpoint",
    "file_logger": "schnetpack.md.simulation_hooks.FileLogger",
    "molecules": "schnetpack.md.simulation_hooks.MoleculeStream",
    "properties": "schnetpack.md.simulation_hooks.PropertyStream",
}

integrator_to_npt = {
    "schnetpack.md.integrators.VelocityVerlet": "schnetpack.md.integrators.NPTVelocityVerlet",
    "schnetpack.md.integrators.RingPolymer": "schnetpack.md.integrators.NPTRingPolymer",
}


def get_alias(name: str):
    if name in config_aliases:
        return config_aliases[name]
    else:
        return name


def config_alias(config: Union[Dict, DictConfig]):
    if "_target_" in config:
        config._target_ = get_alias(config._target_)
    return config


def is_rpmd_integrator(integrator_type: str):
    integrator_type = get_alias(integrator_type)
    integrator_class = str2class(integrator_type)

    if hasattr(integrator_class, "ring_polymer"):
        return integrator_class.ring_polymer
    else:
        log.warning(
            "Could not determine if integrator is suitable for ring polymer simulations."
        )
        return False


def get_npt_integrator(integrator_type: str):
    integrator_type = get_alias(integrator_type)
    integrator_class = str2class(integrator_type)

    if hasattr(integrator_class, "pressure_control"):

        if integrator_class.pressure_control:
            return integrator_type
        else:
            # Integrator is already constant pressure integrator
            if integrator_type in integrator_to_npt.values():
                return integrator_type
            else:
                # Look for constant pressure equivalent
                if integrator_type in integrator_to_npt:
                    log.info(
                        "Switching integrator from {:s} to {:s} for constant pressure simulation...".format(
                            integrator_type, integrator_to_npt[integrator_type]
                        )
                    )
                    return integrator_to_npt[integrator_type]
                # If NPT suitability can not be determined automatically, good luck
                else:
                    log.warning(
                        "Please check whether integrator {:s} is suitable for constant pressure simulations.".format(
                            integrator_type
                        )
                    )
                    return integrator_type
    else:
        log.warning(
            "Please check whether integrator {:s} is suitable for constant pressure simulations.".format(
                integrator_type
            )
        )
        return integrator_type

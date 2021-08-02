import logging
from typing import Union, Dict
from omegaconf import DictConfig
from schnetpack.utils import str2class

log = logging.getLogger(__name__)

config_aliases = {
    # Initial conditions
    "uniform": "schnetpack.md.UniformInit",
    "maxwell-boltzmann": "schnetpack.md.MaxwellBoltzmannInit",
    # Calculators
    "schnetpack": "schnetpack.md.calculators.SchnetPackCalculator",
    # Neighbor lists
    "ase": "schnetpack.md.neighborlist_md.ASENeighborListMD",
    "torch": "schnetpack.md.neighborlist_md.TorchNeighborListMD",
    # Thermostats
    "berendsen": "schnetpack.md.simulation_hooks.BerendsenThermostat",
    "langevin": "schnetpack.md.simulation_hooks.LangevinThermostat",
    "nhc": "schnetpack.md.simulation_hooks.NHCThermostat",
    "pile_local": "schnetpack.md.simulation_hooks.PILELocalThermostat",
    "pile_global": "schnetpack.md.simulation_hooks.PILEGlobalThermostat",
    "trpmd": "schnetpack.md.simulation_hooks.TRPMDThermostat",
    "gle": "schnetpack.md.simulation_hooks.GLEThermostat",
    "pi_gle": "schnetpack.md.simulation_hooks.RPMDGLEThermostat",
    "piglet": "schnetpack.md.simulation_hooks.PIGLETThermostat",
    "pi_nhc": "schnetpack.md.simulation_hooks.NHCRingPolymerThermostat",
    # Barostats:
    "nhc_barostat_iso": "schnetpack.md.simulation_hooks.NHCBarostatIsotropic",
    "nhc_barostat_aniso": "schnetpack.md.simulation_hooks.NHCBarostatAnisotropic",
    "pile_barostat": "schnetpack.md.simulation_hooks.PILEBarostat",
    # Integrators
    "verlet": "schnetpack.md.integrators.VelocityVerlet",
    "verlet_npt": "schnetpack.md.integrators.NPTVelocityVerlet",
    "rpmd": "schnetpack.md.integrators.RingPolymer",
    "rpmd_npt": "schnetpack.md.integrators.NPTRingPolymer",
    # Logging
    "checkpoint": "schnetpack.md.simulation_hooks.Checkpoint",
    "file_logger": "schnetpack.md.simulation_hooks.FileLogger",
    "tensorboard_logger": "schnetpack.md.simulation_hooks.TensorBoardLogger",
    "molecules": "schnetpack.md.simulation_hooks.MoleculeStream",
    "properties": "schnetpack.md.simulation_hooks.PropertyStream",
}

integrator_to_npt = {
    "schnetpack.md.integrators.VelocityVerlet": "schnetpack.md.integrators.NPTVelocityVerlet",
    "schnetpack.md.integrators.RingPolymer": "schnetpack.md.integrators.NPTRingPolymer",
}


def get_alias(name: str):
    """
    Find original class bases on aliases defined in `schnetpack.md.utils.md_config.config_aliases`.

    Args:
        name (str): alias of class.

    Returns:
        str: full name of class.
    """
    if name in config_aliases:
        return config_aliases[name]
    else:
        log.warning(f"{name} not found in aliases, assuming class")
        return name


def config_alias(config: Union[Dict, DictConfig]):
    """
    Resolve aliases in hydra `_target_` entries for automatic instantiation.

    Args:
        config (DictConfig): config dictionary.

    Returns:
        DictConfig: config disctionary with resolved alias.
    """
    if "_target_" in config:
        config._target_ = get_alias(config._target_)
    return config


def is_rpmd_integrator(integrator_type: str):
    """
    Check if an integrator is suitable for ring polymer molecular dynamics.

    Args:
        integrator_type (str): integrator class name

    Returns:
        bool: True if integrator is suitable, False otherwise.
    """
    integrator_class = str2class(integrator_type)

    if hasattr(integrator_class, "ring_polymer"):
        return integrator_class.ring_polymer
    else:
        log.warning(
            "Could not determine if integrator is suitable for ring polymer simulations."
        )
        return False


def get_npt_integrator(integrator_type: str):
    """
    Check if integrator is suitable for constant pressure dynamics and determine the constant pressure equivalent if
    this is not the case.

    Args:
        integrator_type (str): name of the integrator class.

    Returns:
        str: class of suitable constant pressure integrator.
    """
    integrator_class = str2class(integrator_type)

    if hasattr(integrator_class, "pressure_control"):
        if integrator_class.pressure_control:
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
                    "No constant pressure equivalent for integrator {:s} could be found.".format(
                        integrator_type
                    )
                )
            return integrator_type
    else:
        log.warning(
            "Please check whether integrator {:s} is suitable for constant pressure"
            " simulations (`pressure control` attribute).".format(integrator_type)
        )
        return integrator_type

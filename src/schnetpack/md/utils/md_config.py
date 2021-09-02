import logging
from schnetpack.utils import str2class

log = logging.getLogger(__name__)

integrator_to_npt = {
    "schnetpack.md.integrators.VelocityVerlet": "schnetpack.md.integrators.NPTVelocityVerlet",
    "schnetpack.md.integrators.RingPolymer": "schnetpack.md.integrators.NPTRingPolymer",
}


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

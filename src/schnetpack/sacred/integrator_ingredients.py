from sacred import Ingredient
from schnetpack.md.integrators import *
from schnetpack.md.utils import NormalModeTransformer

integrator_ingredient = Ingredient("integrator")


@integrator_ingredient.config
def config():
    """configuration for the integrator ingredient"""
    integrator = "velocity_verlet"
    time_step = 0.5


@integrator_ingredient.named_config
def velocity_verlet():
    """configuration for the velocity verlet integrator"""
    integrator = "velocity_verlet"
    time_step = 0.5


@integrator_ingredient.named_config
def ring_polymer():
    """configuration for the ring polymer integrator"""
    integrator = "ring_polymer"
    time_step = 0.2
    temperature = 300.0


@integrator_ingredient.capture
def get_velocity_verlet(time_step, device="cuda"):
    return VelocityVerlet(time_step=time_step, device=device)


@integrator_ingredient.capture
def get_ring_polymer(n_beads, time_step, temperature, device="cuda"):
    return RingPolymer(
        n_beads=n_beads,
        time_step=time_step,
        temperature=temperature,
        transformation=NormalModeTransformer,
        device=device,
    )


@integrator_ingredient.capture
def build_integrator(_log, integrator, device, n_beads):
    """
    build the integrator object

    Args:
        integrator (str): name of the integrator

    Returns:
        integrator object
    """
    _log.info(
        f'Using {("Velocity Verlet", "Ring Polymer")[integrator == "ring_polymer"]} integrator '
        f'with {n_beads} {("replicas", "beads")[integrator == "ring_polymer"]}'
    )

    if integrator == "velocity_verlet":
        return get_velocity_verlet(device=device)
    elif integrator == "ring_polymer":
        if n_beads == 1:
            _log.warning(
                "Using Ring Polymer integrator with only 1 bead. "
                "In this case Velocity Verlet is more efficient."
            )
        return get_ring_polymer(n_beads=n_beads, device=device)
    else:
        raise NotImplementedError

from sacred import Ingredient
from schnetpack.md.integrators import *
from schnetpack.md.utils import NormalModeTransformer


integrator_ingredient = Ingredient('integrator')


@integrator_ingredient.config
def config():
    """configuration for the integrator ingredient"""
    integrator = 'velocity_verlet'
    time_step = 1


@integrator_ingredient.named_config
def velocity_verlet():
    """settings for the velocity verlet integrator"""
    integrator = 'velocity_verlet'
    time_step = 1


@integrator_ingredient.named_config
def ring_polymer():
    """settings for the ring polymer integrator"""
    integrator = 'ring_polymer'
    n_beads = 10
    time_step = 1
    temperature = 50.
    transformation = NormalModeTransformer
    device = 'cuda'


@integrator_ingredient.capture
def get_velocity_verlet(time_step):
    return VelocityVerlet(time_step=time_step)


@integrator_ingredient.capture
def get_ring_polymer(n_beads, time_step, temperature,
                     transformation=NormalModeTransformer, device='cuda'):
    return RingPolymer(n_beads=n_beads, time_step=time_step,
                       temperature=temperature, transformation=transformation,
                       device=device)


@integrator_ingredient.capture
def build_integrator(integrator):
    """
    build the integrator object

    Args:
        integrator (str): name of the integrator

    Returns:
        integrator object
    """
    if integrator == 'velocity_verlet':
        return get_velocity_verlet()
    elif integrator == 'ring_polymer':
        return get_ring_polymer()
    else:
        raise NotImplementedError

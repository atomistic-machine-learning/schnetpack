from sacred import Ingredient
from schnetpack.md.initial_conditions import MaxwellBoltzmannInit

initializer_ing = Ingredient('initializer')


@initializer_ing.config
def config():
    """configuration for the initializer ingredient"""
    initializer = 'maxwell_boltzmann'
    init_temperature = 300
    remove_translation = False
    remove_rotation = False


@initializer_ing.named_config
def remove_com():
    remove_translation = True
    remove_rotation = True


@initializer_ing.capture
def build_initializer(initializer, init_temperature, remove_translation,
                      remove_rotation):
    if initializer == 'maxwell_boltzmann':
        return MaxwellBoltzmannInit(init_temperature,
                                    remove_translation=remove_translation,
                                    remove_rotation=remove_rotation)
    else:
        raise NotImplementedError

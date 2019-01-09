from sacred import Experiment

from schnetpack.sacred.calculator_ingredients import calculator_ingradient, build_calculator
from schnetpack.sacred.simulator_ingredients import simulator_ingredient, build_simulator
from schnetpack.sacred.model_ingredients import model_ingredient, build_model


md = Experiment('md', ingredients=[simulator_ingredient, calculator_ingradient])


@md.config
def config():
    pass

@md.capture
def setup_simulation():
    calculator = build_calculator()

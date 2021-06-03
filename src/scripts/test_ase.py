import ase
from ase import Atoms
from ase.io import write
import torch
import copy
import argparse

Z = [6, 6, 1, 1, 1, 1, 1, 8, 1]
R = [
    [-4.91832096, 1.53666755, -0.06624112],
    [-3.41563316, 1.44992331, -0.14155274],
    [-5.22246355, 2.28682457, 0.66707348],
    [-5.34200431, 0.57373599, 0.22595757],
    [-5.3338372, 1.81223882, -1.03759683],
    [-3.00608229, 1.18685677, 0.84392479],
    [-2.99789458, 2.42727565, -0.42151118],
    [-3.0796312, 0.46731581, -1.10256879],
    [-2.12365275, 0.40566152, -1.15678517],
]

from schnetpack.interfaces import SpkCalculator, AseInterface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to model file.")
    args = parser.parse_args()

    atoms = Atoms(Z, R)

    model = torch.jit.load(args.model, map_location="cuda")

    # Initialize the calculator
    calc = SpkCalculator(model, 5.0)

    # Perform the calculation
    calc.calculate(atoms, properties=["energy", "forces"])

    print(calc.results)
    print(" I now INIT THE MODEL")

    write("test-inpu.xyz", atoms, format="xyz")

    aseinf = AseInterface("test-inpu.xyz", "workdir", model, 5.0, precision="float32")

    aseinf.calculate_single_point()
    aseinf.optimize()
    aseinf.save_molecule("TEST_SAVE")
    aseinf.compute_normal_modes()
    aseinf.init_md("TEST_MD")
    aseinf.run_md(100)

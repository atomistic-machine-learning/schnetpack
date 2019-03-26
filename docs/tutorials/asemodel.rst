.. _ase interface:

Using a trained model with ASE
==============================

In order to use SchNet as predictor for ase calculators, use the ``MLPotential``
class from ``schnetpack.ase_interface``. The trained model needs to be
wrapped in the ``Model`` class and can then be used by the calculator::

    import torch
    from schnetpack.ase_interface import Model, MLPotential

    # build wrapped model
    path_to_model = 'path/to/best_model'
    model = torch.load(path_to_model)
    wrapped_model = Model(model, type='schnet', device='cpu')
    # build calculator
    calculator = MLPotential(wrapped_model)

The calculator can then be used like any other ase calculator::

    from ase.db import connect

    # load an example atoms object
    conn = connect('data/snippet.db')
    atoms = conn.get_atoms(1)
    # set calculator
    atoms.set_calculator(calculator)

The atoms object can now be used for your molecular dynamics code::

    print('forces', atoms.get_forces())
    print('energy', atoms.get_total_energy())

An example-script for this tutorial can be found at *examples/ase_interface.py*.


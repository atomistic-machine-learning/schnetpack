schnetpack
==========
.. currentmodule:: schnetpack

Structure attributes
--------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    properties.Z
    properties.R
    properties.cell
    properties.pbc
    properties.idx_m
    properties.idx_i
    properties.idx_j
    properties.Rij
    properties.n_atoms

Units
-----
.. autosummary::
    :toctree: generated
    :nosignatures:

    units.convert_units

Saving and loading models
-------------------------
A model folder holds ``hyperparameters.json`` and ``model.safetensors``; see
:doc:`/guides/saving_models`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    save_model
    load_model
    load_hyperparameters
    register

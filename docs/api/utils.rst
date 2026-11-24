schnetpack.utils
================
.. currentmodule:: schnetpack.utils

Loading pickled models
----------------------

Model files pickled by SchNetPack 2.x or an earlier 3.0 version. A model folder
loads with :func:`schnetpack.load_model`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    load_model

Unit cells
----------

Functions on unit cells and periodic positions. They take tensors and return
tensors or plain numbers.

.. currentmodule:: schnetpack.utils.cell

.. autosummary::
    :toctree: generated
    :nosignatures:

    cell_to_params
    params_to_cell
    valid_params
    cell_statistics
    niggli_reduce
    structure_index
    atom_cells
    with_cell
    to_frac
    to_cart
    wrap
    minimum_image

Helpers
-------

.. currentmodule:: schnetpack.utils

.. autosummary::
    :toctree: generated
    :nosignatures:

    as_dtype
    int2precision
    str2class
    required_fields_from_properties

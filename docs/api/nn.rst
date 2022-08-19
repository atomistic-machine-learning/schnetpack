schnetpack.nn
=============
.. currentmodule:: nn


Basic layers
------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Dense


Equivariant layers
------------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    GatedEquivariantBlock


Radial basis
------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    GaussianRBF
    GaussianRBFCentered
    BesselRBF


Cutoff
------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    CosineCutoff
    MollifierCutoff


Activations
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    shifted_softplus


Ops
---

.. autosummary::
    :toctree: generated
    :nosignatures:

    scatter_add

Factory functions
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    build_mlp
    build_gated_equivariant_mlp
    replicate_module

schnetpack.transform
====================
.. automodule:: schnetpack.transform

.. currentmodule:: schnetpack.transform
.. autoclass:: Transform

Atomistic
---------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    AddOffsets
    RemoveOffsets
    SubtractCenterOfMass
    SubtractCenterOfGeometry
    NiggliReduce
    WrapPositions

Casting
-------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    CastMap
    CastTo32
    CastTo64

Neighbor lists
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    MatScipyNeighborList
    ASENeighborList
    VesinNeighborList
    TorchNeighborList
    AllToAllNeighborList
    CachedNeighborList
    DistillationNeighborList
    CountNeighbors
    FilterNeighbors
    CollectAtomTriples

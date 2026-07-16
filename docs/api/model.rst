schnetpack.model
================
.. currentmodule:: model

Everything that maps a batch of atoms to predictions. A model is a
:class:`NeuralNetworkPotential` wrapping one *representation*, which turns a
geometry into per-atom features, and a list of *input* and *output modules*,
which read those features.

Every name below is re-exported at ``schnetpack.model``, so ``spk.model.PaiNN``
and ``spk.model.representation.PaiNN`` are the same class.

Models
------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    AtomisticModel
    NeuralNetworkPotential

Representations
---------------
.. rubric:: Message-passing neural networks

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    SchNet
    PaiNN
    SO3net
    FieldSchNet

Input modules
-------------
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    PairwiseDistances
    StaticExternalFields

Output modules
--------------
.. rubric:: Atom-wise layers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Atomwise
    DipoleMoment
    Polarizability
    Aggregation

.. rubric:: Response layers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Forces
    Strain
    Response

.. rubric:: Physical terms

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ZBLRepulsionEnergy
    CoulombPotential
    DampedCoulombPotential
    EnergyCoulomb
    EnergyEwald

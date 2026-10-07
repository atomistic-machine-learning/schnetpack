schnetpack.dynamics
===================
.. currentmodule:: dynamics

Loops that move structures with a model: sampling a generative model down its reverse
process and relaxing structures on a force field. Every driver shares the calculators and
the hooks below, and the calculators carry the guidance; the step rules are subclasses of
the two families' bases.

Drivers
-------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Dynamics


Optimizers
----------

The time-free loop that steps structures along a force until they are relaxed, and its
step rules. A batch-wise relaxation with one inverse Hessian approximation per structure
is :class:`LBFGS`; steepest descent is :class:`GradientDescent`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Optimizer
    GradientDescent
    LBFGS
    LBFGSState
    DirectDenoising


Samplers
--------

The time-indexed loop and its step rules on the reverse process's drift and
diffusion, and the grids its steps land on.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Sampler
    EulerMaruyama
    Heun
    Ancestral
    TimeGrid
    UniformGrid


Calculators
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Calculator
    ForceCalculator
    GenerativeCalculator
    EnsembleCalculator
    NNEnsemble
    BatchNeighborList


Hooks
-----

Edits of the batch before and/or after every step of a driver.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Hook
    FreezeScaffold


Guidance
--------

Terms a calculator adds to the field it returns: to the forces, or to the score.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Guidance
    HarmonicRestraint


Uncertainty
-----------

.. currentmodule:: uncertainty

How far the members of an ensemble disagree, one value per structure. Shared with
:class:`~schnetpack.interfaces.ase_interface.SpkEnsembleCalculator`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Uncertainty
    AbsoluteUncertainty
    RelativeUncertainty

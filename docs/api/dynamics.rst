schnetpack.dynamics
===================
.. currentmodule:: dynamics

Loops that move structures with a model: sampling a generative model down its reverse
process and relaxing structures on a force field. Every driver shares the calculators and
the constraints below; the step rules are subclasses of the two families' bases. The vocabulary is collected in
``CONTEXT.md``, the design of relaxation in ``docs/adr/0001-relaxation-in-dynamics.md``.

Drivers
-------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Dynamics


Time-free drivers
-----------------

The loop that steps structures along a force until they are relaxed, and its step rules.
A batch-wise relaxation with one inverse Hessian approximation per structure is
:class:`LBFGS`; steepest descent is :class:`Langevin` at ``kT = 0``.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Optimizer
    LBFGS
    LBFGSState
    Langevin
    DirectDenoising
    NoiseSchedule
    ConstantNoise
    AnnealedNoise


Samplers
--------

The time-indexed loop and its step rules on the reverse process's drift and
diffusion.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Sampler
    EulerMaruyama
    Heun
    Ancestral
    AncestralDDPM


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


Constraints
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    StateConstraint
    Scaffold


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

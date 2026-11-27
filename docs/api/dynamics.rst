schnetpack.dynamics
===================
.. currentmodule:: schnetpack.dynamics

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
    Update
    SDEUpdate
    ODEUpdate
    AncestralUpdate
    Euler
    Heun
    TimeGrid
    UniformGrid
    KarrasGrid


Calculators
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Calculator
    ForceCalculator
    FieldCalculator
    EnsembleCalculator
    BatchNeighborList


Hooks
-----

Edits of the batch before and/or after every step of a driver, e.g. to hold a part of
every structure fixed (:doc:`/guides/guided_sampling`).

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Hook
    FreezeScaffold


Guidance
--------

Terms added to the field a driver steps on: a
:class:`ForceCalculator` adds them to the model's forces, an :class:`Update` of a
:class:`Sampler` to its score, velocity or denoised structure. Classifier-free guidance
is not a term but an argument of the :class:`FieldCalculator` (``cfg_scale``). See
:doc:`/guides/guidance` and :doc:`/guides/guided_sampling`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Guidance
    HarmonicRestraint
    ClassifierGuidance


Uncertainty
-----------

.. currentmodule:: schnetpack.uncertainty

How far the models of an ensemble disagree, one value per structure. Shared with
:class:`~schnetpack.interfaces.ase_interface.SpkEnsembleCalculator`. The predictions of
:class:`~schnetpack.model.NNEnsemble` are handed over keyed by the names of
:mod:`schnetpack.properties`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Uncertainty
    AbsoluteUncertainty
    RelativeUncertainty

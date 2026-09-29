schnetpack.dynamics
===================
.. currentmodule:: dynamics

Loops that move structures with a model: sampling a generative model down its reverse
process and relaxing structures on a force field. Every driver shares the calculator,
the integrators, the constraints and the observers below. The vocabulary is collected in
``CONTEXT.md``, the design of relaxation in ``docs/adr/0001-relaxation-in-dynamics.md``.

Drivers
-------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Dynamics
    Sampler
    Relaxer
    RelaxationResult
    DirectDenoising


Calculators
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Calculator
    EnsembleCalculator
    NNEnsemble


Integrators
-----------

Reverse-process solvers and relaxation step rules. A batch-wise relaxation with one
inverse Hessian approximation per structure is :class:`LBFGS`, run by a
:class:`Relaxer`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Integrator
    EulerMaruyama
    Heun
    Ancestral
    AncestralDDPM
    LBFGS
    LBFGSState


Constraints
-----------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    StateConstraint
    AnnealedNoise
    Scaffold
    FieldConstraint
    HarmonicBond


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


Observers
---------

.. currentmodule:: dynamics

What a run reports while it runs. See :mod:`schnetpack.dynamics.observers`.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Observer
    Frame
    SamplingFrame
    RelaxationFrame
    Interval
    FrameCollector
    TrajectoryRecorder
    LogWriter


Trajectories
------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TrajectoryWriter
    TrajectoryReader

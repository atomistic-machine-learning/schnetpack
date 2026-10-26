schnetpack.generative
=====================
.. currentmodule:: schnetpack.generative

Diffusion and flow matching as composable building blocks, in pure PyTorch. A model is
a configuration of independent axes: how noisy samples are paired with clean ones, the
schedule, the prior, the coupling, and what the network predicts. Every name below is
re-exported at ``schnetpack.generative``. Running a trained model lives in
:mod:`schnetpack.dynamics`; training and sampling must share the same variables.


Noising
-------

How every noisy sample is paired with a clean structure, each level a special case of
the one before.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ForwardNoising
    Interpolant
    GaussianInterpolant
    PeriodicGaussianInterpolant
    MaskNoising

.. autosummary::
    :toctree: generated
    :nosignatures:

    expand_t


Schedules
---------

How fast a structure moves from the data at t = 0 to the prior: a(t) and b(t) for any
prior, or a(t) and sigma(t) for a Gaussian one.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Schedule
    GaussianSchedule
    FlowMatching
    VP
    VE
    VELinear
    VPCosine


Priors
------

The distribution of the x1 endpoint: what training draws and where sampling starts.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Prior
    GaussianPrior
    MaskPrior
    UniformCellPrior

.. autosummary::
    :toctree: generated
    :nosignatures:

    noise_like


Couplings
---------

How already-drawn endpoint batches are paired.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Coupling
    IdentityCoupling
    PermutationCoupling
    PCVarianceCoupling
    OTCoupling


Parametrizations
----------------

What the network predicts, its training target, and the conversions between the
fields; and the loss weights along the path.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Parametrization
    ScoreParametrization
    EpsParametrization
    X0Parametrization
    VelocityParametrization
    PseudoForceParametrization
    LogitsParametrization
    MaskedTypeWeight
    SNRWeight


Differential equations
----------------------

The (f, g) SDE chart of a Gaussian schedule and its reversal.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    SDE
    ReverseSDE


Time samplers
-------------

Where along the path training draws its times.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TimeSampler
    UniformTimes
    LogNormalSigmaTimes


Variables and views
-------------------

A generated quantity (positions, types, a cell) with its noising, parametrization, head
and loss, and the view of where it lives in the structure. Several variables share one
time.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Variable
    View
    KeyView
    PeriodicPositions
    CellParameters
    ViewInput

.. autosummary::
    :toctree: generated
    :nosignatures:

    weighted_squared_error


Training
--------

The training step: at tensor level for one variable, and as a data-pipeline transform
for a list of them.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    MatchingLoss
    Diffuse

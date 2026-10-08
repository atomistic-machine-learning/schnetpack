schnetpack.train
================
.. currentmodule:: train


Scheduler
---------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ReduceLROnPlateau

Knowledge distillation
----------------------

A frozen teacher's energies, forces and Hessian-vector products, distilled into
a student. :class:`~schnetpack.lightning.AtomisticTask` takes the teacher as
``teacher``; a hand-written loop uses :func:`distillation_predictions` and
:func:`~schnetpack.objectives.calculate_loss` (see
:mod:`schnetpack.train.distillation`).

.. rubric:: Teacher families

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TeacherWrapper
    SchNetPackTeacher
    MaceTeacher

.. rubric:: Student statistics

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TeacherStats

.. rubric:: Functions

.. autosummary::
    :toctree: generated
    :nosignatures:

    teacher_key
    distillation_predictions
    check_distillation_setup
    student_stats_source

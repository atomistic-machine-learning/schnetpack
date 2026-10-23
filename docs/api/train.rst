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

A frozen teacher's energies, forces and curvatures (energy Hessian-vector products),
distilled into a student. :class:`~schnetpack.lightning.AtomisticTask` takes the teacher as
``teacher``; a hand-written loop uses a :class:`Distillation` and
:func:`~schnetpack.objectives.calculate_loss` (see
:mod:`schnetpack.train.distillation`). The teacher's targets are
``properties.teacher_energy``, ``teacher_forces`` and ``teacher_curvature``.

.. rubric:: Teachers

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    TeacherWrapper
    SchNetPackTeacher
    MaceTeacher

.. rubric:: Distilling

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Distillation

schnetpack.train
================
.. currentmodule:: schnetpack.train


Scheduler
---------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ReduceLROnPlateau

Foundation models
-----------------

Fine tuning
~~~~~~~~~~~

A pretrained MACE model, wrapped in a :class:`~schnetpack.model.MaceModel`, trains like
any SchNetPack model: ``spktrain experiment=mace_finetune``. See
:mod:`schnetpack.model.mace` for the reference energies (E0s) the data pipeline removes.

Training from scratch
~~~~~~~~~~~~~~~~~~~~~

A MACE model with random weights, written by
:func:`~schnetpack.interfaces.mace_interface.build_mace` (needs ``schnetpack[mace]``)
with the elements, neighbor count and energy scale of the training data, trains the same
way: ``spktrain experiment=mace_scratch``. Its E0s are estimated on the training split
(``MaceModel(e0s="estimated")``).

Distillation
~~~~~~~~~~~~

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

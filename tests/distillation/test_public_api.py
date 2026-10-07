"""What ``schnetpack.train`` offers for distillation (ADR-0018 §3, ADR-0025):
the teacher families and what a hand-written loop needs; the helpers stay
importable from their modules. ``schnetpack.model`` offers the train-mode
switch (ADR-0021 §3-4) and the module the student prunes the shared neighbor
list with (ADR-0012 §2)."""

import schnetpack.model as model
import schnetpack.train as train

PUBLIC = [
    "TeacherWrapper",
    "SchNetPackTeacher",
    "MaceTeacher",
    "teacher_key",
    "distillation_predictions",
    "check_distillation_setup",
    "TeacherStats",
    "student_stats_source",
]
HELPERS = {
    "teacher": ["MODEL_FORMATS"],
    "distillation": [
        "draw_probe",
        "student_hvp",
        "student_energy_offsets",
        "needs_teacher",
        "needs_curvature",
    ],
}


def test_the_package_offers_the_families_and_the_loop_api():
    missing = [name for name in PUBLIC if not hasattr(train, name)]

    assert missing == []


def test_a_loop_assembles_the_loss_with_calculate_loss():
    """The one composite-loss function is objectives.calculate_loss (ADR-0023);
    distillation has no one-call loss of its own (ADR-0025)."""
    assert not hasattr(train, "compute_distillation_loss")
    assert not hasattr(train.distillation, "compute_distillation_loss")


def test_helpers_stay_in_their_modules():
    exported = [
        name for names in HELPERS.values() for name in names if hasattr(train, name)
    ]
    unreachable = [
        name
        for module, names in HELPERS.items()
        for name in names
        if not hasattr(getattr(train, module), name)
    ]

    assert exported == [] and unreachable == []


def test_the_model_package_offers_the_train_mode_switch():
    """Curvature switches train mode on. Postprocessing is switched off by
    objectives.predict_without_postprocessing (ADR-0024), not here."""
    assert hasattr(model, "train_mode")
    assert not hasattr(model, "set_postprocessing")
    assert not hasattr(model, "train_mode_sensitive_modules")
    assert hasattr(model.utils, "train_mode_sensitive_modules")


def test_the_model_package_offers_the_students_pruning():
    assert hasattr(model, "FilterShortRange")

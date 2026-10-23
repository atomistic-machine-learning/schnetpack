"""What ``schnetpack.train`` offers for distillation: the teacher classes and
the object a hand-written loop needs. ``schnetpack.model`` offers the
train-mode switch and the module the student prunes the shared neighbor list
with."""

import schnetpack.model as model
import schnetpack.train as train

PUBLIC = [
    "TeacherWrapper",
    "SchNetPackTeacher",
    "MaceTeacher",
    "Distillation",
]


def test_the_package_offers_the_teachers_and_the_loop_api():
    missing = [name for name in PUBLIC if not hasattr(train, name)]

    assert missing == []


def test_the_model_package_offers_train_mode_and_pruning():
    assert hasattr(model, "train_mode")
    assert hasattr(model, "FilterShortRange")

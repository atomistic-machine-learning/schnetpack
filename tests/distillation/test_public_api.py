"""What ``schnetpack.model`` offers distillation: the train-mode switch
(ADR-0021 §3-4, ADR-0025) and the module the student prunes the shared neighbor
list with (ADR-0012 §2)."""

import schnetpack.model as model


def test_the_model_package_offers_the_train_mode_switch():
    """Curvature switches train mode on. Postprocessing is switched off by
    objectives.predict_without_postprocessing (ADR-0024), not here."""
    assert hasattr(model, "train_mode")
    assert not hasattr(model, "set_postprocessing")
    assert not hasattr(model, "train_mode_sensitive_modules")
    assert hasattr(model.utils, "train_mode_sensitive_modules")


def test_the_model_package_offers_the_students_pruning():
    assert hasattr(model, "FilterShortRange")

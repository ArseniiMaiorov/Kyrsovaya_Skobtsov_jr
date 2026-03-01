from __future__ import annotations

import numpy as np
import pytest

from src.metrics import metrics as metrics_module
from src.metrics.metrics import (
    MetricsError,
    calculate_multiclass_roc_auc,
    evaluate_multiclass_classification,
    plot_multiclass_roc_curves,
)


def _make_proba() -> np.ndarray:
    return np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
            [0.1, 0.2, 0.7],
        ],
        dtype=float,
    )


def test_calculate_multiclass_roc_auc_success():
    score, present = calculate_multiclass_roc_auc(
        y_true=[0, 1, 2, 0, 1, 2],
        y_proba=_make_proba(),
        labels=(0, 1, 2),
    )

    assert 0 <= score <= 1
    assert present == (0, 1, 2)


def test_calculate_multiclass_roc_auc_partial_classes():
    y_true = [0, 0, 1, 1]
    y_proba = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=float,
    )

    score, present = calculate_multiclass_roc_auc(y_true=y_true, y_proba=y_proba, labels=(0, 1, 2))

    assert score == pytest.approx(1.0)
    assert present == (0, 1)


def test_calculate_multiclass_roc_auc_requires_two_classes():
    y_true = [0, 0, 0]
    y_proba = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="менее двух классов"):
        calculate_multiclass_roc_auc(y_true=y_true, y_proba=y_proba, labels=(0, 1, 2))


def test_calculate_multiclass_roc_auc_validates_distribution():
    with pytest.raises(ValueError, match="диапазоне|Сумма вероятностей"):
        calculate_multiclass_roc_auc(
            y_true=[0, 1, 2],
            y_proba=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=float),
            labels=(0, 1, 2),
        )

    with pytest.raises(ValueError, match="Сумма вероятностей"):
        calculate_multiclass_roc_auc(
            y_true=[0, 1, 2],
            y_proba=np.array([[0.6, 0.2, 0.1], [0.2, 0.5, 0.1], [0.1, 0.2, 0.5]], dtype=float),
            labels=(0, 1, 2),
        )


def test_plot_multiclass_roc_curves_success(tmp_path):
    plot_path = tmp_path / "roc.png"
    result = plot_multiclass_roc_curves(
        y_true=[0, 1, 2, 0, 1, 2],
        y_proba=_make_proba(),
        labels=(0, 1, 2),
        save_path=plot_path,
    )

    assert plot_path.exists()
    assert result["path"] == str(plot_path)
    assert set(result["present_labels"]) == {0, 1, 2}
    assert set(result["per_class_auc"]) == {"0", "1", "2"}


def test_plot_multiclass_roc_curves_with_single_class_still_saves(tmp_path):
    plot_path = tmp_path / "roc_single.png"
    y_proba = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]], dtype=float)
    result = plot_multiclass_roc_curves(
        y_true=[0, 0],
        y_proba=y_proba,
        labels=(0, 1, 2),
        save_path=plot_path,
    )

    assert plot_path.exists()
    assert result["per_class_auc"] == {}


def test_evaluate_multiclass_success_with_proba():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]

    metrics = evaluate_multiclass_classification(y_true, y_pred, labels=(0, 1, 2), y_proba=_make_proba())

    assert 0 <= float(metrics["macro_f1"]) <= 1
    assert metrics["roc_auc_ovr_macro"] is not None
    assert metrics["labels"] == [0, 1, 2]
    assert len(metrics["confusion_matrix"]) == 3


def test_evaluate_multiclass_without_proba():
    metrics = evaluate_multiclass_classification([0, 1, 2], [0, 1, 1], labels=(0, 1, 2), y_proba=None)

    assert metrics["roc_auc_ovr_macro"] is None
    assert "roc_auc_note" in metrics


def test_evaluate_multiclass_partial_roc_auc_note():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_proba = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=float,
    )

    metrics = evaluate_multiclass_classification(y_true, y_pred, labels=(0, 1, 2), y_proba=y_proba)

    assert metrics["roc_auc_ovr_macro"] == pytest.approx(1.0)
    assert "присутствующим классам" in str(metrics["roc_auc_note"])


def test_evaluate_multiclass_roc_auc_unavailable_note():
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    y_proba = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.85, 0.1, 0.05],
            [0.7, 0.2, 0.1],
        ],
        dtype=float,
    )

    metrics = evaluate_multiclass_classification(y_true, y_pred, labels=(0, 1, 2), y_proba=y_proba)

    assert metrics["roc_auc_ovr_macro"] is None
    assert "ROC AUC не вычислен" in str(metrics.get("roc_auc_note"))


def test_evaluate_multiclass_macro_metrics_use_fixed_labels():
    metrics = evaluate_multiclass_classification([0, 0, 0], [0, 0, 0], labels=(0, 1, 2), y_proba=None)

    assert float(metrics["macro_f1"]) == pytest.approx(1.0 / 3.0)
    assert float(metrics["macro_precision"]) == pytest.approx(1.0 / 3.0)
    assert float(metrics["macro_recall"]) == pytest.approx(1.0 / 3.0)
    assert float(metrics["macro_f1"]) == pytest.approx(float(metrics["classification_report"]["macro avg"]["f1-score"]))


def test_evaluate_multiclass_roc_auc_value_error_note():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    y_proba = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )

    metrics = evaluate_multiclass_classification(y_true, y_pred, labels=(0, 1, 2), y_proba=y_proba)

    assert metrics["roc_auc_ovr_macro"] is None
    assert "ROC AUC не вычислен" in str(metrics.get("roc_auc_note"))


def test_evaluate_multiclass_roc_auc_nan_note(monkeypatch):
    def _fake_roc_auc(*args, **kwargs):
        return float("nan"), (0, 1, 2)

    monkeypatch.setattr(metrics_module, "calculate_multiclass_roc_auc", _fake_roc_auc)

    metrics = evaluate_multiclass_classification(
        [0, 1, 2],
        [0, 1, 2],
        labels=(0, 1, 2),
        y_proba=np.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
            ],
            dtype=float,
        ),
    )

    assert metrics["roc_auc_ovr_macro"] is None
    assert "NaN" in str(metrics["roc_auc_note"])


def test_evaluate_multiclass_empty_labels_error():
    with pytest.raises(MetricsError, match="Список меток"):
        evaluate_multiclass_classification([0], [0], labels=(), y_proba=None)


def test_evaluate_multiclass_duplicate_labels_error():
    with pytest.raises(MetricsError, match="дубликаты"):
        evaluate_multiclass_classification([0], [0], labels=(0, 0), y_proba=None)


def test_evaluate_multiclass_y_true_empty_error():
    with pytest.raises(MetricsError, match="не должен быть пустым"):
        evaluate_multiclass_classification([], [], labels=(0, 1, 2), y_proba=None)


def test_evaluate_multiclass_length_mismatch_error():
    with pytest.raises(MetricsError, match="одинаковой длины"):
        evaluate_multiclass_classification([0, 1], [0], labels=(0, 1, 2), y_proba=None)


def test_evaluate_multiclass_proba_ndim_error():
    y_proba = np.array([0.5, 0.5, 0.0])
    with pytest.raises(MetricsError, match="матрицей"):
        evaluate_multiclass_classification([0], [0], labels=(0, 1, 2), y_proba=y_proba)


def test_evaluate_multiclass_y_true_y_pred_ndim_error():
    y_true = np.array([[0], [1]])
    y_pred = np.array([[0], [1]])
    with pytest.raises(MetricsError, match="одномерными"):
        evaluate_multiclass_classification(y_true, y_pred, labels=(0, 1, 2), y_proba=None)


def test_evaluate_multiclass_proba_shape_error():
    y_proba = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(MetricsError, match="Неверная форма y_proba"):
        evaluate_multiclass_classification([0, 1], [0, 1], labels=(0, 1, 2), y_proba=y_proba)


def test_evaluate_multiclass_proba_nan_error():
    y_proba = np.array([[0.5, np.nan, 0.5]])
    with pytest.raises(MetricsError, match="NaN"):
        evaluate_multiclass_classification([0], [0], labels=(0, 1, 2), y_proba=y_proba)

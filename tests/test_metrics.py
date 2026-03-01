from __future__ import annotations

import numpy as np
import pytest

from src.metrics.metrics import MetricsError, evaluate_multiclass_classification


def test_evaluate_multiclass_success_with_proba():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    y_proba = np.array(
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

    metrics = evaluate_multiclass_classification(y_true, y_pred, labels=(0, 1, 2), y_proba=y_proba)

    assert 0 <= float(metrics["macro_f1"]) <= 1
    assert metrics["roc_auc_ovr_macro"] is not None
    assert metrics["labels"] == [0, 1, 2]
    assert len(metrics["confusion_matrix"]) == 3


def test_evaluate_multiclass_without_proba():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 1]

    metrics = evaluate_multiclass_classification(y_true, y_pred, labels=(0, 1, 2), y_proba=None)

    assert metrics["roc_auc_ovr_macro"] is None
    assert "roc_auc_note" in metrics


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


def test_evaluate_multiclass_roc_auc_value_error_note():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    # Каждая строка не нормирована до суммы 1 -> sklearn выбрасывает ValueError.
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

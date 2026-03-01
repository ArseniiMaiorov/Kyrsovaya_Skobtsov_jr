"""Единый расчет метрик для задач классификации."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsError(ValueError):
    """Исключение для ошибок валидации входов метрик."""


def _normalize_labels(labels: Iterable[int]) -> tuple[int, ...]:
    labels_tuple = tuple(int(label) for label in labels)
    if not labels_tuple:
        raise MetricsError("Список меток не должен быть пустым")
    if len(set(labels_tuple)) != len(labels_tuple):
        raise MetricsError("Список меток не должен содержать дубликаты")
    return labels_tuple


def _validate_true_pred(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise MetricsError("y_true и y_pred должны быть одномерными массивами")
    if len(y_true) == 0:
        raise MetricsError("y_true не должен быть пустым")
    if len(y_true) != len(y_pred):
        raise MetricsError("y_true и y_pred должны быть одинаковой длины")


def _validate_proba(y_proba: np.ndarray | None, n_samples: int, n_classes: int) -> None:
    if y_proba is None:
        return

    if y_proba.ndim != 2:
        raise MetricsError("y_proba должен быть матрицей размера (n_samples, n_classes)")

    if y_proba.shape != (n_samples, n_classes):
        raise MetricsError(
            "Неверная форма y_proba: "
            f"ожидалось {(n_samples, n_classes)}, получено {tuple(y_proba.shape)}"
        )

    if np.isnan(y_proba).any() or np.isinf(y_proba).any():
        raise MetricsError("y_proba содержит NaN или бесконечные значения")


def _compute_roc_auc_ovr_macro(
    y_true: np.ndarray,
    y_proba: np.ndarray | None,
    labels: tuple[int, ...],
) -> tuple[float | None, str | None]:
    if y_proba is None:
        return None, "Вероятности классов не переданы"

    try:
        score = roc_auc_score(
            y_true,
            y_proba,
            labels=list(labels),
            multi_class="ovr",
            average="macro",
        )
        score_float = float(score)
        if np.isnan(score_float):
            return None, "ROC AUC не вычислен: получено значение NaN"
        return score_float, None
    except ValueError as exc:
        return None, f"ROC AUC не вычислен: {exc}"


def evaluate_multiclass_classification(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: Iterable[int],
    y_proba: np.ndarray | None = None,
) -> dict[str, object]:
    """Считает единый набор метрик multiclass-классификации."""
    labels_tuple = _normalize_labels(labels)

    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)

    _validate_true_pred(y_true_arr, y_pred_arr)
    _validate_proba(y_proba, n_samples=len(y_true_arr), n_classes=len(labels_tuple))

    conf = confusion_matrix(y_true_arr, y_pred_arr, labels=list(labels_tuple))
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(labels_tuple),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(labels_tuple),
        zero_division=0,
    )

    roc_auc, roc_auc_note = _compute_roc_auc_ovr_macro(
        y_true=y_true_arr,
        y_proba=y_proba,
        labels=labels_tuple,
    )

    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": roc_auc,
        "confusion_matrix": conf.tolist(),
        "labels": list(labels_tuple),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }

    if roc_auc_note is not None:
        result["roc_auc_note"] = roc_auc_note

    return result

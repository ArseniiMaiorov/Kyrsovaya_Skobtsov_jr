"""Единый расчет метрик для задач классификации."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt


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


def _validate_probability_distribution(y_proba: np.ndarray) -> None:
    if (y_proba < 0).any() or (y_proba > 1).any():
        raise ValueError("Значения y_proba должны лежать в диапазоне [0, 1]")

    row_sums = np.sum(y_proba, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Сумма вероятностей по классам в каждой строке должна быть равна 1")


def calculate_multiclass_roc_auc(
    y_true: Iterable[int] | np.ndarray,
    y_proba: np.ndarray,
    labels: Iterable[int],
) -> tuple[float, tuple[int, ...]]:
    """Вычисляет macro ROC AUC в режиме One-vs-Rest по присутствующим классам."""
    labels_tuple = _normalize_labels(labels)
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    _validate_proba(y_proba, n_samples=len(y_true_arr), n_classes=len(labels_tuple))
    _validate_probability_distribution(y_proba)

    present_labels = tuple(label for label in labels_tuple if label in set(y_true_arr.tolist()))
    if len(present_labels) < 2:
        raise ValueError("ROC AUC не вычисляется, если в y_true присутствует менее двух классов")

    per_class_auc: list[float] = []
    label_to_index = {label: idx for idx, label in enumerate(labels_tuple)}
    for label in present_labels:
        binary_true = (y_true_arr == label).astype(np.int64)
        class_scores = np.asarray(y_proba[:, label_to_index[label]], dtype=np.float64)
        per_class_auc.append(float(roc_auc_score(binary_true, class_scores)))

    return float(np.mean(per_class_auc)), present_labels


def plot_multiclass_roc_curves(
    y_true: Iterable[int] | np.ndarray,
    y_proba: np.ndarray,
    labels: Iterable[int],
    save_path: str | Path,
) -> dict[str, object]:
    """Строит ROC-кривые OvR для присутствующих классов и сохраняет их в файл."""
    labels_tuple = _normalize_labels(labels)
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    _validate_proba(y_proba, n_samples=len(y_true_arr), n_classes=len(labels_tuple))
    _validate_probability_distribution(y_proba)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    present_labels = tuple(label for label in labels_tuple if label in set(y_true_arr.tolist()))
    label_to_index = {label: idx for idx, label in enumerate(labels_tuple)}
    per_class_auc: dict[str, float] = {}

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for label in present_labels:
        binary_true = (y_true_arr == label).astype(np.int64)
        if len(np.unique(binary_true)) < 2:
            continue
        class_scores = np.asarray(y_proba[:, label_to_index[label]], dtype=np.float64)
        fpr, tpr, _ = roc_curve(binary_true, class_scores)
        roc_auc_value = float(auc(fpr, tpr))
        per_class_auc[str(label)] = roc_auc_value
        ax.plot(fpr, tpr, linewidth=1.8, label=f"Класс {label} (AUC={roc_auc_value:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Случайное угадывание")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые One-vs-Rest")
    ax.grid(alpha=0.25)
    if per_class_auc:
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "ROC-кривые недоступны:\nменее двух классов в y_true", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": str(output_path),
        "present_labels": list(int(label) for label in present_labels),
        "per_class_auc": per_class_auc,
    }


def _compute_roc_auc_ovr_macro(
    y_true: np.ndarray,
    y_proba: np.ndarray | None,
    labels: tuple[int, ...],
) -> tuple[float | None, str | None]:
    if y_proba is None:
        return None, "Вероятности классов не переданы"

    try:
        score, present_labels = calculate_multiclass_roc_auc(
            y_true=y_true,
            y_proba=y_proba,
            labels=labels,
        )
        score_float = float(score)
        if np.isnan(score_float):
            return None, "ROC AUC не вычислен: получено значение NaN"
        if len(present_labels) != len(labels):
            present_text = ", ".join(str(label) for label in present_labels)
            return score_float, (
                "ROC AUC рассчитан по присутствующим классам "
                f"({present_text}); отсутствующие классы исключены из усреднения"
            )
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
        "macro_precision": float(
            precision_score(y_true_arr, y_pred_arr, labels=list(labels_tuple), average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true_arr, y_pred_arr, labels=list(labels_tuple), average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=list(labels_tuple), average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true_arr, y_pred_arr, labels=list(labels_tuple), average="weighted", zero_division=0)
        ),
        "roc_auc_ovr_macro": roc_auc,
        "confusion_matrix": conf.tolist(),
        "labels": list(labels_tuple),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }

    if roc_auc_note is not None:
        result["roc_auc_note"] = roc_auc_note

    return result

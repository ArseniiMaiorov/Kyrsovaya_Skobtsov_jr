"""Baseline-модели и единый запуск эксперимента."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.metrics.metrics import evaluate_multiclass_classification

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class BaselineModelError(ValueError):
    """Исключение для ошибок обучения и запуска baseline-моделей."""


def _validate_random_state(random_state: int) -> None:
    if not isinstance(random_state, int) or random_state < 0:
        raise BaselineModelError("random_state должен быть неотрицательным целым числом")


def build_baseline_model(
    model_name: str,
    random_state: int,
    class_weight: str | dict[int, float] | None = "balanced",
) -> BaseEstimator:
    """Создает baseline-модель по имени."""
    _validate_random_state(random_state)

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
            solver="saga",
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    raise BaselineModelError(
        "Неподдерживаемая baseline-модель: "
        f"{model_name}. Доступно: logistic_regression, random_forest"
    )


def _validate_training_arrays(x_train: np.ndarray, y_train: np.ndarray) -> None:
    if x_train.ndim != 2:
        raise BaselineModelError("x_train должен быть двумерным массивом")
    if y_train.ndim != 1:
        raise BaselineModelError("y_train должен быть одномерным массивом")
    if x_train.shape[0] == 0:
        raise BaselineModelError("x_train не должен быть пустым")
    if x_train.shape[0] != y_train.shape[0]:
        raise BaselineModelError("Количество строк x_train должно совпадать с длиной y_train")


def fit_baseline_model(model: BaseEstimator, x_train: np.ndarray, y_train: np.ndarray) -> BaseEstimator:
    """Обучает baseline-модель."""
    _validate_training_arrays(x_train, y_train)
    model.fit(x_train, y_train)
    return model


def _decision_to_probability(decision_values: np.ndarray) -> np.ndarray:
    if decision_values.ndim == 1:
        probs_pos = 1.0 / (1.0 + np.exp(-decision_values))
        probs = np.column_stack([1.0 - probs_pos, probs_pos])
        return probs.astype(np.float64)

    if decision_values.ndim == 2:
        shifted = decision_values - np.max(decision_values, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        sums = np.sum(exp_scores, axis=1, keepdims=True)
        probs = exp_scores / sums
        return probs.astype(np.float64)

    raise BaselineModelError("decision_function вернул массив с неподдерживаемой размерностью")


def _align_probabilities_to_labels(
    y_proba: np.ndarray,
    model_classes: np.ndarray,
    labels: tuple[int, ...],
) -> np.ndarray:
    if y_proba.ndim != 2:
        raise BaselineModelError("y_proba должен быть двумерной матрицей")
    if y_proba.shape[1] != len(model_classes):
        raise BaselineModelError("Число столбцов y_proba должно совпадать с числом классов модели")

    aligned = np.zeros((y_proba.shape[0], len(labels)), dtype=np.float64)
    label_to_index = {int(label): idx for idx, label in enumerate(labels)}
    for class_idx, class_label in enumerate(model_classes):
        class_value = int(class_label)
        if class_value not in label_to_index:
            raise BaselineModelError(f"Модель вернула класс вне списка labels: {class_value}")
        aligned[:, label_to_index[class_value]] = y_proba[:, class_idx]
    return aligned


def predict_with_optional_proba(
    model: BaseEstimator,
    x_eval: np.ndarray,
    labels: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Возвращает предсказания классов и вероятности (если доступны)."""
    if x_eval.ndim != 2:
        raise BaselineModelError("x_eval должен быть двумерным массивом")
    if x_eval.shape[0] == 0:
        raise BaselineModelError("x_eval не должен быть пустым")

    y_pred = np.asarray(model.predict(x_eval), dtype=np.int64)

    y_proba: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        y_proba = np.asarray(model.predict_proba(x_eval), dtype=np.float64)
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x_eval), dtype=np.float64)
        y_proba = _decision_to_probability(decision)

    if y_proba is not None and labels is not None and hasattr(model, "classes_"):
        labels_tuple = tuple(int(label) for label in labels)
        y_proba = _align_probabilities_to_labels(
            y_proba=y_proba,
            model_classes=np.asarray(getattr(model, "classes_")),
            labels=labels_tuple,
        )

    return y_pred, y_proba


def analyze_baseline_feature_importance(
    model: BaseEstimator,
    feature_names: Iterable[str],
    labels: Iterable[int],
    save_path: str | Path,
    top_k: int = 10,
) -> dict[str, list[dict[str, float | str]]]:
    """Анализирует коэффициенты логистической регрессии и сохраняет график top-k признаков."""
    if not hasattr(model, "coef_"):
        raise BaselineModelError("Анализ важности признаков поддерживается только для моделей с coef_")

    feature_names_tuple = tuple(str(name) for name in feature_names)
    if not feature_names_tuple:
        raise BaselineModelError("feature_names не должен быть пустым")
    if len(feature_names_tuple) != int(np.asarray(model.coef_).shape[1]):
        raise BaselineModelError("Число feature_names должно совпадать с числом коэффициентов модели")
    if not isinstance(top_k, int) or top_k <= 0:
        raise BaselineModelError("top_k должен быть положительным целым числом")

    labels_tuple = tuple(int(label) for label in labels)
    if not labels_tuple:
        raise BaselineModelError("labels не должен быть пустым")

    coefficients = np.asarray(model.coef_, dtype=np.float64)
    model_classes = tuple(int(label) for label in getattr(model, "classes_", labels_tuple))

    aligned = np.zeros((len(labels_tuple), coefficients.shape[1]), dtype=np.float64)
    if coefficients.shape[0] == 1 and len(model_classes) == 2:
        class_to_row = {
            model_classes[0]: -coefficients[0],
            model_classes[1]: coefficients[0],
        }
        for row_idx, label in enumerate(labels_tuple):
            if label in class_to_row:
                aligned[row_idx] = class_to_row[label]
    else:
        for row_idx, class_label in enumerate(model_classes):
            if class_label in labels_tuple:
                aligned[labels_tuple.index(class_label)] = coefficients[row_idx]

    result: dict[str, list[dict[str, float | str]]] = {}
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(labels_tuple), 1, figsize=(12.0, max(3.8, 2.8 * len(labels_tuple))))
    if len(labels_tuple) == 1:
        axes = [axes]

    for row_idx, label in enumerate(labels_tuple):
        row = aligned[row_idx]
        take = min(top_k, row.shape[0])
        top_indices = np.argsort(np.abs(row))[-take:][::-1]
        entries = [
            {
                "feature": feature_names_tuple[int(idx)],
                "coef": float(row[int(idx)]),
                "abs_coef": float(abs(row[int(idx)])),
            }
            for idx in top_indices
        ]
        result[f"class_{label}"] = entries

        ax = axes[row_idx]
        names = [str(item["feature"]) for item in entries][::-1]
        values = [float(item["coef"]) for item in entries][::-1]
        colors = ["#2A9D8F" if value >= 0 else "#D62828" for value in values]
        ax.barh(names, values, color=colors)
        ax.set_title(f"Топ-{take} признаков для класса {label}")
        ax.set_xlabel("Коэффициент логистической регрессии")
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return result


def run_baseline_experiment(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    labels: Iterable[int],
    random_state: int,
    class_weight: str | dict[int, float] | None = "balanced",
    return_model: bool = False,
) -> dict[str, Any]:
    """Обучает baseline-модель и считает метрики на eval-выборке."""
    labels_tuple = tuple(int(label) for label in labels)
    if not labels_tuple:
        raise BaselineModelError("labels не должен быть пустым")

    model = build_baseline_model(model_name, random_state=random_state, class_weight=class_weight)
    model = fit_baseline_model(model, x_train=x_train, y_train=y_train)

    y_pred, y_proba = predict_with_optional_proba(model, x_eval=x_eval, labels=labels_tuple)
    metrics = evaluate_multiclass_classification(
        y_true=y_eval,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels_tuple,
    )

    result = {
        "model_name": model_name,
        "train_size": int(x_train.shape[0]),
        "eval_size": int(x_eval.shape[0]),
        "metrics": metrics,
    }
    if return_model:
        result["model"] = model
        result["y_pred"] = y_pred
        result["y_proba"] = y_proba
    return result

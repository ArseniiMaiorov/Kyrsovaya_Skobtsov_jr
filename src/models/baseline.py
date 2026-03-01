"""Baseline-модели и единый запуск эксперимента."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.metrics.metrics import evaluate_multiclass_classification


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


def predict_with_optional_proba(model: BaseEstimator, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
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

    return y_pred, y_proba


def run_baseline_experiment(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    labels: Iterable[int],
    random_state: int,
    class_weight: str | dict[int, float] | None = "balanced",
) -> dict[str, Any]:
    """Обучает baseline-модель и считает метрики на eval-выборке."""
    labels_tuple = tuple(int(label) for label in labels)
    if not labels_tuple:
        raise BaselineModelError("labels не должен быть пустым")

    model = build_baseline_model(model_name, random_state=random_state, class_weight=class_weight)
    model = fit_baseline_model(model, x_train=x_train, y_train=y_train)

    y_pred, y_proba = predict_with_optional_proba(model, x_eval=x_eval)
    metrics = evaluate_multiclass_classification(
        y_true=y_eval,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels_tuple,
    )

    return {
        "model_name": model_name,
        "train_size": int(x_train.shape[0]),
        "eval_size": int(x_eval.shape[0]),
        "metrics": metrics,
    }

"""Обучение и оценка гибридной модели на window-based данных."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
import numpy as np
import tensorflow as tf

from src.metrics.metrics import evaluate_multiclass_classification
from src.models.hybrid import HybridModelError, build_hybrid_classifier, compile_hybrid_classifier

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class HybridTrainingError(ValueError):
    """Исключение для ошибок обучения и оценки гибридной модели."""


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise HybridTrainingError(f"{field_name} должен быть положительным целым числом")
    return int(value)


def _validate_window_inputs(x_data: np.ndarray, y_data: np.ndarray, prefix: str) -> None:
    if x_data.ndim != 3:
        raise HybridTrainingError(f"{prefix} должен иметь размерность (n_samples, T, n_features)")
    if y_data.ndim != 1:
        raise HybridTrainingError(f"{prefix.replace('x_', 'y_')} должен быть одномерным вектором")
    if x_data.shape[0] == 0:
        raise HybridTrainingError(f"{prefix} не должен быть пустым")
    if x_data.shape[0] != y_data.shape[0]:
        raise HybridTrainingError(f"Количество объектов в {prefix} и соответствующем y должно совпадать")


def _validate_train_val_shapes(x_train: np.ndarray, x_val: np.ndarray) -> None:
    if x_train.shape[1:] != x_val.shape[1:]:
        raise HybridTrainingError("Формы окон train и val должны совпадать по T и числу признаков")


def build_training_callbacks(training_cfg: Mapping[str, Any]) -> list[tf.keras.callbacks.Callback]:
    """Создаёт обязательные callbacks по ТЗ."""
    if not isinstance(training_cfg, Mapping):
        raise HybridTrainingError("training_cfg должен быть словарем")

    early_stopping_patience = _require_positive_int(
        training_cfg.get("early_stopping_patience"),
        "early_stopping_patience",
    )
    reduce_lr_patience = _require_positive_int(
        training_cfg.get("reduce_lr_patience"),
        "reduce_lr_patience",
    )

    reduce_lr_factor = training_cfg.get("reduce_lr_factor")
    if not isinstance(reduce_lr_factor, (int, float)) or not 0 < reduce_lr_factor < 1:
        raise HybridTrainingError("reduce_lr_factor должен быть числом в диапазоне (0, 1)")

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=reduce_lr_patience,
            factor=float(reduce_lr_factor),
        ),
    ]


def compute_balanced_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Вычисляет class_weight по формуле balanced только для присутствующих классов."""
    if y_train.ndim != 1:
        raise HybridTrainingError("y_train должен быть одномерным вектором")
    if len(y_train) == 0:
        raise HybridTrainingError("y_train не должен быть пустым")

    labels, counts = np.unique(y_train, return_counts=True)
    n_samples = int(len(y_train))
    n_classes = int(len(labels))

    return {
        int(label): float(n_samples / (n_classes * count))
        for label, count in zip(labels, counts, strict=True)
    }


def train_hybrid_classifier(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    max_epochs: int,
    callbacks: list[tf.keras.callbacks.Callback],
    class_weight: Mapping[int, float] | None = None,
) -> tf.keras.callbacks.History:
    """Обучает гибридную модель на train с контролем по val."""
    if not isinstance(model, tf.keras.Model):
        raise HybridTrainingError("model должен быть экземпляром tf.keras.Model")

    _validate_window_inputs(x_train, y_train, "x_train")
    _validate_window_inputs(x_val, y_val, "x_val")
    _validate_train_val_shapes(x_train, x_val)

    batch = min(_require_positive_int(batch_size, "batch_size"), int(x_train.shape[0]))
    epochs = _require_positive_int(max_epochs, "max_epochs")

    if not isinstance(callbacks, list):
        raise HybridTrainingError("callbacks должен быть списком")
    if class_weight is not None and not isinstance(class_weight, Mapping):
        raise HybridTrainingError("class_weight должен быть словарем или None")

    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=callbacks,
        class_weight=dict(class_weight) if class_weight is not None else None,
        shuffle=False,
        verbose=0,
    )


def history_to_serializable_dict(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    """Преобразует Keras History в JSON-совместимый словарь."""
    if not isinstance(history, tf.keras.callbacks.History):
        raise HybridTrainingError("history должен быть объектом tf.keras.callbacks.History")

    payload: dict[str, list[float]] = {}
    for key, values in history.history.items():
        payload[str(key)] = [float(value) for value in values]
    return payload


def plot_training_curves(history: tf.keras.callbacks.History | Mapping[str, list[float]], save_path: str | Path) -> str:
    """Строит кривые обучения по loss и accuracy."""
    if isinstance(history, tf.keras.callbacks.History):
        history_dict = history_to_serializable_dict(history)
    elif isinstance(history, Mapping):
        history_dict = {str(key): [float(value) for value in values] for key, values in history.items()}
    else:
        raise HybridTrainingError("history должен быть History или словарем")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    axes[0].plot(history_dict.get("loss", []), label="Train", linewidth=1.8)
    axes[0].plot(history_dict.get("val_loss", []), label="Validation", linewidth=1.8)
    axes[0].set_title("Loss по эпохам")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    if "accuracy" in history_dict and "val_accuracy" in history_dict:
        axes[1].plot(history_dict["accuracy"], label="Train", linewidth=1.8)
        axes[1].plot(history_dict["val_accuracy"], label="Validation", linewidth=1.8)
        axes[1].set_title("Accuracy по эпохам")
        axes[1].set_xlabel("Эпоха")
        axes[1].set_ylabel("Accuracy")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "Accuracy-история недоступна", ha="center", va="center")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best") if axes[1].has_data() else None

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def save_history_artifacts(
    history: tf.keras.callbacks.History,
    output_dir: str | Path,
) -> dict[str, str]:
    """Сохраняет полную историю обучения и графики."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history_dict = history_to_serializable_dict(history)
    history_json_path = output_path / "training_history.json"
    history_json_path.write_text(json.dumps(history_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    curves_path = output_path / "training_curves.png"
    plot_training_curves(history_dict, curves_path)

    return {
        "history_json": str(history_json_path),
        "training_curves": str(curves_path),
    }


def summarize_history(history: tf.keras.callbacks.History) -> dict[str, Any]:
    """Возвращает краткую сводку по истории обучения."""
    if not isinstance(history, tf.keras.callbacks.History):
        raise HybridTrainingError("history должен быть объектом tf.keras.callbacks.History")

    loss_values = list(history.history.get("loss", []))
    val_loss_values = list(history.history.get("val_loss", []))
    if not loss_values or not val_loss_values:
        raise HybridTrainingError("history не содержит loss и val_loss")

    best_epoch_idx = min(range(len(val_loss_values)), key=lambda idx: float(val_loss_values[idx]))
    return {
        "epochs_ran": int(len(loss_values)),
        "best_epoch": int(best_epoch_idx + 1),
        "best_val_loss": float(val_loss_values[best_epoch_idx]),
        "final_train_loss": float(loss_values[-1]),
        "final_val_loss": float(val_loss_values[-1]),
    }


def predict_hybrid_probabilities(model: tf.keras.Model, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает предсказанные классы и вероятности гибридной модели."""
    if not isinstance(model, tf.keras.Model):
        raise HybridTrainingError("model должен быть экземпляром tf.keras.Model")
    if x_eval.ndim != 3:
        raise HybridTrainingError("x_eval должен иметь размерность (n_samples, T, n_features)")
    if x_eval.shape[0] == 0:
        raise HybridTrainingError("x_eval не должен быть пустым")

    y_proba = np.asarray(model.predict(x_eval, verbose=0), dtype=np.float64)
    y_pred = np.argmax(y_proba, axis=1).astype(np.int64)
    return y_pred, y_proba


def evaluate_hybrid_classifier(
    model: tf.keras.Model,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    labels: Iterable[int],
) -> dict[str, object]:
    """Считает метрики классификации на eval-выборке."""
    if not isinstance(model, tf.keras.Model):
        raise HybridTrainingError("model должен быть экземпляром tf.keras.Model")

    labels_tuple = tuple(int(label) for label in labels)
    if not labels_tuple:
        raise HybridTrainingError("labels не должен быть пустым")

    _validate_window_inputs(x_eval, y_eval, "x_eval")

    y_pred, y_proba = predict_hybrid_probabilities(model, x_eval)

    return evaluate_multiclass_classification(
        y_true=y_eval,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels_tuple,
    )


def visualize_conv_filters(model: tf.keras.Model, save_path: str | Path, max_filters: int = 8) -> str:
    """Сохраняет визуализацию весов первого Conv1D-слоя."""
    if not isinstance(model, tf.keras.Model):
        raise HybridTrainingError("model должен быть экземпляром tf.keras.Model")
    try:
        conv_layer = model.get_layer("conv1d_1")
    except ValueError as exc:
        raise HybridTrainingError("В модели отсутствует слой conv1d_1") from exc

    weights = conv_layer.get_weights()
    if not weights:
        raise HybridTrainingError("Слой conv1d_1 не содержит весов")

    kernel = np.asarray(weights[0], dtype=np.float64)
    n_filters = min(int(max_filters), int(kernel.shape[2]))
    n_cols = min(4, n_filters)
    n_rows = int(np.ceil(n_filters / n_cols))

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.8 * n_rows))
    axes_arr = np.atleast_1d(axes).reshape(n_rows, n_cols)
    for plot_idx in range(n_rows * n_cols):
        ax = axes_arr[plot_idx // n_cols, plot_idx % n_cols]
        if plot_idx >= n_filters:
            ax.axis("off")
            continue
        ax.imshow(kernel[:, :, plot_idx].T, aspect="auto", cmap="viridis")
        ax.set_title(f"Фильтр {plot_idx + 1}")
        ax.set_xlabel("Временной шаг ядра")
        ax.set_ylabel("Признак")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _project_to_2d(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64)
    first = u[:, 0] * s[0]
    if len(s) > 1:
        second = u[:, 1] * s[1]
    else:
        second = np.zeros_like(first)
    return np.column_stack([first, second]).astype(np.float64)


def visualize_hidden_representations(
    model: tf.keras.Model,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    save_path: str | Path,
) -> str:
    """Визуализирует скрытые представления последнего рекуррентного слоя."""
    if not isinstance(model, tf.keras.Model):
        raise HybridTrainingError("model должен быть экземпляром tf.keras.Model")
    _validate_window_inputs(x_eval, y_eval, "x_eval")

    recurrent_layers = [
        layer
        for layer in model.layers
        if isinstance(layer, (tf.keras.layers.GRU, tf.keras.layers.LSTM, tf.keras.layers.Bidirectional))
    ]
    if not recurrent_layers:
        raise HybridTrainingError("В модели отсутствуют рекуррентные слои")

    extractor = tf.keras.Model(inputs=model.input, outputs=recurrent_layers[-1].output)
    hidden = np.asarray(extractor.predict(x_eval, verbose=0), dtype=np.float64)
    hidden_flat = hidden.reshape(hidden.shape[0], -1)
    coords = _project_to_2d(hidden_flat)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for class_id in sorted({int(value) for value in y_eval.tolist()}):
        mask = y_eval == class_id
        ax.scatter(coords[mask, 0], coords[mask, 1], alpha=0.75, label=f"Класс {class_id}")
    ax.set_title("2D-проекция скрытых представлений рекуррентного блока")
    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_attention_weights(
    model: tf.keras.Model,
    sample_window: np.ndarray,
    save_path: str | Path,
) -> str:
    """Сохраняет веса внимания для одного окна, если модель содержит attention."""
    if not isinstance(model, tf.keras.Model):
        raise HybridTrainingError("model должен быть экземпляром tf.keras.Model")
    try:
        attention_layer = model.get_layer("attention")
    except ValueError as exc:
        raise HybridTrainingError("В модели отсутствует слой attention") from exc

    if sample_window.ndim != 2:
        raise HybridTrainingError("sample_window должен иметь форму (T, n_features)")

    pre_attention_model = tf.keras.Model(inputs=model.input, outputs=attention_layer.input)
    sequence_output = np.asarray(pre_attention_model.predict(sample_window[np.newaxis, ...], verbose=0), dtype=np.float32)
    scores = attention_layer.score(attention_layer.proj(sequence_output))
    weights = tf.nn.softmax(scores, axis=1).numpy().reshape(-1)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.plot(np.arange(len(weights)), weights, color="#1D3557", linewidth=1.8)
    ax.set_title("Веса внимания по временным шагам")
    ax.set_xlabel("Временной шаг")
    ax.set_ylabel("Attention weight")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def run_hybrid_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: Iterable[int],
    hybrid_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Полный цикл: сборка, компиляция, обучение и оценка гибридной модели."""
    labels_tuple = tuple(int(label) for label in labels)
    if not labels_tuple:
        raise HybridTrainingError("labels не должен быть пустым")

    try:
        model = build_hybrid_classifier(
            hybrid_cfg=hybrid_cfg,
            input_shape=(int(x_train.shape[1]), int(x_train.shape[2])),
            n_classes=len(labels_tuple),
        )
        model = compile_hybrid_classifier(model, hybrid_cfg)
    except HybridModelError as exc:
        raise HybridTrainingError(str(exc)) from exc

    callbacks = build_training_callbacks(training_cfg)
    class_weight = compute_balanced_class_weights(y_train)

    history = train_hybrid_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=int(hybrid_cfg["batch_size"]),
        max_epochs=int(hybrid_cfg["max_epochs"]),
        callbacks=callbacks,
        class_weight=class_weight,
    )

    history_summary = summarize_history(history)
    history_series = history_to_serializable_dict(history)
    artifacts: dict[str, str] = {}
    if artifacts_dir is not None:
        artifacts = save_history_artifacts(history, artifacts_dir)

    y_pred, y_proba = predict_hybrid_probabilities(model, x_val)
    metrics = evaluate_multiclass_classification(
        y_true=y_val,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels_tuple,
    )

    return {
        "model": model,
        "class_weight": class_weight,
        "history": history_summary,
        "history_series": history_series,
        "artifacts": artifacts,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

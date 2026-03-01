"""Предобучение автоэнкодера и fine-tuning классификатора после переноса encoder-весов."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import tensorflow as tf

from src.models.autoencoder import (
    AutoencoderModelError,
    apply_encoder_weights,
    build_hybrid_autoencoder,
)
from src.models.hybrid import HybridModelError, build_hybrid_classifier, compile_hybrid_classifier
from src.training.hybrid_training import (
    HybridTrainingError,
    build_training_callbacks,
    compute_balanced_class_weights,
    evaluate_hybrid_classifier,
    summarize_history,
    train_hybrid_classifier,
)


class AutoencoderTrainingError(ValueError):
    """Исключение для ошибок pretrain/fine-tuning контура."""


def _validate_unlabeled_windows(x_unlabeled: np.ndarray, field_name: str, min_windows: int = 1) -> None:
    if not isinstance(x_unlabeled, np.ndarray):
        raise AutoencoderTrainingError(f"{field_name} должен быть numpy.ndarray")
    if x_unlabeled.ndim != 3:
        raise AutoencoderTrainingError(f"{field_name} должен иметь размерность (n_samples, T, n_features)")
    if x_unlabeled.shape[0] < int(min_windows):
        raise AutoencoderTrainingError(f"{field_name} должен содержать минимум {int(min_windows)} окна")


def split_unlabeled_windows_for_pretrain(
    x_unlabeled: np.ndarray,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Делит неразмеченные окна по времени на train/val для AE без перемешивания."""
    _validate_unlabeled_windows(x_unlabeled, "x_unlabeled", min_windows=2)
    if not isinstance(val_ratio, (int, float)) or not 0 < val_ratio < 1:
        raise AutoencoderTrainingError("val_ratio должен быть числом в диапазоне (0, 1)")

    n_windows = int(x_unlabeled.shape[0])
    val_count = max(1, int(np.floor(n_windows * float(val_ratio))))
    val_count = min(val_count, n_windows - 1)
    split_idx = n_windows - val_count

    x_train = np.asarray(x_unlabeled[:split_idx], dtype=np.float32)
    x_val = np.asarray(x_unlabeled[split_idx:], dtype=np.float32)
    return x_train, x_val


def train_autoencoder(
    model: tf.keras.Model,
    x_train: np.ndarray,
    x_val: np.ndarray,
    batch_size: int,
    max_epochs: int,
    callbacks: list[tf.keras.callbacks.Callback],
) -> tf.keras.callbacks.History:
    """Обучает AE на задаче реконструкции X -> X."""
    if not isinstance(model, tf.keras.Model):
        raise AutoencoderTrainingError("model должен быть экземпляром tf.keras.Model")

    _validate_unlabeled_windows(x_train, "x_train", min_windows=1)
    _validate_unlabeled_windows(x_val, "x_val", min_windows=1)
    if x_train.shape[1:] != x_val.shape[1:]:
        raise AutoencoderTrainingError("Формы окон x_train и x_val должны совпадать")
    if not isinstance(callbacks, list):
        raise AutoencoderTrainingError("callbacks должен быть списком")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise AutoencoderTrainingError("batch_size должен быть положительным целым числом")
    if not isinstance(max_epochs, int) or max_epochs <= 0:
        raise AutoencoderTrainingError("max_epochs должен быть положительным целым числом")

    effective_batch_size = min(int(batch_size), int(x_train.shape[0]))
    return model.fit(
        x_train,
        x_train,
        validation_data=(x_val, x_val),
        epochs=int(max_epochs),
        batch_size=effective_batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=0,
    )


def evaluate_reconstruction(model: tf.keras.Model, x_eval: np.ndarray) -> dict[str, float]:
    """Считает качество реконструкции AE на eval-окнах."""
    if not isinstance(model, tf.keras.Model):
        raise AutoencoderTrainingError("model должен быть экземпляром tf.keras.Model")

    _validate_unlabeled_windows(x_eval, "x_eval", min_windows=1)
    reconstructed = np.asarray(model.predict(x_eval, verbose=0), dtype=np.float32)
    mse_per_window = np.mean(np.square(reconstructed - x_eval), axis=(1, 2))
    return {
        "mean_reconstruction_mse": float(np.mean(mse_per_window)),
        "max_reconstruction_mse": float(np.max(mse_per_window)),
        "min_reconstruction_mse": float(np.min(mse_per_window)),
    }


def run_autoencoder_pretraining(
    x_unlabeled: np.ndarray,
    hybrid_cfg: Mapping[str, Any],
    autoencoder_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Полный цикл AE pretrain на неразмеченных окнах."""
    if not isinstance(autoencoder_cfg, Mapping):
        raise AutoencoderTrainingError("autoencoder_cfg должен быть словарем")

    x_pretrain_train, x_pretrain_val = split_unlabeled_windows_for_pretrain(
        x_unlabeled=x_unlabeled,
        val_ratio=float(autoencoder_cfg["pretrain_val_ratio"]),
    )

    try:
        model = build_hybrid_autoencoder(
            hybrid_cfg=hybrid_cfg,
            input_shape=(int(x_pretrain_train.shape[1]), int(x_pretrain_train.shape[2])),
        )
    except AutoencoderModelError as exc:
        raise AutoencoderTrainingError(str(exc)) from exc

    callbacks = build_training_callbacks(training_cfg)
    history = train_autoencoder(
        model=model,
        x_train=x_pretrain_train,
        x_val=x_pretrain_val,
        batch_size=int(autoencoder_cfg["batch_size"]),
        max_epochs=int(autoencoder_cfg["pretrain_max_epochs"]),
        callbacks=callbacks,
    )
    history_summary = summarize_history(history)
    reconstruction_metrics = evaluate_reconstruction(model, x_pretrain_val)

    return {
        "model": model,
        "history": history_summary,
        "reconstruction_metrics": reconstruction_metrics,
        "split_summary": {
            "train_windows": int(x_pretrain_train.shape[0]),
            "val_windows": int(x_pretrain_val.shape[0]),
        },
    }


def run_pretrained_hybrid_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: Iterable[int],
    hybrid_cfg: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
    encoder_weights: Mapping[str, list[np.ndarray]],
) -> dict[str, Any]:
    """Собирает классификатор, переносит encoder-веса и дообучает на размеченных данных."""
    labels_tuple = tuple(int(label) for label in labels)
    if not labels_tuple:
        raise AutoencoderTrainingError("labels не должен быть пустым")

    try:
        model = build_hybrid_classifier(
            hybrid_cfg=hybrid_cfg,
            input_shape=(int(x_train.shape[1]), int(x_train.shape[2])),
            n_classes=len(labels_tuple),
        )
        model = compile_hybrid_classifier(model, hybrid_cfg)
        transferred_layers = apply_encoder_weights(model, encoder_weights)
    except (HybridModelError, AutoencoderModelError) as exc:
        raise AutoencoderTrainingError(str(exc)) from exc

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
    metrics = evaluate_hybrid_classifier(
        model=model,
        x_eval=x_val,
        y_eval=y_val,
        labels=labels_tuple,
    )

    return {
        "model": model,
        "class_weight": class_weight,
        "history": history_summary,
        "metrics": metrics,
        "transferred_layers": transferred_layers,
    }

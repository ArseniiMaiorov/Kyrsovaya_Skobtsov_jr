"""Сборка автоэнкодера на базе encoder-части гибридного классификатора."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import tensorflow as tf

from src.models.hybrid import HybridModelError, create_optimizer, validate_hybrid_config

ENCODER_WEIGHT_PREFIXES = ("conv1d_", "batch_norm_", "gru_")


class AutoencoderModelError(ValueError):
    """Исключение для ошибок сборки автоэнкодера и переноса весов."""


def validate_autoencoder_config(hybrid_cfg: Mapping[str, Any], input_shape: tuple[int, int]) -> dict[str, Any]:
    """Проверяет, что конфиг гибридной модели пригоден для сборки AE-энкодера."""
    try:
        return validate_hybrid_config(hybrid_cfg, input_shape=input_shape, n_classes=1)
    except HybridModelError as exc:
        raise AutoencoderModelError(str(exc)) from exc


def build_hybrid_autoencoder(hybrid_cfg: Mapping[str, Any], input_shape: tuple[int, int]) -> tf.keras.Model:
    """Собирает и компилирует AE: encoder = Conv1D + GRU, decoder = GRU + TimeDistributed(Dense)."""
    config = validate_autoencoder_config(hybrid_cfg, input_shape=input_shape)

    tf.keras.backend.clear_session()

    inputs = tf.keras.layers.Input(shape=input_shape, name="telemetry_input")
    x = inputs

    for layer_idx in range(config["n_conv_layers"]):
        x = tf.keras.layers.Conv1D(
            filters=config["conv_filters"],
            kernel_size=config["conv_kernel_size"],
            padding="same",
            name=f"conv1d_{layer_idx + 1}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{layer_idx + 1}")(x)
        x = tf.keras.layers.Activation(config["activation"], name=f"conv_activation_{layer_idx + 1}")(x)
        x = tf.keras.layers.Dropout(config["conv_dropout"], name=f"conv_dropout_{layer_idx + 1}")(x)

    for layer_idx in range(config["n_gru_layers"]):
        return_sequences = layer_idx < (config["n_gru_layers"] - 1)
        x = tf.keras.layers.GRU(
            units=config["gru_units"],
            return_sequences=return_sequences,
            name=f"gru_{layer_idx + 1}",
        )(x)

    encoded = x
    x = tf.keras.layers.RepeatVector(config["time_steps"], name="repeat_vector")(encoded)
    x = tf.keras.layers.GRU(
        units=config["gru_units"],
        return_sequences=True,
        name="decoder_gru_1",
    )(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(config["n_features"]),
        name="reconstruction_head",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hybrid_cnn_gru_autoencoder")
    model.compile(
        optimizer=create_optimizer(config["optimizer"]),
        loss="mse",
    )
    return model


def _extract_layer_weights(layer: tf.keras.layers.Layer) -> list[np.ndarray]:
    return [np.array(weights, copy=True) for weights in layer.get_weights()]


def extract_encoder_weights(model: tf.keras.Model) -> dict[str, list[np.ndarray]]:
    """Извлекает веса encoder-части (Conv/BatchNorm/GRU) по именам слоёв."""
    if not isinstance(model, tf.keras.Model):
        raise AutoencoderModelError("model должен быть экземпляром tf.keras.Model")

    weights_by_layer: dict[str, list[np.ndarray]] = {}
    for layer in model.layers:
        if not layer.name.startswith(ENCODER_WEIGHT_PREFIXES):
            continue
        layer_weights = _extract_layer_weights(layer)
        if not layer_weights:
            continue
        weights_by_layer[layer.name] = layer_weights

    if not weights_by_layer:
        raise AutoencoderModelError("Не удалось извлечь веса энкодера из переданной модели")
    return weights_by_layer


def apply_encoder_weights(
    model: tf.keras.Model,
    encoder_weights: Mapping[str, list[np.ndarray]],
) -> list[str]:
    """Переносит веса encoder-слоёв в классификатор по совпадающим именам."""
    if not isinstance(model, tf.keras.Model):
        raise AutoencoderModelError("model должен быть экземпляром tf.keras.Model")
    if not isinstance(encoder_weights, Mapping) or not encoder_weights:
        raise AutoencoderModelError("encoder_weights должен быть непустым словарем")

    transferred_layers: list[str] = []
    for layer in model.layers:
        if layer.name not in encoder_weights:
            continue
        weights = list(encoder_weights[layer.name])
        if not weights:
            continue
        layer.set_weights(weights)
        transferred_layers.append(layer.name)

    if not transferred_layers:
        raise AutoencoderModelError("Не удалось перенести ни одного слоя энкодера")
    return transferred_layers

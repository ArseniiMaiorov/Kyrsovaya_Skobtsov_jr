"""Сборка и компиляция гибридной модели 1D-CNN -> GRU -> Dense."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf

SUPPORTED_ACTIVATIONS = {"relu", "elu", "tanh"}
SUPPORTED_OPTIMIZERS = {"adam", "rmsprop", "nadam"}
SUPPORTED_LOSSES = {"categorical_crossentropy", "sparse_categorical_crossentropy"}
SUPPORTED_RNN_TYPES = {"gru", "lstm", "bi_gru", "bi_lstm"}


class HybridModelError(ValueError):
    """Исключение для ошибок конфигурации и сборки гибридной модели."""


class AttentionLayer(tf.keras.layers.Layer):
    """Простой слой внимания для последовательностей."""

    def __init__(self, units: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.units = _require_positive_int(units, "attention_units")
        self.proj = tf.keras.layers.Dense(self.units, activation="tanh", name=f"{self.name}_proj")
        self.score = tf.keras.layers.Dense(1, name=f"{self.name}_score")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        scores = self.score(self.proj(inputs))
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * inputs, axis=1)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["units"] = self.units
        return config


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise HybridModelError(f"{field_name} должен быть положительным целым числом")
    return int(value)


def _require_non_negative_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or value < 0:
        raise HybridModelError(f"{field_name} должен быть числом >= 0")
    return float(value)


def _require_dropout(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise HybridModelError(f"{field_name} должен быть числом в диапазоне [0, 1)")
    return float(value)


def _validate_input_shape(input_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(input_shape, tuple) or len(input_shape) != 2:
        raise HybridModelError("input_shape должен быть кортежем вида (T, n_features)")

    time_steps = _require_positive_int(input_shape[0], "input_shape[0]")
    n_features = _require_positive_int(input_shape[1], "input_shape[1]")
    return time_steps, n_features


def validate_hybrid_config(hybrid_cfg: Mapping[str, Any], input_shape: tuple[int, int], n_classes: int) -> dict[str, Any]:
    """Проверяет конфиг гибридной модели и возвращает нормализованный словарь."""
    if not isinstance(hybrid_cfg, Mapping):
        raise HybridModelError("hybrid_cfg должен быть словарем")

    time_steps, n_features = _validate_input_shape(input_shape)
    class_count = _require_positive_int(n_classes, "n_classes")

    config = {
        "n_conv_layers": _require_positive_int(hybrid_cfg.get("n_conv_layers"), "n_conv_layers"),
        "conv_filters": _require_positive_int(hybrid_cfg.get("conv_filters"), "conv_filters"),
        "conv_kernel_size": _require_positive_int(hybrid_cfg.get("conv_kernel_size"), "conv_kernel_size"),
        "n_gru_layers": _require_positive_int(hybrid_cfg.get("n_gru_layers"), "n_gru_layers"),
        "gru_units": _require_positive_int(hybrid_cfg.get("gru_units"), "gru_units"),
        "n_dense_layers": _require_positive_int(hybrid_cfg.get("n_dense_layers"), "n_dense_layers"),
        "dense_units": _require_positive_int(hybrid_cfg.get("dense_units"), "dense_units"),
        "conv_dropout": _require_dropout(hybrid_cfg.get("conv_dropout"), "conv_dropout"),
        "dense_dropout": _require_dropout(hybrid_cfg.get("dense_dropout"), "dense_dropout"),
        "l2_dense": _require_non_negative_float(hybrid_cfg.get("l2_dense"), "l2_dense"),
        "time_steps": time_steps,
        "n_features": n_features,
        "n_classes": class_count,
    }

    if config["conv_kernel_size"] > time_steps:
        raise HybridModelError("conv_kernel_size не может быть больше длины окна T")

    activation = hybrid_cfg.get("activation")
    if activation not in SUPPORTED_ACTIVATIONS:
        supported = ", ".join(sorted(SUPPORTED_ACTIVATIONS))
        raise HybridModelError(f"activation должен быть одним из: {supported}")
    config["activation"] = str(activation)

    optimizer = hybrid_cfg.get("optimizer")
    if optimizer not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
        raise HybridModelError(f"optimizer должен быть одним из: {supported}")
    config["optimizer"] = str(optimizer)

    loss = hybrid_cfg.get("loss")
    if loss not in SUPPORTED_LOSSES:
        supported = ", ".join(sorted(SUPPORTED_LOSSES))
        raise HybridModelError(f"loss должен быть одним из: {supported}")
    config["loss"] = str(loss)

    rnn_type = hybrid_cfg.get("rnn_type", "gru")
    if rnn_type not in SUPPORTED_RNN_TYPES:
        supported = ", ".join(sorted(SUPPORTED_RNN_TYPES))
        raise HybridModelError(f"rnn_type должен быть одним из: {supported}")
    config["rnn_type"] = str(rnn_type)

    use_attention = hybrid_cfg.get("use_attention", False)
    if not isinstance(use_attention, bool):
        raise HybridModelError("use_attention должен быть булевым значением")
    config["use_attention"] = use_attention

    attention_units = hybrid_cfg.get("attention_units", config["gru_units"])
    if use_attention:
        config["attention_units"] = _require_positive_int(attention_units, "attention_units")
    else:
        config["attention_units"] = int(attention_units) if isinstance(attention_units, int) else config["gru_units"]

    return config


def create_optimizer(optimizer_name: str) -> tf.keras.optimizers.Optimizer:
    """Создаёт оптимизатор Keras по имени."""
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam()
    if optimizer_name == "rmsprop":
        return tf.keras.optimizers.RMSprop()
    if optimizer_name == "nadam":
        return tf.keras.optimizers.Nadam()

    supported = ", ".join(sorted(SUPPORTED_OPTIMIZERS))
    raise HybridModelError(f"optimizer должен быть одним из: {supported}")


def _build_recurrent_block(
    x: tf.Tensor,
    config: Mapping[str, Any],
    layer_idx: int,
    return_sequences: bool,
) -> tf.Tensor:
    units = int(config["gru_units"])
    rnn_type = str(config["rnn_type"])
    layer_number = layer_idx + 1

    if rnn_type == "gru":
        return tf.keras.layers.GRU(
            units=units,
            return_sequences=return_sequences,
            name=f"gru_{layer_number}",
        )(x)

    if rnn_type == "lstm":
        return tf.keras.layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            name=f"lstm_{layer_number}",
        )(x)

    if rnn_type == "bi_gru":
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=units, return_sequences=return_sequences),
            name=f"bi_gru_{layer_number}",
        )(x)

    if rnn_type == "bi_lstm":
        return tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=units, return_sequences=return_sequences),
            name=f"bi_lstm_{layer_number}",
        )(x)

    supported = ", ".join(sorted(SUPPORTED_RNN_TYPES))
    raise HybridModelError(f"rnn_type должен быть одним из: {supported}")


def build_hybrid_classifier(hybrid_cfg: Mapping[str, Any], input_shape: tuple[int, int], n_classes: int) -> tf.keras.Model:
    """Собирает гибридную модель 1D-CNN -> RNN -> Dense без компиляции."""
    config = validate_hybrid_config(hybrid_cfg, input_shape=input_shape, n_classes=n_classes)

    tf.keras.backend.clear_session()
    regularizer = tf.keras.regularizers.L2(config["l2_dense"]) if config["l2_dense"] > 0 else None

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
        return_sequences = bool(config["use_attention"]) or layer_idx < (config["n_gru_layers"] - 1)
        x = _build_recurrent_block(
            x=x,
            config=config,
            layer_idx=layer_idx,
            return_sequences=return_sequences,
        )

    if bool(config["use_attention"]):
        x = AttentionLayer(int(config["attention_units"]), name="attention")(x)

    for layer_idx in range(config["n_dense_layers"]):
        x = tf.keras.layers.Dense(
            units=config["dense_units"],
            kernel_regularizer=regularizer,
            name=f"dense_{layer_idx + 1}",
        )(x)
        x = tf.keras.layers.Activation(config["activation"], name=f"dense_activation_{layer_idx + 1}")(x)
        x = tf.keras.layers.Dropout(config["dense_dropout"], name=f"dense_dropout_{layer_idx + 1}")(x)

    outputs = tf.keras.layers.Dense(config["n_classes"], activation="softmax", name="classifier_head")(x)

    model_name = "hybrid_cnn_gru_classifier"
    if config["rnn_type"] != "gru" or bool(config["use_attention"]):
        attention_suffix = "_attention" if bool(config["use_attention"]) else ""
        model_name = f"hybrid_cnn_{config['rnn_type']}{attention_suffix}_classifier"
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def compile_hybrid_classifier(model: tf.keras.Model, hybrid_cfg: Mapping[str, Any]) -> tf.keras.Model:
    """Компилирует гибридную модель по имени оптимизатора и функции потерь."""
    if not isinstance(model, tf.keras.Model):
        raise HybridModelError("model должен быть экземпляром tf.keras.Model")
    if not isinstance(hybrid_cfg, Mapping):
        raise HybridModelError("hybrid_cfg должен быть словарем")

    optimizer = create_optimizer(str(hybrid_cfg.get("optimizer")))
    loss = hybrid_cfg.get("loss")
    if loss not in SUPPORTED_LOSSES:
        supported = ", ".join(sorted(SUPPORTED_LOSSES))
        raise HybridModelError(f"loss должен быть одним из: {supported}")

    model.compile(
        optimizer=optimizer,
        loss=str(loss),
        metrics=["accuracy"],
    )
    return model

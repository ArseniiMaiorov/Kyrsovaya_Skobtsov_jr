from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.autoencoder import (
    AutoencoderModelError,
    apply_encoder_weights,
    build_hybrid_autoencoder,
    extract_encoder_weights,
    validate_autoencoder_config,
)
from src.models.hybrid import build_hybrid_classifier


def _make_hybrid_config() -> dict:
    return {
        "n_conv_layers": 1,
        "conv_filters": 8,
        "conv_kernel_size": 3,
        "n_gru_layers": 1,
        "gru_units": 6,
        "n_dense_layers": 1,
        "dense_units": 5,
        "activation": "relu",
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "conv_dropout": 0.1,
        "dense_dropout": 0.1,
        "l2_dense": 0.0,
        "batch_size": 4,
        "max_epochs": 2,
    }


def test_validate_autoencoder_config_success_and_error():
    config = validate_autoencoder_config(_make_hybrid_config(), input_shape=(4, 2))
    assert config["time_steps"] == 4
    assert config["n_features"] == 2

    broken = _make_hybrid_config()
    broken["conv_kernel_size"] = 9
    with pytest.raises(AutoencoderModelError, match="conv_kernel_size"):
        validate_autoencoder_config(broken, input_shape=(4, 2))


def test_build_hybrid_autoencoder_and_forward_pass():
    tf.keras.utils.set_random_seed(42)
    model = build_hybrid_autoencoder(_make_hybrid_config(), input_shape=(4, 2))

    x_batch = np.ones((2, 4, 2), dtype=np.float32)
    reconstructed = model(x_batch, training=False).numpy()

    assert model.name == "hybrid_cnn_gru_autoencoder"
    assert reconstructed.shape == (2, 4, 2)
    tf.keras.backend.clear_session()


def test_extract_and_apply_encoder_weights_success_and_errors():
    tf.keras.utils.set_random_seed(42)
    autoencoder = build_hybrid_autoencoder(_make_hybrid_config(), input_shape=(4, 2))
    weights = extract_encoder_weights(autoencoder)

    assert "conv1d_1" in weights
    assert "gru_1" in weights

    classifier = build_hybrid_classifier(_make_hybrid_config(), input_shape=(4, 2), n_classes=3)
    transferred = apply_encoder_weights(classifier, weights)

    assert "conv1d_1" in transferred
    assert "gru_1" in transferred

    with pytest.raises(AutoencoderModelError, match="model"):
        extract_encoder_weights("bad")  # type: ignore[arg-type]

    inputs = tf.keras.layers.Input(shape=(4, 2))
    outputs = tf.keras.layers.Activation("linear", name="gru_1")(inputs)
    empty_encoder_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    with pytest.raises(AutoencoderModelError, match="Не удалось извлечь"):
        extract_encoder_weights(empty_encoder_model)

    with pytest.raises(AutoencoderModelError, match="model"):
        apply_encoder_weights("bad", weights)  # type: ignore[arg-type]

    with pytest.raises(AutoencoderModelError, match="encoder_weights"):
        apply_encoder_weights(classifier, {})

    with pytest.raises(AutoencoderModelError, match="Не удалось перенести"):
        apply_encoder_weights(classifier, {"conv1d_1": []})

    with pytest.raises(AutoencoderModelError, match="Не удалось перенести"):
        apply_encoder_weights(classifier, {"missing_layer": [np.zeros((1,), dtype=np.float32)]})

    tf.keras.backend.clear_session()

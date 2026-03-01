from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.hybrid import (
    AttentionLayer,
    _build_recurrent_block,
    HybridModelError,
    build_hybrid_classifier,
    compile_hybrid_classifier,
    create_optimizer,
    validate_hybrid_config,
)


def _make_hybrid_config() -> dict:
    return {
        "n_conv_layers": 2,
        "conv_filters": 8,
        "conv_kernel_size": 3,
        "n_gru_layers": 2,
        "gru_units": 6,
        "n_dense_layers": 2,
        "dense_units": 5,
        "activation": "relu",
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "conv_dropout": 0.1,
        "dense_dropout": 0.2,
        "l2_dense": 0.001,
        "batch_size": 4,
        "max_epochs": 2,
        "rnn_type": "gru",
        "use_attention": False,
        "attention_units": 6,
    }


def test_validate_hybrid_config_success():
    config = validate_hybrid_config(_make_hybrid_config(), input_shape=(8, 3), n_classes=3)
    assert config["time_steps"] == 8
    assert config["n_features"] == 3
    assert config["n_classes"] == 3
    assert config["rnn_type"] == "gru"
    assert config["use_attention"] is False


def test_validate_hybrid_config_not_mapping_error():
    with pytest.raises(HybridModelError, match="hybrid_cfg"):
        validate_hybrid_config("bad", input_shape=(8, 3), n_classes=3)


@pytest.mark.parametrize(
    ("input_shape", "error"),
    [
        ((8,), "input_shape"),
        ((0, 3), "input_shape\\[0\\]"),
        ((8, 0), "input_shape\\[1\\]"),
    ],
)
def test_validate_hybrid_config_input_shape_errors(input_shape, error):
    with pytest.raises(HybridModelError, match=error):
        validate_hybrid_config(_make_hybrid_config(), input_shape=input_shape, n_classes=3)


def test_validate_hybrid_config_n_classes_error():
    with pytest.raises(HybridModelError, match="n_classes"):
        validate_hybrid_config(_make_hybrid_config(), input_shape=(8, 3), n_classes=0)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("n_conv_layers", 0, "n_conv_layers"),
        ("conv_filters", 0, "conv_filters"),
        ("conv_kernel_size", 0, "conv_kernel_size"),
        ("n_gru_layers", 0, "n_gru_layers"),
        ("gru_units", 0, "gru_units"),
        ("n_dense_layers", 0, "n_dense_layers"),
        ("dense_units", 0, "dense_units"),
    ],
)
def test_validate_hybrid_config_positive_int_errors(field, value, error):
    cfg = _make_hybrid_config()
    cfg[field] = value
    with pytest.raises(HybridModelError, match=error):
        validate_hybrid_config(cfg, input_shape=(8, 3), n_classes=3)


def test_validate_hybrid_config_kernel_too_large_error():
    cfg = _make_hybrid_config()
    cfg["conv_kernel_size"] = 9
    with pytest.raises(HybridModelError, match="conv_kernel_size"):
        validate_hybrid_config(cfg, input_shape=(8, 3), n_classes=3)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("conv_dropout", 1.0, "conv_dropout"),
        ("dense_dropout", 1.0, "dense_dropout"),
        ("l2_dense", -0.1, "l2_dense"),
    ],
)
def test_validate_hybrid_config_float_errors(field, value, error):
    cfg = _make_hybrid_config()
    cfg[field] = value
    with pytest.raises(HybridModelError, match=error):
        validate_hybrid_config(cfg, input_shape=(8, 3), n_classes=3)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("activation", "sigmoid", "activation"),
        ("optimizer", "sgd", "optimizer"),
        ("loss", "mse", "loss"),
    ],
)
def test_validate_hybrid_config_choice_errors(field, value, error):
    cfg = _make_hybrid_config()
    cfg[field] = value
    with pytest.raises(HybridModelError, match=error):
        validate_hybrid_config(cfg, input_shape=(8, 3), n_classes=3)


def test_validate_hybrid_config_rnn_and_attention_errors():
    cfg = _make_hybrid_config()
    cfg["rnn_type"] = "rnn"
    with pytest.raises(HybridModelError, match="rnn_type"):
        validate_hybrid_config(cfg, input_shape=(8, 3), n_classes=3)

    cfg = _make_hybrid_config()
    cfg["use_attention"] = "yes"
    with pytest.raises(HybridModelError, match="use_attention"):
        validate_hybrid_config(cfg, input_shape=(8, 3), n_classes=3)

    cfg = _make_hybrid_config()
    cfg["use_attention"] = True
    cfg["attention_units"] = 0
    with pytest.raises(HybridModelError, match="attention_units"):
        validate_hybrid_config(cfg, input_shape=(8, 3), n_classes=3)


def test_create_optimizer_success_and_error():
    assert create_optimizer("adam").__class__.__name__ == "Adam"
    assert create_optimizer("rmsprop").__class__.__name__ == "RMSprop"
    assert create_optimizer("nadam").__class__.__name__ == "Nadam"

    with pytest.raises(HybridModelError, match="optimizer"):
        create_optimizer("sgd")


def test_build_hybrid_classifier_and_forward_pass():
    tf.keras.utils.set_random_seed(42)
    model = build_hybrid_classifier(_make_hybrid_config(), input_shape=(8, 3), n_classes=3)

    x_batch = np.ones((2, 8, 3), dtype=np.float32)
    y_pred = model(x_batch, training=False).numpy()

    assert model.name == "hybrid_cnn_gru_classifier"
    assert y_pred.shape == (2, 3)
    assert np.allclose(y_pred.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize("rnn_type", ["gru", "lstm", "bi_gru", "bi_lstm"])
def test_build_hybrid_classifier_supports_rnn_types(rnn_type):
    cfg = _make_hybrid_config()
    cfg["rnn_type"] = rnn_type
    tf.keras.utils.set_random_seed(42)
    model = build_hybrid_classifier(cfg, input_shape=(8, 3), n_classes=3)
    output = model(np.ones((1, 8, 3), dtype=np.float32), training=False).numpy()

    assert output.shape == (1, 3)
    tf.keras.backend.clear_session()


def test_build_hybrid_classifier_with_attention():
    cfg = _make_hybrid_config()
    cfg["use_attention"] = True
    tf.keras.utils.set_random_seed(42)
    model = build_hybrid_classifier(cfg, input_shape=(8, 3), n_classes=3)

    assert model.get_layer("attention") is not None
    assert "attention" in model.name
    tf.keras.backend.clear_session()


def test_attention_layer_get_config():
    layer = AttentionLayer(4, name="attention_test")
    config = layer.get_config()

    assert config["units"] == 4


def test_build_recurrent_block_invalid_rnn_type_branch():
    x = tf.keras.Input(shape=(8, 3))
    with pytest.raises(HybridModelError, match="rnn_type"):
        _build_recurrent_block(
            x=x,
            config={"gru_units": 4, "rnn_type": "bad"},
            layer_idx=0,
            return_sequences=False,
        )


def test_compile_hybrid_classifier_success_and_errors():
    tf.keras.utils.set_random_seed(42)
    model = build_hybrid_classifier(_make_hybrid_config(), input_shape=(8, 3), n_classes=3)
    compiled = compile_hybrid_classifier(model, _make_hybrid_config())

    assert compiled.optimizer is not None

    with pytest.raises(HybridModelError, match="model"):
        compile_hybrid_classifier("bad", _make_hybrid_config())

    with pytest.raises(HybridModelError, match="hybrid_cfg"):
        compile_hybrid_classifier(model, "bad")

    cfg = _make_hybrid_config()
    cfg["loss"] = "mse"
    with pytest.raises(HybridModelError, match="loss"):
        compile_hybrid_classifier(model, cfg)

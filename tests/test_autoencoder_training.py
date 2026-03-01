from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.autoencoder import extract_encoder_weights
from src.training.autoencoder_training import (
    AutoencoderTrainingError,
    evaluate_reconstruction,
    run_autoencoder_pretraining,
    run_pretrained_hybrid_experiment,
    split_unlabeled_windows_for_pretrain,
    train_autoencoder,
)
from src.training.hybrid_training import build_training_callbacks


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
        "batch_size": 16,
        "max_epochs": 2,
    }


def _make_autoencoder_config() -> dict:
    return {
        "batch_size": 16,
        "pretrain_max_epochs": 2,
        "pretrain_val_ratio": 0.25,
        "use_stage6_best_genome": True,
    }


def _make_training_config() -> dict:
    return {
        "early_stopping_patience": 2,
        "reduce_lr_patience": 1,
        "reduce_lr_factor": 0.5,
    }


def _make_labeled_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.array(
        [
            [[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1]],
            [[0.2, 0.0], [0.2, 0.1], [0.3, 0.0], [0.3, 0.1]],
            [[1.0, 1.0], [1.0, 1.1], [1.1, 1.0], [1.1, 1.1]],
            [[1.2, 1.0], [1.2, 1.1], [1.3, 1.0], [1.3, 1.1]],
            [[2.0, 2.0], [2.0, 2.1], [2.1, 2.0], [2.1, 2.1]],
            [[2.2, 2.0], [2.2, 2.1], [2.3, 2.0], [2.3, 2.1]],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    x_val = np.array(
        [
            [[0.05, 0.05], [0.1, 0.1], [0.0, 0.0], [0.1, 0.0]],
            [[1.05, 1.05], [1.1, 1.1], [1.0, 1.0], [1.1, 1.0]],
            [[2.05, 2.05], [2.1, 2.1], [2.0, 2.0], [2.1, 2.0]],
        ],
        dtype=np.float32,
    )
    y_val = np.array([0, 1, 2], dtype=np.int64)
    return x_train, y_train, x_val, y_val


def _make_unlabeled_windows() -> np.ndarray:
    return np.array(
        [
            [[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1]],
            [[0.2, 0.0], [0.2, 0.1], [0.3, 0.0], [0.3, 0.1]],
            [[1.0, 1.0], [1.0, 1.1], [1.1, 1.0], [1.1, 1.1]],
            [[1.2, 1.0], [1.2, 1.1], [1.3, 1.0], [1.3, 1.1]],
            [[2.0, 2.0], [2.0, 2.1], [2.1, 2.0], [2.1, 2.1]],
            [[2.2, 2.0], [2.2, 2.1], [2.3, 2.0], [2.3, 2.1]],
        ],
        dtype=np.float32,
    )


def test_split_unlabeled_windows_for_pretrain_success_and_errors():
    x_unlabeled = _make_unlabeled_windows()
    x_train, x_val = split_unlabeled_windows_for_pretrain(x_unlabeled, val_ratio=0.25)

    assert x_train.shape[0] == 5
    assert x_val.shape[0] == 1

    with pytest.raises(AutoencoderTrainingError, match="numpy.ndarray"):
        split_unlabeled_windows_for_pretrain("bad", 0.25)  # type: ignore[arg-type]

    with pytest.raises(AutoencoderTrainingError, match="размерность"):
        split_unlabeled_windows_for_pretrain(np.array([1.0, 2.0], dtype=np.float32), 0.25)

    with pytest.raises(AutoencoderTrainingError, match="минимум 2 окна"):
        split_unlabeled_windows_for_pretrain(np.ones((1, 4, 2), dtype=np.float32), 0.25)

    with pytest.raises(AutoencoderTrainingError, match="val_ratio"):
        split_unlabeled_windows_for_pretrain(x_unlabeled, val_ratio=1.0)


def test_train_autoencoder_and_evaluate_reconstruction_success_and_errors():
    from src.models.autoencoder import build_hybrid_autoencoder

    tf.keras.utils.set_random_seed(42)
    model = build_hybrid_autoencoder(_make_hybrid_config(), input_shape=(4, 2))
    x_train, x_val = split_unlabeled_windows_for_pretrain(_make_unlabeled_windows(), val_ratio=0.25)
    callbacks = build_training_callbacks(_make_training_config())

    history = train_autoencoder(
        model=model,
        x_train=x_train,
        x_val=x_val,
        batch_size=16,
        max_epochs=2,
        callbacks=callbacks,
    )
    metrics = evaluate_reconstruction(model, x_val)

    assert history.history["loss"]
    assert metrics["mean_reconstruction_mse"] >= 0

    with pytest.raises(AutoencoderTrainingError, match="model"):
        train_autoencoder("bad", x_train, x_val, 4, 2, callbacks)  # type: ignore[arg-type]

    with pytest.raises(AutoencoderTrainingError, match="callbacks"):
        train_autoencoder(model, x_train, x_val, 4, 2, "bad")  # type: ignore[arg-type]

    with pytest.raises(AutoencoderTrainingError, match="batch_size"):
        train_autoencoder(model, x_train, x_val, 0, 2, callbacks)

    with pytest.raises(AutoencoderTrainingError, match="max_epochs"):
        train_autoencoder(model, x_train, x_val, 4, 0, callbacks)

    with pytest.raises(AutoencoderTrainingError, match="Формы окон"):
        train_autoencoder(model, x_train, np.ones((1, 5, 2), dtype=np.float32), 4, 2, callbacks)

    with pytest.raises(AutoencoderTrainingError, match="model"):
        evaluate_reconstruction("bad", x_val)  # type: ignore[arg-type]

    tf.keras.backend.clear_session()


def test_run_autoencoder_pretraining_success_and_error():
    tf.keras.utils.set_random_seed(42)
    result = run_autoencoder_pretraining(
        x_unlabeled=_make_unlabeled_windows(),
        hybrid_cfg=_make_hybrid_config(),
        autoencoder_cfg=_make_autoencoder_config(),
        training_cfg=_make_training_config(),
    )

    assert "model" in result
    assert result["split_summary"]["train_windows"] == 5
    assert result["split_summary"]["val_windows"] == 1
    assert result["history"]["best_epoch"] >= 1

    with pytest.raises(AutoencoderTrainingError, match="autoencoder_cfg"):
        run_autoencoder_pretraining(
            x_unlabeled=_make_unlabeled_windows(),
            hybrid_cfg=_make_hybrid_config(),
            autoencoder_cfg="bad",  # type: ignore[arg-type]
            training_cfg=_make_training_config(),
        )

    bad_hybrid_cfg = _make_hybrid_config()
    bad_hybrid_cfg["conv_kernel_size"] = 9
    with pytest.raises(AutoencoderTrainingError, match="conv_kernel_size"):
        run_autoencoder_pretraining(
            x_unlabeled=_make_unlabeled_windows(),
            hybrid_cfg=bad_hybrid_cfg,
            autoencoder_cfg=_make_autoencoder_config(),
            training_cfg=_make_training_config(),
        )

    tf.keras.backend.clear_session()


def test_run_pretrained_hybrid_experiment_success_and_errors():
    tf.keras.utils.set_random_seed(42)
    ae_result = run_autoencoder_pretraining(
        x_unlabeled=_make_unlabeled_windows(),
        hybrid_cfg=_make_hybrid_config(),
        autoencoder_cfg=_make_autoencoder_config(),
        training_cfg=_make_training_config(),
    )
    encoder_weights = extract_encoder_weights(ae_result["model"])

    x_train, y_train, x_val, y_val = _make_labeled_dataset()
    result = run_pretrained_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=(0, 1, 2),
        hybrid_cfg=_make_hybrid_config(),
        training_cfg=_make_training_config(),
        encoder_weights=encoder_weights,
    )

    assert result["transferred_layers"]
    assert "macro_f1" in result["metrics"]
    assert result["history"]["best_epoch"] >= 1

    with pytest.raises(AutoencoderTrainingError, match="labels не должен быть пустым"):
        run_pretrained_hybrid_experiment(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            labels=(),
            hybrid_cfg=_make_hybrid_config(),
            training_cfg=_make_training_config(),
            encoder_weights=encoder_weights,
        )

    with pytest.raises(AutoencoderTrainingError, match="encoder_weights"):
        run_pretrained_hybrid_experiment(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            labels=(0, 1, 2),
            hybrid_cfg=_make_hybrid_config(),
            training_cfg=_make_training_config(),
            encoder_weights={},
        )

    tf.keras.backend.clear_session()

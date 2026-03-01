from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from src.models.hybrid import build_hybrid_classifier, compile_hybrid_classifier
from src.training.hybrid_training import (
    _project_to_2d,
    HybridTrainingError,
    build_training_callbacks,
    compute_balanced_class_weights,
    evaluate_hybrid_classifier,
    history_to_serializable_dict,
    plot_attention_weights,
    plot_training_curves,
    predict_hybrid_probabilities,
    run_hybrid_experiment,
    save_history_artifacts,
    summarize_history,
    train_hybrid_classifier,
    visualize_conv_filters,
    visualize_hidden_representations,
)


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
        "rnn_type": "gru",
        "use_attention": False,
        "attention_units": 6,
    }


def _make_training_config() -> dict:
    return {
        "early_stopping_patience": 2,
        "reduce_lr_patience": 1,
        "reduce_lr_factor": 0.5,
    }


def _make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
            [[0.2, 0.0], [0.1, 0.0], [0.2, 0.1], [0.0, 0.2]],
            [[1.0, 1.0], [1.1, 1.0], [1.0, 1.1], [1.1, 1.1]],
            [[1.2, 1.0], [1.1, 1.0], [1.2, 1.1], [1.1, 1.2]],
            [[2.0, 2.0], [2.1, 2.0], [2.0, 2.1], [2.1, 2.1]],
            [[2.2, 2.0], [2.1, 2.0], [2.2, 2.1], [2.1, 2.2]],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    x_val = np.array(
        [
            [[0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.0, 0.0]],
            [[1.1, 1.0], [1.0, 1.1], [1.1, 1.1], [1.0, 1.0]],
            [[2.1, 2.0], [2.0, 2.1], [2.1, 2.1], [2.0, 2.0]],
        ],
        dtype=np.float32,
    )
    y_val = np.array([0, 1, 2], dtype=np.int64)
    return x_train, y_train, x_val, y_val


def _make_compiled_model() -> tf.keras.Model:
    tf.keras.utils.set_random_seed(42)
    model = build_hybrid_classifier(_make_hybrid_config(), input_shape=(4, 2), n_classes=3)
    return compile_hybrid_classifier(model, _make_hybrid_config())


def test_build_training_callbacks_success_and_errors():
    callbacks = build_training_callbacks(_make_training_config())
    assert len(callbacks) == 2
    assert callbacks[0].__class__.__name__ == "EarlyStopping"
    assert callbacks[1].__class__.__name__ == "ReduceLROnPlateau"

    with pytest.raises(HybridTrainingError, match="training_cfg"):
        build_training_callbacks("bad")

    broken = _make_training_config()
    broken["early_stopping_patience"] = 0
    with pytest.raises(HybridTrainingError, match="early_stopping_patience"):
        build_training_callbacks(broken)

    broken = _make_training_config()
    broken["reduce_lr_patience"] = 0
    with pytest.raises(HybridTrainingError, match="reduce_lr_patience"):
        build_training_callbacks(broken)

    broken = _make_training_config()
    broken["reduce_lr_factor"] = 1.0
    with pytest.raises(HybridTrainingError, match="reduce_lr_factor"):
        build_training_callbacks(broken)


def test_compute_balanced_class_weights_success_and_errors():
    y_train = np.array([0, 0, 1, 2], dtype=np.int64)
    weights = compute_balanced_class_weights(y_train)

    assert weights[0] == pytest.approx(4 / (3 * 2))
    assert weights[1] == pytest.approx(4 / (3 * 1))
    assert weights[2] == pytest.approx(4 / (3 * 1))

    with pytest.raises(HybridTrainingError, match="одномерным"):
        compute_balanced_class_weights(np.array([[0], [1]], dtype=np.int64))

    with pytest.raises(HybridTrainingError, match="не должен быть пустым"):
        compute_balanced_class_weights(np.empty((0,), dtype=np.int64))


def test_train_hybrid_classifier_validation_errors():
    model = _make_compiled_model()
    x_train, y_train, x_val, y_val = _make_dataset()
    callbacks = build_training_callbacks(_make_training_config())
    class_weight = compute_balanced_class_weights(y_train)

    with pytest.raises(HybridTrainingError, match="model"):
        train_hybrid_classifier("bad", x_train, y_train, x_val, y_val, 4, 2, callbacks)

    with pytest.raises(HybridTrainingError, match="x_train"):
        train_hybrid_classifier(model, np.array([1.0, 2.0]), y_train, x_val, y_val, 4, 2, callbacks)

    with pytest.raises(HybridTrainingError, match="y_train"):
        train_hybrid_classifier(model, x_train, np.array([[0], [1]]), x_val, y_val, 4, 2, callbacks)

    with pytest.raises(HybridTrainingError, match="не должен быть пустым"):
        train_hybrid_classifier(model, np.empty((0, 4, 2), dtype=np.float32), np.empty((0,), dtype=np.int64), x_val, y_val, 4, 2, callbacks)

    with pytest.raises(HybridTrainingError, match="должно совпадать"):
        train_hybrid_classifier(model, x_train, y_train[:-1], x_val, y_val, 4, 2, callbacks)

    with pytest.raises(HybridTrainingError, match="Формы окон"):
        train_hybrid_classifier(model, x_train, y_train, np.ones((3, 5, 2), dtype=np.float32), y_val, 4, 2, callbacks)

    with pytest.raises(HybridTrainingError, match="batch_size"):
        train_hybrid_classifier(model, x_train, y_train, x_val, y_val, 0, 2, callbacks)

    with pytest.raises(HybridTrainingError, match="max_epochs"):
        train_hybrid_classifier(model, x_train, y_train, x_val, y_val, 4, 0, callbacks)

    with pytest.raises(HybridTrainingError, match="callbacks"):
        train_hybrid_classifier(model, x_train, y_train, x_val, y_val, 4, 2, "bad")

    with pytest.raises(HybridTrainingError, match="class_weight"):
        train_hybrid_classifier(model, x_train, y_train, x_val, y_val, 4, 2, callbacks, class_weight="bad")

    tf.keras.backend.clear_session()


def test_train_hybrid_classifier_smoke_and_summary():
    model = _make_compiled_model()
    x_train, y_train, x_val, y_val = _make_dataset()
    callbacks = build_training_callbacks(_make_training_config())
    class_weight = compute_balanced_class_weights(y_train)

    history = train_hybrid_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        batch_size=16,
        max_epochs=2,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    summary = summarize_history(history)

    assert summary["epochs_ran"] >= 1
    assert summary["best_epoch"] >= 1
    assert "best_val_loss" in summary
    history_dict = history_to_serializable_dict(history)
    assert "loss" in history_dict
    tf.keras.backend.clear_session()


def test_summarize_history_errors():
    with pytest.raises(HybridTrainingError, match="history"):
        summarize_history("bad")

    history = tf.keras.callbacks.History()
    history.history = {}
    with pytest.raises(HybridTrainingError, match="history не содержит"):
        summarize_history(history)

    with pytest.raises(HybridTrainingError, match="history должен быть объектом"):
        history_to_serializable_dict("bad")


def test_evaluate_hybrid_classifier_success_and_errors():
    model = _make_compiled_model()
    x_train, y_train, x_val, y_val = _make_dataset()
    callbacks = build_training_callbacks(_make_training_config())

    train_hybrid_classifier(model, x_train, y_train, x_val, y_val, 4, 1, callbacks)
    metrics = evaluate_hybrid_classifier(model, x_val, y_val, labels=(0, 1, 2))

    assert "macro_f1" in metrics
    assert "classification_report_text" in metrics

    with pytest.raises(HybridTrainingError, match="model"):
        evaluate_hybrid_classifier("bad", x_val, y_val, labels=(0, 1, 2))

    with pytest.raises(HybridTrainingError, match="labels не должен быть пустым"):
        evaluate_hybrid_classifier(model, x_val, y_val, labels=())

    with pytest.raises(HybridTrainingError, match="x_eval"):
        evaluate_hybrid_classifier(model, np.array([1.0, 2.0]), y_val, labels=(0, 1, 2))

    y_pred, y_proba = predict_hybrid_probabilities(model, x_val)
    assert y_pred.shape == (3,)
    assert y_proba.shape == (3, 3)

    with pytest.raises(HybridTrainingError, match="x_eval должен иметь размерность"):
        predict_hybrid_probabilities(model, np.array([1.0, 2.0]))

    with pytest.raises(HybridTrainingError, match="model должен быть экземпляром"):
        predict_hybrid_probabilities("bad", x_val)

    with pytest.raises(HybridTrainingError, match="не должен быть пустым"):
        predict_hybrid_probabilities(model, np.empty((0, 4, 2), dtype=np.float32))

    tf.keras.backend.clear_session()


def test_run_hybrid_experiment_success_and_errors(tmp_path):
    x_train, y_train, x_val, y_val = _make_dataset()

    result = run_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=(0, 1, 2),
        hybrid_cfg=_make_hybrid_config(),
        training_cfg=_make_training_config(),
        artifacts_dir=tmp_path / "artifacts",
    )

    assert "model" in result
    assert "history" in result
    assert "history_series" in result
    assert "artifacts" in result
    assert "y_proba" in result
    assert "metrics" in result
    assert result["history"]["best_epoch"] >= 1
    assert (tmp_path / "artifacts" / "training_history.json").exists()
    assert (tmp_path / "artifacts" / "training_curves.png").exists()

    with pytest.raises(HybridTrainingError, match="labels не должен быть пустым"):
        run_hybrid_experiment(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            labels=(),
            hybrid_cfg=_make_hybrid_config(),
            training_cfg=_make_training_config(),
        )

    bad_cfg = _make_hybrid_config()
    bad_cfg["conv_kernel_size"] = 9
    with pytest.raises(HybridTrainingError, match="conv_kernel_size"):
        run_hybrid_experiment(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            labels=(0, 1, 2),
            hybrid_cfg=bad_cfg,
            training_cfg=_make_training_config(),
        )

    tf.keras.backend.clear_session()


def test_plot_helpers_and_artifacts(tmp_path, monkeypatch):
    model = _make_compiled_model()
    x_train, y_train, x_val, y_val = _make_dataset()
    callbacks = build_training_callbacks(_make_training_config())
    history = train_hybrid_classifier(model, x_train, y_train, x_val, y_val, 4, 1, callbacks)

    history_plot = tmp_path / "curves.png"
    plot_training_curves(history, history_plot)
    assert history_plot.exists()

    history_plot_from_dict = tmp_path / "curves_dict.png"
    plot_training_curves(history_to_serializable_dict(history), history_plot_from_dict)
    assert history_plot_from_dict.exists()

    history_without_accuracy = {
        "loss": [1.0, 0.8],
        "val_loss": [1.1, 0.9],
    }
    history_plot_no_acc = tmp_path / "curves_no_acc.png"
    plot_training_curves(history_without_accuracy, history_plot_no_acc)
    assert history_plot_no_acc.exists()

    artifacts = save_history_artifacts(history, tmp_path / "history_dir")
    assert Path(artifacts["history_json"]).exists()
    assert Path(artifacts["training_curves"]).exists()

    conv_plot = tmp_path / "conv.png"
    hidden_plot = tmp_path / "hidden.png"
    assert Path(visualize_conv_filters(model, conv_plot, max_filters=5)).exists()
    assert Path(visualize_hidden_representations(model, x_val, y_val, hidden_plot)).exists()

    cfg = _make_hybrid_config()
    cfg["use_attention"] = True
    attention_model = build_hybrid_classifier(cfg, input_shape=(4, 2), n_classes=3)
    attention_model = compile_hybrid_classifier(attention_model, cfg)
    train_hybrid_classifier(attention_model, x_train, y_train, x_val, y_val, 4, 1, callbacks)
    attention_plot = tmp_path / "attention.png"
    assert Path(plot_attention_weights(attention_model, x_val[0], attention_plot)).exists()

    with pytest.raises(HybridTrainingError, match="history должен быть History или словарем"):
        plot_training_curves(123, tmp_path / "x.png")

    with pytest.raises(HybridTrainingError, match="model должен быть экземпляром"):
        visualize_conv_filters("bad", conv_plot)

    with pytest.raises(HybridTrainingError, match="model должен быть экземпляром"):
        visualize_hidden_representations("bad", x_val, y_val, hidden_plot)

    with pytest.raises(HybridTrainingError, match="model должен быть экземпляром"):
        plot_attention_weights("bad", x_val[0], attention_plot)

    with pytest.raises(HybridTrainingError, match="sample_window должен иметь форму"):
        plot_attention_weights(attention_model, x_val, attention_plot)

    no_attention_model = _make_compiled_model()
    with pytest.raises(HybridTrainingError, match="attention"):
        plot_attention_weights(no_attention_model, x_val[0], attention_plot)

    no_conv_inputs = tf.keras.layers.Input(shape=(4, 2))
    no_conv_outputs = tf.keras.layers.Flatten()(no_conv_inputs)
    no_conv_outputs = tf.keras.layers.Dense(3, activation="softmax")(no_conv_outputs)
    no_conv_model = tf.keras.Model(inputs=no_conv_inputs, outputs=no_conv_outputs)
    with pytest.raises(HybridTrainingError, match="conv1d_1"):
        visualize_conv_filters(no_conv_model, conv_plot)

    class _EmptyWeightsLayer:
        @staticmethod
        def get_weights():
            return []

    monkeypatch.setattr(model, "get_layer", lambda name: _EmptyWeightsLayer())
    with pytest.raises(HybridTrainingError, match="не содержит весов"):
        visualize_conv_filters(model, conv_plot)

    no_rnn_inputs = tf.keras.layers.Input(shape=(4, 2))
    no_rnn_x = tf.keras.layers.Conv1D(4, 3, padding="same", name="conv1d_1")(no_rnn_inputs)
    no_rnn_x = tf.keras.layers.Flatten()(no_rnn_x)
    no_rnn_outputs = tf.keras.layers.Dense(3, activation="softmax")(no_rnn_x)
    no_rnn_model = tf.keras.Model(inputs=no_rnn_inputs, outputs=no_rnn_outputs)
    with pytest.raises(HybridTrainingError, match="рекуррентные слои"):
        visualize_hidden_representations(no_rnn_model, x_val, y_val, hidden_plot)

    tf.keras.backend.clear_session()


def test_project_to_2d_edge_cases():
    single = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    projected_single = _project_to_2d(single)
    assert projected_single.shape == (1, 2)
    assert np.allclose(projected_single, 0.0)

    repeated = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    projected_repeated = _project_to_2d(repeated)
    assert projected_repeated.shape == (2, 2)

    one_feature = np.array([[1.0], [2.0]], dtype=np.float64)
    projected_one_feature = _project_to_2d(one_feature)
    assert projected_one_feature.shape == (2, 2)

from __future__ import annotations

import numpy as np
import pytest

from src.models.baseline import (
    _align_probabilities_to_labels,
    analyze_baseline_feature_importance,
    BaselineModelError,
    build_baseline_model,
    fit_baseline_model,
    predict_with_optional_proba,
    run_baseline_experiment,
)


def _make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1],
            [2.0, 2.0],
            [2.1, 2.1],
            [0.2, 0.0],
            [1.2, 1.0],
            [2.2, 2.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=np.int64)

    x_eval = np.array(
        [
            [0.05, 0.05],
            [1.05, 1.05],
            [2.05, 2.05],
        ],
        dtype=np.float32,
    )
    y_eval = np.array([0, 1, 2], dtype=np.int64)

    return x_train, y_train, x_eval, y_eval


def test_build_baseline_model_success():
    model_lr = build_baseline_model("logistic_regression", random_state=42)
    model_rf = build_baseline_model("random_forest", random_state=42)

    assert model_lr.__class__.__name__ == "LogisticRegression"
    assert model_rf.__class__.__name__ == "RandomForestClassifier"


def test_build_baseline_model_random_state_error():
    with pytest.raises(BaselineModelError, match="random_state"):
        build_baseline_model("logistic_regression", random_state=-1)


def test_build_baseline_model_unknown_name_error():
    with pytest.raises(BaselineModelError, match="Неподдерживаемая baseline-модель"):
        build_baseline_model("svm", random_state=42)


def test_fit_baseline_model_validation_errors():
    model = build_baseline_model("logistic_regression", random_state=42)

    with pytest.raises(BaselineModelError, match="двумерным"):
        fit_baseline_model(model, x_train=np.array([1, 2, 3]), y_train=np.array([0, 1, 2]))

    with pytest.raises(BaselineModelError, match="одномерным"):
        fit_baseline_model(model, x_train=np.array([[1, 2], [3, 4]]), y_train=np.array([[0], [1]]))

    with pytest.raises(BaselineModelError, match="не должен быть пустым"):
        fit_baseline_model(model, x_train=np.empty((0, 2)), y_train=np.empty((0,), dtype=np.int64))

    with pytest.raises(BaselineModelError, match="должно совпадать"):
        fit_baseline_model(model, x_train=np.array([[1, 2], [3, 4]]), y_train=np.array([0]))


def test_predict_with_optional_proba_predict_proba_branch():
    x_train, y_train, x_eval, _ = _make_dataset()
    model = build_baseline_model("logistic_regression", random_state=42)
    model = fit_baseline_model(model, x_train, y_train)

    y_pred, y_proba = predict_with_optional_proba(model, x_eval)

    assert y_pred.shape == (3,)
    assert y_proba is not None
    assert y_proba.shape == (3, 3)


def test_align_probabilities_to_labels_success():
    y_proba = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    aligned = _align_probabilities_to_labels(y_proba, np.array([0, 1]), (0, 1, 2))

    assert aligned.shape == (2, 3)
    assert np.allclose(aligned[:, 2], 0.0)


def test_align_probabilities_to_labels_validation_errors():
    with pytest.raises(BaselineModelError, match="двумерной"):
        _align_probabilities_to_labels(np.array([0.5, 0.5]), np.array([0, 1]), (0, 1, 2))

    with pytest.raises(BaselineModelError, match="Число столбцов"):
        _align_probabilities_to_labels(np.array([[0.5, 0.5]]), np.array([0, 1, 2]), (0, 1, 2))

    with pytest.raises(BaselineModelError, match="вне списка labels"):
        _align_probabilities_to_labels(np.array([[0.5, 0.5]]), np.array([0, 9]), (0, 1, 2))


class _DummyDecisionModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0], dtype=np.int64)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[0], 3), dtype=np.float64)


class _DummyDecisionModel1D:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0], dtype=np.int64)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[0],), dtype=np.float64)


class _DummyBadDecisionModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0], dtype=np.int64)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[0], 2, 2), dtype=np.float64)


class _DummyPredictOnlyModel:
    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0], dtype=np.int64)


def test_predict_with_optional_proba_decision_function_2d_branch():
    x_eval = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = _DummyDecisionModel()

    y_pred, y_proba = predict_with_optional_proba(model, x_eval)

    assert y_pred.tolist() == [0, 0]
    assert y_proba is not None
    assert y_proba.shape == (2, 3)
    assert np.allclose(y_proba.sum(axis=1), 1.0)


def test_predict_with_optional_proba_decision_function_1d_branch():
    x_eval = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = _DummyDecisionModel1D()

    y_pred, y_proba = predict_with_optional_proba(model, x_eval)

    assert y_pred.tolist() == [0, 0]
    assert y_proba is not None
    assert y_proba.shape == (2, 2)


def test_predict_with_optional_proba_bad_decision_error():
    x_eval = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = _DummyBadDecisionModel()

    with pytest.raises(BaselineModelError, match="неподдерживаемой размерностью"):
        predict_with_optional_proba(model, x_eval)


def test_predict_with_optional_proba_predict_only_branch():
    x_eval = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = _DummyPredictOnlyModel()

    y_pred, y_proba = predict_with_optional_proba(model, x_eval)

    assert y_pred.tolist() == [0, 0]
    assert y_proba is None


def test_predict_with_optional_proba_aligns_missing_class_column():
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    x_eval = np.array([[0.05, 0.02], [1.05, 1.02]], dtype=np.float32)

    model = build_baseline_model("logistic_regression", random_state=42)
    model = fit_baseline_model(model, x_train, y_train)

    _, y_proba = predict_with_optional_proba(model, x_eval, labels=(0, 1, 2))

    assert y_proba is not None
    assert y_proba.shape == (2, 3)
    assert np.allclose(y_proba[:, 2], 0.0)


def test_predict_with_optional_proba_x_eval_errors():
    model = _DummyPredictOnlyModel()

    with pytest.raises(BaselineModelError, match="двумерным"):
        predict_with_optional_proba(model, np.array([1.0, 2.0]))

    with pytest.raises(BaselineModelError, match="не должен быть пустым"):
        predict_with_optional_proba(model, np.empty((0, 2)))


def test_run_baseline_experiment_success_logistic_and_forest():
    x_train, y_train, x_eval, y_eval = _make_dataset()

    result_lr = run_baseline_experiment(
        model_name="logistic_regression",
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        labels=(0, 1, 2),
        random_state=42,
    )

    result_rf = run_baseline_experiment(
        model_name="random_forest",
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        labels=(0, 1, 2),
        random_state=42,
    )

    assert result_lr["model_name"] == "logistic_regression"
    assert result_rf["model_name"] == "random_forest"
    assert "macro_f1" in result_lr["metrics"]
    assert "macro_f1" in result_rf["metrics"]


def test_run_baseline_experiment_return_model_and_feature_importance(tmp_path):
    x_train, y_train, x_eval, y_eval = _make_dataset()

    result = run_baseline_experiment(
        model_name="logistic_regression",
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        labels=(0, 1, 2),
        random_state=42,
        return_model=True,
    )

    plot_path = tmp_path / "importance.png"
    importance = analyze_baseline_feature_importance(
        model=result["model"],
        feature_names=("f1", "f2"),
        labels=(0, 1, 2),
        save_path=plot_path,
        top_k=2,
    )

    assert plot_path.exists()
    assert "model" in result
    assert "y_proba" in result
    assert set(importance) == {"class_0", "class_1", "class_2"}


def test_analyze_baseline_feature_importance_errors(tmp_path):
    x_train, y_train, _, _ = _make_dataset()
    model = build_baseline_model("logistic_regression", random_state=42)
    model = fit_baseline_model(model, x_train, y_train)

    with pytest.raises(BaselineModelError, match="feature_names не должен быть пустым"):
        analyze_baseline_feature_importance(model, (), labels=(0, 1, 2), save_path=tmp_path / "a.png")

    with pytest.raises(BaselineModelError, match="Число feature_names"):
        analyze_baseline_feature_importance(model, ("f1",), labels=(0, 1, 2), save_path=tmp_path / "a.png")

    with pytest.raises(BaselineModelError, match="top_k"):
        analyze_baseline_feature_importance(model, ("f1", "f2"), labels=(0, 1, 2), save_path=tmp_path / "a.png", top_k=0)

    with pytest.raises(BaselineModelError, match="labels не должен быть пустым"):
        analyze_baseline_feature_importance(model, ("f1", "f2"), labels=(), save_path=tmp_path / "a.png")

    with pytest.raises(BaselineModelError, match="coef_"):
        analyze_baseline_feature_importance(_DummyPredictOnlyModel(), ("f1", "f2"), labels=(0, 1, 2), save_path=tmp_path / "a.png")


def test_analyze_baseline_feature_importance_binary_branch(tmp_path):
    x_train = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=np.float32)
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    model = build_baseline_model("logistic_regression", random_state=42)
    model = fit_baseline_model(model, x_train, y_train)

    plot_path = tmp_path / "binary_importance.png"
    result = analyze_baseline_feature_importance(
        model=model,
        feature_names=("f1", "f2"),
        labels=(0, 1, 2),
        save_path=plot_path,
        top_k=2,
    )

    assert plot_path.exists()
    assert len(result["class_0"]) == 2


def test_analyze_baseline_feature_importance_single_label_plot(tmp_path):
    x_train, y_train, _, _ = _make_dataset()
    model = build_baseline_model("logistic_regression", random_state=42)
    model = fit_baseline_model(model, x_train, y_train)

    plot_path = tmp_path / "single_label.png"
    result = analyze_baseline_feature_importance(
        model=model,
        feature_names=("f1", "f2"),
        labels=(1,),
        save_path=plot_path,
        top_k=2,
    )

    assert plot_path.exists()
    assert "class_1" in result


def test_run_baseline_experiment_with_missing_class_in_train():
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [1.0, 1.0],
            [1.1, 1.1],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    x_eval = np.array([[0.05, 0.05], [1.05, 1.05]], dtype=np.float32)
    y_eval = np.array([0, 1], dtype=np.int64)

    result = run_baseline_experiment(
        model_name="logistic_regression",
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        y_eval=y_eval,
        labels=(0, 1, 2),
        random_state=42,
    )

    assert result["metrics"]["labels"] == [0, 1, 2]
    assert "macro_f1" in result["metrics"]


def test_run_baseline_experiment_empty_labels_error():
    x_train, y_train, x_eval, y_eval = _make_dataset()

    with pytest.raises(BaselineModelError, match="labels не должен быть пустым"):
        run_baseline_experiment(
            model_name="logistic_regression",
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            labels=(),
            random_state=42,
        )

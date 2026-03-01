from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    PreprocessingError,
    fit_improved_preprocessor,
    fit_raw_preprocessor,
    prepare_improved_splits,
    prepare_raw_data,
    prepare_raw_splits,
    transform_improved_labeled,
    transform_improved_unlabeled,
    transform_raw_labeled,
    transform_raw_unlabeled,
)


def _make_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 100.0, None, 5.0],
            "f2": [10.0, None, 30.0, 40.0, 50.0, 60.0],
            "Class": [0, 1, 0, 2, 1, 0],
        }
    )


def test_prepare_raw_data_success_with_median_imputation():
    df = pd.DataFrame({"f1": [1.0, None], "f2": [3.0, 4.0], "Class": [0, 1]})
    x, y, features = prepare_raw_data(df, target_col="Class")

    assert x.shape == (2, 2)
    assert y.tolist() == [0, 1]
    assert features == ("f1", "f2")
    assert x.dtype == np.float32
    assert not np.isnan(x).any()


def test_prepare_raw_data_missing_target_error():
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    with pytest.raises(PreprocessingError, match="Отсутствует целевая колонка"):
        prepare_raw_data(df, target_col="Class")


def test_prepare_raw_data_no_features_error():
    df = pd.DataFrame({"Class": [0, 1]})
    with pytest.raises(PreprocessingError, match="Не найдены признаки"):
        prepare_raw_data(df, target_col="Class")


def test_prepare_raw_data_target_non_numeric_error():
    df = pd.DataFrame({"f1": [1.0, 2.0], "Class": ["x", "y"]})
    with pytest.raises(PreprocessingError, match="Целевая колонка должна быть числовой"):
        prepare_raw_data(df, target_col="Class")


def test_fit_raw_preprocessor_and_transform_success():
    train_df = _make_train_df()
    preprocessor = fit_raw_preprocessor(train_df, target_col="Class")

    val_df = pd.DataFrame({"f1": [2.0, None], "f2": [20.0, 70.0], "Class": [1, 0]})
    x_val, y_val = transform_raw_labeled(val_df, preprocessor, target_col="Class")
    x_unlabeled = transform_raw_unlabeled(pd.DataFrame({"f1": [2.0, None], "f2": [20.0, 70.0]}), preprocessor)

    assert x_val.shape == (2, 2)
    assert y_val.tolist() == [1, 0]
    assert x_unlabeled.shape == (2, 2)
    assert not np.isnan(x_val).any()
    assert not np.isnan(x_unlabeled).any()


def test_fit_raw_preprocessor_all_missing_feature_error():
    df = pd.DataFrame({"f1": [None, None], "Class": [0, 1]})
    with pytest.raises(PreprocessingError, match="Невозможно вычислить медианы"):
        fit_raw_preprocessor(df, target_col="Class")


def test_transform_raw_feature_mismatch_error():
    train_df = _make_train_df()
    preprocessor = fit_raw_preprocessor(train_df, target_col="Class")

    bad_df = pd.DataFrame({"f1": [1.0], "f3": [2.0], "Class": [0]})
    with pytest.raises(PreprocessingError, match="Несовпадение признаков"):
        transform_raw_labeled(bad_df, preprocessor, target_col="Class")

    bad_unlabeled = pd.DataFrame({"f1": [1.0], "Class": [0]})
    with pytest.raises(PreprocessingError, match="Несовпадение признаков"):
        transform_raw_unlabeled(bad_unlabeled, preprocessor)


def test_prepare_raw_splits_success():
    train_df = _make_train_df()
    val_df = pd.DataFrame({"f1": [2.0, None], "f2": [20.0, 70.0], "Class": [1, 0]})
    test_df = pd.DataFrame({"f1": [1.5, 7.0], "f2": [15.0, 80.0], "Class": [0, 2]})

    result = prepare_raw_splits(train_df, val_df, test_df, target_col="Class")
    assert result["train"][0].shape == (6, 2)
    assert result["val"][0].shape == (2, 2)
    assert result["test"][0].shape == (2, 2)


def test_fit_improved_preprocessor_quantile_errors():
    train_df = _make_train_df()

    with pytest.raises(PreprocessingError, match="кортежем"):
        fit_improved_preprocessor(train_df, target_col="Class", clip_quantiles=[0.1, 0.9])  # type: ignore[arg-type]

    with pytest.raises(PreprocessingError, match="содержать числа"):
        fit_improved_preprocessor(train_df, target_col="Class", clip_quantiles=("0.1", 0.9))  # type: ignore[arg-type]

    with pytest.raises(PreprocessingError, match="0 <= lower < upper <= 1"):
        fit_improved_preprocessor(train_df, target_col="Class", clip_quantiles=(0.9, 0.1))


def test_fit_improved_preprocessor_missing_target_error():
    with pytest.raises(PreprocessingError, match="Отсутствует целевая колонка"):
        fit_improved_preprocessor(pd.DataFrame({"f1": [1.0]}), target_col="Class")


def test_fit_improved_preprocessor_all_missing_feature_error():
    df = pd.DataFrame({"f1": [None, None], "Class": [0, 1]})
    with pytest.raises(PreprocessingError, match="Невозможно вычислить медианы"):
        fit_improved_preprocessor(df, target_col="Class")


def test_transform_improved_labeled_success_and_robust_centering():
    train_df = _make_train_df()
    val_df = pd.DataFrame({"f1": [2.0, 6.0], "f2": [20.0, 70.0], "Class": [1, 0]})

    preprocessor = fit_improved_preprocessor(train_df, target_col="Class")
    x_train, _ = transform_improved_labeled(train_df, preprocessor, target_col="Class")
    x_val, y_val = transform_improved_labeled(val_df, preprocessor, target_col="Class")

    assert x_val.shape == (2, 2)
    assert y_val.tolist() == [1, 0]
    assert float(np.abs(np.median(x_train, axis=0)).max()) < 1e-5


def test_transform_improved_labeled_feature_mismatch_error():
    train_df = _make_train_df()
    preprocessor = fit_improved_preprocessor(train_df, target_col="Class")

    bad_df = pd.DataFrame({"f1": [1.0], "f3": [2.0], "Class": [0]})
    with pytest.raises(PreprocessingError, match="Несовпадение признаков"):
        transform_improved_labeled(bad_df, preprocessor, target_col="Class")


def test_transform_improved_unlabeled_feature_mismatch_error():
    train_df = _make_train_df()
    preprocessor = fit_improved_preprocessor(train_df, target_col="Class")

    bad_df = pd.DataFrame({"f1": [1.0], "Class": [0]})
    with pytest.raises(PreprocessingError, match="Несовпадение признаков"):
        transform_improved_unlabeled(bad_df, preprocessor)


def test_prepare_improved_splits_success():
    train_df = _make_train_df()
    val_df = pd.DataFrame({"f1": [2.0, 6.0], "f2": [20.0, 70.0], "Class": [1, 0]})
    test_df = pd.DataFrame({"f1": [1.5, 7.0], "f2": [15.0, 80.0], "Class": [0, 2]})

    result = prepare_improved_splits(train_df, val_df, test_df, target_col="Class", clip_quantiles=(0.01, 0.99))

    x_train, y_train = result["train"]
    x_val, y_val = result["val"]
    x_test, y_test = result["test"]

    assert x_train.shape[1] == 2
    assert x_val.shape[1] == 2
    assert x_test.shape[1] == 2
    assert y_train.shape[0] == len(train_df)
    assert y_val.shape[0] == len(val_df)
    assert y_test.shape[0] == len(test_df)
    assert float(np.abs(np.median(x_train, axis=0)).max()) < 1e-5


def test_transform_improved_unlabeled_success():
    train_df = _make_train_df()
    preprocessor = fit_improved_preprocessor(train_df, target_col="Class")

    unlabeled_df = pd.DataFrame({"f1": [2.0, 3.0], "f2": [20.0, 30.0]})
    x = transform_improved_unlabeled(unlabeled_df, preprocessor)

    assert x.shape == (2, 2)
    assert x.dtype == np.float32

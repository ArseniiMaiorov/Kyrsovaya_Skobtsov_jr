from __future__ import annotations

import pandas as pd
import pytest

from src.data.eda import (
    EDAError,
    build_eda_summary,
    get_basic_statistics,
    get_class_distribution,
    get_constant_features,
    get_high_correlation_pairs,
    get_missing_summary,
    get_numeric_feature_names,
    get_outlier_share_iqr,
    get_placeholder_counts,
    get_top_missing_features,
)


def test_get_numeric_feature_names_excludes_target():
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "Class": [0, 1], "txt": ["a", "b"]})
    result = get_numeric_feature_names(df, exclude=("Class",))
    assert result == ["f1", "f2"]


def test_get_missing_summary_non_empty_and_empty():
    df = pd.DataFrame({"a": [1, None], "b": [None, None]})
    summary = get_missing_summary(df)
    assert summary["total_missing"] == 3
    assert summary["per_column"]["a"] == 1
    assert summary["per_column"]["b"] == 2
    assert summary["per_column_share"]["a"] == 0.5

    empty_summary = get_missing_summary(pd.DataFrame())
    assert empty_summary["total_missing"] == 0
    assert empty_summary["missing_share"] == 0.0
    assert empty_summary["per_column_share"] == {}


def test_get_class_distribution_success():
    df = pd.DataFrame({"Class": [2, 0, 2, 1, 0]})
    assert get_class_distribution(df, "Class") == {0: 2, 1: 1, 2: 2}


def test_get_class_distribution_missing_column_error():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(EDAError, match="не найдена"):
        get_class_distribution(df, "Class")


def test_get_outlier_share_iqr_branches():
    df = pd.DataFrame(
        {
            "empty_col": [None, None, None, None, None, None],
            "constant_col": [5, 5, 5, 5, 5, 5],
            "normal_col": [1, 2, 2, 2, 3, 100],
        }
    )

    result = get_outlier_share_iqr(df, ["empty_col", "constant_col", "normal_col"])
    assert result["empty_col"] == 0.0
    assert result["constant_col"] == 0.0
    assert result["normal_col"] > 0.0


def test_get_top_missing_features_success_and_validation():
    df = pd.DataFrame({"f1": [1.0, None, None], "f2": [1.0, 2.0, None], "f3": [1.0, 2.0, 3.0]})
    top_features = get_top_missing_features(df, ["f1", "f2", "f3"], top_n=2)

    assert top_features[0][0] == "f1"
    assert top_features[0][1] == pytest.approx(2 / 3)
    assert top_features[0][2] == 2
    assert get_top_missing_features(df, [], top_n=2) == []

    with pytest.raises(EDAError, match="top_n"):
        get_top_missing_features(df, ["f1"], top_n=0)


def test_get_basic_statistics_placeholder_counts_and_constant_features():
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [None, None, None],
            "f3": [-999.0, -9999.0, 5.0],
            "f4": [7.0, 7.0, 7.0],
        }
    )

    stats = get_basic_statistics(df, ["f1", "f2"])
    placeholders = get_placeholder_counts(df, ["f3"])
    constant_features = get_constant_features(df, ["f1", "f2", "f4"])

    assert stats["f1"]["mean"] == 2.0
    assert stats["f2"]["mean"] is None
    assert placeholders["f3"]["-999"] == 1
    assert placeholders["f3"]["-9999"] == 1
    assert constant_features == ["f2", "f4"]


def test_get_high_correlation_pairs_validation_errors():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})

    with pytest.raises(EDAError, match="Порог корреляции"):
        get_high_correlation_pairs(df, ["a", "b"], threshold=1.5)

    with pytest.raises(EDAError, match="max_pairs"):
        get_high_correlation_pairs(df, ["a", "b"], max_pairs=0)


def test_get_high_correlation_pairs_empty_columns():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert get_high_correlation_pairs(df, []) == []


def test_get_high_correlation_pairs_success():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "z": [5, 4, 3, 2, 1],
        }
    )
    pairs = get_high_correlation_pairs(df, ["x", "y", "z"], threshold=0.95, max_pairs=5)
    assert len(pairs) >= 1
    assert any(left == "x" and right == "y" for left, right, _ in pairs)


def test_build_eda_summary_success():
    labeled_df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, None, 100.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
            "Class": [0, 1, 0, 2],
        }
    )
    unlabeled_df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, -999.0]})

    summary = build_eda_summary(labeled_df, unlabeled_df, target_col="Class")
    assert summary["labeled_shape"] == [4, 3]
    assert summary["unlabeled_shape"] == [2, 2]
    assert summary["numeric_feature_count"] == 2
    assert summary["class_distribution"] == {0: 2, 1: 1, 2: 1}
    assert summary["class_imbalance_ratio"] == 2.0
    assert "labeled_missing" in summary
    assert "unlabeled_missing" in summary
    assert "labeled_top_missing_features" in summary
    assert "unlabeled_top_missing_features" in summary
    assert "labeled_basic_statistics" in summary
    assert "unlabeled_basic_statistics" in summary
    assert "labeled_placeholder_counts" in summary
    assert "unlabeled_placeholder_counts" in summary
    assert "constant_features" in summary
    assert "top_outlier_features" in summary
    assert "high_correlation_pairs" in summary

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.rolling_validation import (
    RollingValidationError,
    _compute_split_counts,
    build_train_rolling_window_plan,
    materialize_rolling_fold,
    save_rolling_plan_artifact,
    validate_rolling_no_index_leakage,
)


def _make_single_series_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(24):
        rows.append(
            {
                "time": 100 + idx,
                "f1": float(idx),
                "f2": float(idx) / 10.0,
                "Class": idx % 3,
            }
        )
    return pd.DataFrame(rows)


def _make_multi_series_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for series_id, offset in (("A", 0), ("B", 100)):
        for idx in range(24):
            rows.append(
                {
                    "session": series_id,
                    "time": offset + (24 - idx),
                    "f1": float(offset + idx),
                    "f2": float(offset + idx) / 10.0,
                    "Class": idx % 3,
                }
            )
    return pd.DataFrame(rows)


def test_compute_split_counts_rebalances_zero_segments():
    assert _compute_split_counts(3, 0.7, 0.15, 0.15) == (1, 1, 1)


def test_compute_split_counts_too_short_error():
    with pytest.raises(RollingValidationError, match="минимум 3 строки"):
        _compute_split_counts(2, 0.7, 0.15, 0.15)


def test_build_train_rolling_window_plan_success_without_time_and_series():
    df = pd.DataFrame(
        {
            "f1": [float(idx) for idx in range(24)],
            "f2": [float(idx) / 10.0 for idx in range(24)],
            "Class": [idx % 3 for idx in range(24)],
        }
    )

    plan = build_train_rolling_window_plan(
        df=df,
        target_col="Class",
        window_size=2,
        stride=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        n_folds=3,
    )

    assert plan["time_column"] is None
    assert plan["series_column"] is None
    assert len(plan["folds"]) == 3
    assert len(plan["folds"][0]["train_row_positions_for_fit"]) < len(plan["folds"][1]["train_row_positions_for_fit"])
    validate_rolling_no_index_leakage(plan)


def test_build_train_rolling_window_plan_success_with_time_and_series():
    df = _make_multi_series_df()

    plan = build_train_rolling_window_plan(
        df=df,
        target_col="Class",
        window_size=2,
        stride=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        n_folds=3,
    )

    assert plan["time_column"] == "time"
    assert plan["series_column"] == "session"
    assert len(plan["folds"][0]["series_summaries"]) == 2
    first_record = plan["folds"][0]["train"][0]
    assert first_record["row_positions"] == [23, 22]


def test_build_train_rolling_window_plan_validation_errors():
    df = _make_single_series_df()

    with pytest.raises(RollingValidationError, match="Отсутствует целевая колонка"):
        build_train_rolling_window_plan(df.drop(columns=["Class"]), "Class", 2, 1, 0.7, 0.15, 0.15)

    with pytest.raises(RollingValidationError, match="window_size"):
        build_train_rolling_window_plan(df, "Class", 0, 1, 0.7, 0.15, 0.15)

    with pytest.raises(RollingValidationError, match="stride"):
        build_train_rolling_window_plan(df, "Class", 2, 0, 0.7, 0.15, 0.15)

    with pytest.raises(RollingValidationError, match="n_folds"):
        build_train_rolling_window_plan(df, "Class", 2, 1, 0.7, 0.15, 0.15, n_folds=0)

    with pytest.raises(RollingValidationError, match="train_ratio"):
        build_train_rolling_window_plan(df, "Class", 2, 1, 0.0, 0.5, 0.5)

    with pytest.raises(RollingValidationError, match="Сумма train_ratio"):
        build_train_rolling_window_plan(df, "Class", 2, 1, 0.7, 0.2, 0.2)


def test_build_train_rolling_window_plan_short_train_segment_error():
    df = pd.DataFrame(
        {
            "time": list(range(10)),
            "f1": [float(idx) for idx in range(10)],
            "f2": [float(idx) / 10.0 for idx in range(10)],
            "Class": [idx % 3 for idx in range(10)],
        }
    )

    with pytest.raises(RollingValidationError, match="слишком короткий"):
        build_train_rolling_window_plan(
            df=df,
            target_col="Class",
            window_size=3,
            stride=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            n_folds=3,
        )


def test_materialize_rolling_fold_success():
    df = _make_single_series_df()
    plan = build_train_rolling_window_plan(df, "Class", 2, 1, 0.7, 0.15, 0.15, n_folds=3)
    features = df[["f1", "f2"]].to_numpy(dtype=np.float32)
    targets = df["Class"].to_numpy(dtype=np.int64)

    splits = materialize_rolling_fold(features, targets, plan["folds"][0], window_size=2)

    assert splits["train"][0].shape[1:] == (2, 2)
    assert splits["train"][1].ndim == 1
    assert splits["val"][0].shape[1:] == (2, 2)


def test_materialize_rolling_fold_validation_errors_and_empty_branch():
    fold_plan = {"train": [{"row_positions": [0, 1], "target_position": 1}], "val": [{"row_positions": [2, 3], "target_position": 3}]}
    features = np.ones((4, 2), dtype=np.float32)
    targets = np.array([0, 1, 2, 0], dtype=np.int64)

    with pytest.raises(RollingValidationError, match="двумерной"):
        materialize_rolling_fold(np.array([1.0, 2.0]), targets, fold_plan, window_size=2)

    with pytest.raises(RollingValidationError, match="не должна быть пустой"):
        materialize_rolling_fold(np.empty((0, 2), dtype=np.float32), targets[:0], fold_plan, window_size=2)

    with pytest.raises(RollingValidationError, match="одномерным"):
        materialize_rolling_fold(features, np.ones((4, 1), dtype=np.int64), fold_plan, window_size=2)

    with pytest.raises(RollingValidationError, match="должно совпадать"):
        materialize_rolling_fold(features, targets[:3], fold_plan, window_size=2)

    empty_splits = materialize_rolling_fold(features, targets, {"train": [], "val": []}, window_size=2)
    assert empty_splits["train"][0].shape == (0, 2, 2)
    assert empty_splits["val"][1].shape == (0,)


def test_validate_rolling_no_index_leakage_error():
    plan = {
        "folds": [
            {
                "fold_index": 1,
                "train": [{"row_positions": [0, 1], "target_position": 1}],
                "val": [{"row_positions": [1, 2], "target_position": 2}],
            }
        ]
    }

    with pytest.raises(RollingValidationError, match="Утечка данных"):
        validate_rolling_no_index_leakage(plan)


def test_save_rolling_plan_artifact(tmp_path):
    plan = {"n_folds": 3, "folds": []}
    path = tmp_path / "nested" / "rolling.json"

    save_rolling_plan_artifact(plan, path)

    assert json.loads(path.read_text(encoding="utf-8")) == plan

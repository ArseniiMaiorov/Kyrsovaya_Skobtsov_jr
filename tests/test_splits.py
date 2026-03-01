from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.splits import (
    DataSplitError,
    build_inference_window_plan,
    build_split_report,
    build_window_split_plan,
    detect_series_column,
    detect_time_column,
    flatten_windows,
    get_class_distribution,
    materialize_labeled_window_splits,
    materialize_unlabeled_windows,
    save_split_artifact,
    validate_no_index_leakage,
)


def _make_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    times_a = [5, 1, 3, 2, 4, 6, 9, 7, 8, 10, 12, 11, 14, 13]
    times_b = [105, 101, 103, 102, 104, 106, 109, 107, 108, 110, 112, 111, 114, 113]

    for idx, time_value in enumerate(times_a):
        rows.append(
            {
                "session": "A",
                "time": time_value,
                "f1": float(idx + 1),
                "f2": float(idx) / 10.0,
                "Class": [0, 0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2][idx],
            }
        )

    for idx, time_value in enumerate(times_b):
        rows.append(
            {
                "session": "B",
                "time": time_value,
                "f1": float(idx + 21),
                "f2": 2.0 + float(idx) / 10.0,
                "Class": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1][idx],
            }
        )

    return pd.DataFrame(rows)


def test_detect_time_and_series_column():
    cols = ["abc", "Time", "series_id"]
    assert detect_time_column(cols) == "Time"
    assert detect_series_column(cols) == "series_id"
    assert detect_time_column(["abc"]) is None
    assert detect_series_column(["abc"]) is None


def test_build_window_split_plan_success_and_sorting():
    df = _make_df()
    plan = build_window_split_plan(
        df,
        target_col="Class",
        window_size=2,
        stride=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert plan["time_column"] == "time"
    assert plan["series_column"] == "session"
    assert len(plan["series_summaries"]) == 2
    assert plan["train_row_positions_for_fit"]

    first_train = plan["splits"]["train"][0]
    # После сортировки по time для серии A первые 2 строки должны соответствовать времени 1 и 2.
    assert first_train["row_positions"] == [1, 3]

    validate_no_index_leakage(plan)
    train_rows = {pos for record in plan["splits"]["train"] for pos in record["row_positions"]}
    val_rows = {pos for record in plan["splits"]["val"] for pos in record["row_positions"]}
    test_rows = {pos for record in plan["splits"]["test"] for pos in record["row_positions"]}
    assert not train_rows.intersection(val_rows)
    assert not train_rows.intersection(test_rows)
    assert not val_rows.intersection(test_rows)


def test_build_window_split_plan_builds_non_empty_splits_for_long_enough_series():
    df = pd.DataFrame(
        {
            "f1": [float(idx) for idx in range(14)],
            "f2": [float(idx) / 10.0 for idx in range(14)],
            "Class": [idx % 3 for idx in range(14)],
        }
    )
    plan = build_window_split_plan(
        df,
        target_col="Class",
        window_size=2,
        stride=1,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert len(plan["splits"]["train"]) > 0
    assert len(plan["splits"]["val"]) > 0
    assert len(plan["splits"]["test"]) > 0


def test_build_window_split_plan_missing_target_error():
    with pytest.raises(DataSplitError, match="Отсутствует целевая колонка"):
        build_window_split_plan(
            pd.DataFrame({"f1": [1, 2, 3]}),
            target_col="Class",
            window_size=2,
            stride=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"window_size": 1, "stride": 1, "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15, "random_state": 42}, "window_size"),
        ({"window_size": 2, "stride": 0, "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15, "random_state": 42}, "stride"),
        ({"window_size": 2, "stride": 1, "train_ratio": 0.0, "val_ratio": 0.5, "test_ratio": 0.5, "random_state": 42}, "train_ratio"),
        ({"window_size": 2, "stride": 1, "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.2, "random_state": 42}, "Сумма train_ratio"),
        ({"window_size": 2, "stride": 1, "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15, "random_state": -1}, "random_state"),
    ],
)
def test_build_window_split_plan_param_errors(kwargs, error):
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [1.0, 2.0, 3.0, 4.0], "Class": [0, 1, 2, 0]})
    with pytest.raises(DataSplitError, match=error):
        build_window_split_plan(df, target_col="Class", **kwargs)


def test_build_window_split_plan_short_series_error():
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0], "Class": [0]})
    with pytest.raises(DataSplitError, match="слишком короткая"):
        build_window_split_plan(
            df,
            target_col="Class",
            window_size=2,
            stride=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )


def test_build_window_split_plan_not_enough_windows_error():
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [0.1, 0.2], "Class": [0, 1]})
    with pytest.raises(DataSplitError, match="Недостаточно элементов"):
        build_window_split_plan(
            df,
            target_col="Class",
            window_size=2,
            stride=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )


def test_build_window_split_plan_split_segment_too_short_error():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [0.1, 0.2, 0.3], "Class": [0, 1, 2]})
    with pytest.raises(DataSplitError, match="непересекающегося split"):
        build_window_split_plan(
            df,
            target_col="Class",
            window_size=2,
            stride=1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )


def test_materialize_labeled_window_splits_success_and_report():
    df = _make_df()
    plan = build_window_split_plan(df, "Class", 2, 1, 0.7, 0.15, 0.15, 42)
    features = df[["f1", "f2"]].to_numpy(dtype=np.float32)
    targets = df["Class"].to_numpy(dtype=np.int64)

    splits = materialize_labeled_window_splits(features, targets, plan)
    report = build_split_report(splits)

    assert splits["train"][0].ndim == 3
    assert splits["train"][0].shape[2] == 2
    assert report["train_shape"][1] == 2
    assert "train_class_distribution" in report


def test_materialize_labeled_window_splits_validation_errors():
    plan = {
        "window_size": 2,
        "splits": {
            "train": [{"row_positions": [0, 1], "target_position": 1}],
            "val": [{"row_positions": [1, 2], "target_position": 2}],
            "test": [{"row_positions": [2, 3], "target_position": 3}],
        },
    }

    with pytest.raises(DataSplitError, match="двумерной"):
        materialize_labeled_window_splits(np.array([1.0, 2.0]), np.array([0, 1]), plan)

    with pytest.raises(DataSplitError, match="одномерным"):
        materialize_labeled_window_splits(np.ones((4, 2), dtype=np.float32), np.ones((4, 1), dtype=np.int64), plan)

    with pytest.raises(DataSplitError, match="должно совпадать"):
        materialize_labeled_window_splits(np.ones((3, 2), dtype=np.float32), np.ones((4,), dtype=np.int64), plan)


def test_materialize_labeled_window_splits_empty_branch():
    plan = {
        "window_size": 2,
        "splits": {"train": [], "val": [], "test": []},
    }
    features = np.ones((4, 2), dtype=np.float32)
    targets = np.array([0, 1, 2, 0], dtype=np.int64)

    splits = materialize_labeled_window_splits(features, targets, plan)
    assert splits["train"][0].shape == (0, 2, 2)
    assert splits["train"][1].shape == (0,)


def test_build_inference_window_plan_and_materialize_success():
    df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [5.0, 6.0, 7.0, 8.0]})
    plan = build_inference_window_plan(df, window_size=2, stride=1)
    x = materialize_unlabeled_windows(df.to_numpy(dtype=np.float32), plan)

    assert len(plan["windows"]) == 3
    assert x.shape == (3, 2, 2)


def test_build_inference_window_plan_param_and_empty_errors():
    df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
    with pytest.raises(DataSplitError, match="window_size"):
        build_inference_window_plan(df, window_size=1, stride=1)
    with pytest.raises(DataSplitError, match="stride"):
        build_inference_window_plan(df, window_size=2, stride=0)
    with pytest.raises(DataSplitError, match="Не удалось построить ни одного окна"):
        build_inference_window_plan(df, window_size=2, stride=1)


def test_materialize_unlabeled_windows_validation_and_empty_branch():
    plan = {"window_size": 2, "windows": []}
    with pytest.raises(DataSplitError, match="двумерной"):
        materialize_unlabeled_windows(np.array([1.0, 2.0]), plan)
    with pytest.raises(DataSplitError, match="не должна быть пустой"):
        materialize_unlabeled_windows(np.empty((0, 2), dtype=np.float32), plan)

    x = materialize_unlabeled_windows(np.ones((4, 2), dtype=np.float32), plan)
    assert x.shape == (0, 2, 2)


def test_get_class_distribution_success_and_errors():
    targets = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    assert get_class_distribution(targets) == {0: 2, 1: 1, 2: 2}

    with pytest.raises(DataSplitError, match="одномерным"):
        get_class_distribution(np.array([[0], [1]], dtype=np.int64))
    with pytest.raises(DataSplitError, match="не должен быть пустым"):
        get_class_distribution(np.empty((0,), dtype=np.int64))


def test_flatten_windows_success_and_errors():
    x = np.ones((3, 2, 4), dtype=np.float32)
    flat = flatten_windows(x)
    assert flat.shape == (3, 8)

    with pytest.raises(DataSplitError, match="размерности"):
        flatten_windows(np.ones((3, 8), dtype=np.float32))
    with pytest.raises(DataSplitError, match="не должен быть пустым"):
        flatten_windows(np.empty((0, 2, 4), dtype=np.float32))


def test_validate_no_index_leakage_error():
    plan = {
        "splits": {
            "train": [{"row_positions": [0, 1], "target_position": 1}],
            "val": [{"row_positions": [1, 2], "target_position": 2}],
            "test": [{"row_positions": [3, 4], "target_position": 4}],
        }
    }
    with pytest.raises(DataSplitError, match="train и val"):
        validate_no_index_leakage(plan)


def test_validate_no_index_leakage_train_test_error():
    plan = {
        "splits": {
            "train": [{"row_positions": [2, 3], "target_position": 3}],
            "val": [{"row_positions": [4, 5], "target_position": 5}],
            "test": [{"row_positions": [3, 6], "target_position": 6}],
        }
    }
    with pytest.raises(DataSplitError, match="train и test"):
        validate_no_index_leakage(plan)


def test_validate_no_index_leakage_val_test_error():
    plan = {
        "splits": {
            "train": [{"row_positions": [0, 1], "target_position": 1}],
            "val": [{"row_positions": [3, 4], "target_position": 4}],
            "test": [{"row_positions": [4, 5], "target_position": 5}],
        }
    }
    with pytest.raises(DataSplitError, match="val и test"):
        validate_no_index_leakage(plan)


def test_save_split_artifact(tmp_path):
    plan = {
        "window_size": 2,
        "stride": 1,
        "random_state": 42,
        "time_column": None,
        "series_column": None,
        "series_summaries": [],
        "splits": {"train": [], "val": [], "test": []},
        "train_row_positions_for_fit": [],
    }
    path = tmp_path / "split.json"
    save_split_artifact(plan, str(path))

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["window_size"] == 2

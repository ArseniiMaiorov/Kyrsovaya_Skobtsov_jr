"""Дополнительная rolling-валидация внутри официального train-сегмента."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.splits import detect_series_column, detect_time_column


class RollingValidationError(ValueError):
    """Исключение для ошибок построения rolling-диагностики."""


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise RollingValidationError(f"{field_name} должен быть положительным целым числом")
    return int(value)


def _require_ratio(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or not 0 < value < 1:
        raise RollingValidationError(f"{field_name} должен быть числом в диапазоне (0, 1)")
    return float(value)


def _compute_split_counts(total_rows: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total_rows < 3:
        raise RollingValidationError("Для официального split нужно минимум 3 строки")

    train_rows = int(math.floor(total_rows * train_ratio))
    val_rows = int(math.floor(total_rows * val_ratio))
    test_rows = total_rows - train_rows - val_rows
    counts = [train_rows, val_rows, test_rows]

    for idx in range(3):
        if counts[idx] > 0:
            continue
        donor_idx = max(range(3), key=lambda item: counts[item])
        counts[donor_idx] -= 1
        counts[idx] += 1

    return counts[0], counts[1], counts[2]


def _compute_block_sizes(total_rows: int, n_blocks: int, min_block_size: int) -> list[int]:
    if total_rows < n_blocks * min_block_size:
        raise RollingValidationError(
            "Официальный train-сегмент слишком короткий для rolling-валидации: "
            f"нужно минимум {n_blocks * min_block_size} строк, получено {total_rows}"
        )

    block_sizes = [min_block_size] * n_blocks
    remainder = total_rows - (n_blocks * min_block_size)
    for idx in range(remainder):
        block_sizes[idx % n_blocks] += 1
    return block_sizes


def _iter_ordered_series(df: pd.DataFrame, time_col: str | None, series_col: str | None) -> list[tuple[str, pd.DataFrame]]:
    if series_col is None:
        groups = [("__single_series__", df)]
    else:
        groups = [(str(series_id), series_df) for series_id, series_df in df.groupby(series_col, sort=False)]

    ordered_groups: list[tuple[str, pd.DataFrame]] = []
    for series_id, series_df in groups:
        if time_col is None:
            ordered_groups.append((series_id, series_df))
        else:
            ordered_groups.append((series_id, series_df.sort_values(by=time_col, kind="stable")))
    return ordered_groups


def _build_window_records(series_id: str, row_positions: list[int], window_size: int, stride: int, split_name: str) -> list[dict[str, Any]]:
    starts = list(range(0, len(row_positions) - window_size + 1, stride))
    return [
        {
            "series_id": series_id,
            "split": split_name,
            "row_positions": row_positions[start : start + window_size],
            "target_position": int(row_positions[start + window_size - 1]),
            "start_position": int(row_positions[start]),
            "end_position": int(row_positions[start + window_size - 1]),
        }
        for start in starts
    ]


def build_train_rolling_window_plan(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Строит expanding-window rolling-фолды внутри официального train-сегмента."""
    if target_col not in df.columns:
        raise RollingValidationError(f"Отсутствует целевая колонка: '{target_col}'")

    window = _require_positive_int(window_size, "window_size")
    step = _require_positive_int(stride, "stride")
    folds_count = _require_positive_int(n_folds, "n_folds")
    train_share = _require_ratio(train_ratio, "train_ratio")
    val_share = _require_ratio(val_ratio, "val_ratio")
    test_share = _require_ratio(test_ratio, "test_ratio")

    if not math.isclose(train_share + val_share + test_share, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RollingValidationError("Сумма train_ratio, val_ratio и test_ratio должна быть равна 1")

    time_col = detect_time_column(df.columns)
    series_col = detect_series_column(df.columns)
    fold_entries = [
        {
            "fold_index": fold_idx + 1,
            "train": [],
            "val": [],
            "train_row_positions_for_fit": set(),
            "val_row_positions": set(),
            "series_summaries": [],
        }
        for fold_idx in range(folds_count)
    ]

    for series_id, ordered_df in _iter_ordered_series(df, time_col=time_col, series_col=series_col):
        ordered_positions = [int(pos) for pos in ordered_df.index.tolist()]
        train_rows, _, _ = _compute_split_counts(len(ordered_positions), train_share, val_share, test_share)
        official_train_positions = ordered_positions[:train_rows]
        block_sizes = _compute_block_sizes(train_rows, n_blocks=folds_count + 1, min_block_size=window)

        boundaries = [0]
        for block_size in block_sizes:
            boundaries.append(boundaries[-1] + block_size)

        for fold_idx, fold_entry in enumerate(fold_entries):
            train_end = boundaries[fold_idx + 1]
            val_left = boundaries[fold_idx + 1]
            val_right = boundaries[fold_idx + 2]
            fold_train_positions = official_train_positions[:train_end]
            fold_val_positions = official_train_positions[val_left:val_right]

            train_records = _build_window_records(series_id, fold_train_positions, window, step, "train")
            val_records = _build_window_records(series_id, fold_val_positions, window, step, "val")

            fold_entry["train"].extend(train_records)
            fold_entry["val"].extend(val_records)
            fold_entry["train_row_positions_for_fit"].update(fold_train_positions)
            fold_entry["val_row_positions"].update(fold_val_positions)
            fold_entry["series_summaries"].append(
                {
                    "series_id": series_id,
                    "official_train_rows": train_rows,
                    "fold_train_rows": len(fold_train_positions),
                    "fold_val_rows": len(fold_val_positions),
                    "fold_train_windows": len(train_records),
                    "fold_val_windows": len(val_records),
                }
            )

    normalized_folds: list[dict[str, Any]] = []
    for fold_entry in fold_entries:
        normalized_folds.append(
            {
                "fold_index": int(fold_entry["fold_index"]),
                "train": list(fold_entry["train"]),
                "val": list(fold_entry["val"]),
                "train_row_positions_for_fit": sorted(int(pos) for pos in fold_entry["train_row_positions_for_fit"]),
                "val_row_positions": sorted(int(pos) for pos in fold_entry["val_row_positions"]),
                "series_summaries": list(fold_entry["series_summaries"]),
            }
        )

    return {
        "window_size": window,
        "stride": step,
        "n_folds": folds_count,
        "time_column": time_col,
        "series_column": series_col,
        "folds": normalized_folds,
    }


def _validate_feature_rows(features: np.ndarray) -> None:
    if features.ndim != 2:
        raise RollingValidationError("Матрица признаков должна быть двумерной")
    if features.shape[0] == 0:
        raise RollingValidationError("Матрица признаков не должна быть пустой")


def materialize_rolling_fold(
    features: np.ndarray,
    targets: np.ndarray,
    fold_plan: dict[str, Any],
    window_size: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Преобразует один rolling-фолд в массивы `X/y` для train и val."""
    _validate_feature_rows(features)
    if targets.ndim != 1:
        raise RollingValidationError("Вектор targets должен быть одномерным")
    if len(targets) != features.shape[0]:
        raise RollingValidationError("Количество строк в features должно совпадать с длиной targets")

    n_features = int(features.shape[1])
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for split_name in ("train", "val"):
        records = list(fold_plan[split_name])
        x_list = [features[np.asarray(record["row_positions"], dtype=np.int64)] for record in records]
        y_list = [int(targets[int(record["target_position"])]) for record in records]
        if x_list:
            x_split = np.stack(x_list, axis=0).astype(np.float32)
            y_split = np.asarray(y_list, dtype=np.int64)
        else:
            x_split = np.empty((0, window_size, n_features), dtype=np.float32)
            y_split = np.empty((0,), dtype=np.int64)
        result[split_name] = (x_split, y_split)

    return result


def validate_rolling_no_index_leakage(plan: dict[str, Any]) -> None:
    """Проверяет, что в каждом rolling-фолде строки train и val не пересекаются."""
    for fold_plan in plan["folds"]:
        train_rows = {int(pos) for record in fold_plan["train"] for pos in record["row_positions"]}
        val_rows = {int(pos) for record in fold_plan["val"] for pos in record["row_positions"]}
        if train_rows.intersection(val_rows):
            raise RollingValidationError(
                f"Утечка данных в fold {fold_plan['fold_index']}: пересечение строк между train и val"
            )


def save_rolling_plan_artifact(plan: dict[str, Any], path: str | Path) -> None:
    """Сохраняет JSON-артефакт с планом rolling-валидации."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as file_obj:
        json.dump(plan, file_obj, ensure_ascii=False, indent=2)

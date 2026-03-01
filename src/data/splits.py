"""Построение временных окон и time-based split без перемешивания."""

from __future__ import annotations

import json
from typing import Any, Iterable

import math
import numpy as np
import pandas as pd

TIME_COLUMN_CANDIDATES = (
    "timestamp",
    "time",
    "datetime",
    "date",
)

SERIES_COLUMN_CANDIDATES = (
    "session",
    "pass",
    "orbit",
    "track",
    "series_id",
)

SPLIT_NAMES = ("train", "val", "test")


class DataSplitError(ValueError):
    """Исключение для ошибок формирования окон и разбиения."""


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def detect_time_column(columns: Iterable[str]) -> str | None:
    """Определяет имя временной колонки по эвристике, если она есть."""
    normalized = {_normalize_name(name): name for name in columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def detect_series_column(columns: Iterable[str]) -> str | None:
    """Определяет имя колонки серии/сеанса по эвристике, если она есть."""
    normalized = {_normalize_name(name): name for name in columns}
    for candidate in SERIES_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _validate_split_params(
    window_size: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> None:
    if not isinstance(window_size, int) or window_size <= 1:
        raise DataSplitError("window_size должен быть целым числом больше 1")
    if not isinstance(stride, int) or stride <= 0:
        raise DataSplitError("stride должен быть положительным целым числом")

    for name, value in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if not isinstance(value, (int, float)) or not 0 < value < 1:
            raise DataSplitError(f"{name} должен быть числом в диапазоне (0, 1)")

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise DataSplitError("Сумма train_ratio, val_ratio и test_ratio должна быть равна 1")

    if not isinstance(random_state, int) or random_state < 0:
        raise DataSplitError("random_state должен быть неотрицательным целым числом")


def _compute_split_counts(total_items: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if total_items < 3:
        raise DataSplitError("Недостаточно элементов: для разбиения нужно минимум 3 элемента")

    counts = [
        int(math.floor(total_items * train_ratio)),
        int(math.floor(total_items * val_ratio)),
        total_items,
    ]
    counts[2] -= counts[0] + counts[1]

    for idx in range(3):
        if counts[idx] > 0:
            continue

        donor_idx = max(range(3), key=lambda i: counts[i])
        counts[donor_idx] -= 1
        counts[idx] += 1

    return counts[0], counts[1], counts[2]


def _iter_series_views(df: pd.DataFrame, time_col: str | None, series_col: str | None) -> list[tuple[str, pd.DataFrame]]:
    if series_col is None:
        groups = [("__single_series__", df)]
    else:
        groups = [(str(series_id), series_df) for series_id, series_df in df.groupby(series_col, sort=False)]

    ordered_groups: list[tuple[str, pd.DataFrame]] = []
    for series_id, series_df in groups:
        if time_col is not None:
            ordered = series_df.sort_values(by=time_col, kind="stable")
        else:
            ordered = series_df
        ordered_groups.append((series_id, ordered))
    return ordered_groups


def _build_window_record(series_id: str, row_positions: list[int], split_name: str) -> dict[str, Any]:
    return {
        "series_id": series_id,
        "split": split_name,
        "row_positions": row_positions,
        "target_position": int(row_positions[-1]),
        "start_position": int(row_positions[0]),
        "end_position": int(row_positions[-1]),
    }


def build_window_split_plan(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> dict[str, Any]:
    """Строит план временных окон и их разбиение по времени (70/15/15 или по переданным долям)."""
    _validate_split_params(window_size, stride, train_ratio, val_ratio, test_ratio, random_state)

    if target_col not in df.columns:
        raise DataSplitError(f"Отсутствует целевая колонка: '{target_col}'")

    time_col = detect_time_column(df.columns)
    series_col = detect_series_column(df.columns)

    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    train_row_positions_for_fit: set[int] = set()
    series_summaries: list[dict[str, Any]] = []

    for series_id, ordered_df in _iter_series_views(df, time_col=time_col, series_col=series_col):
        ordered_positions = [int(pos) for pos in ordered_df.index.tolist()]
        if len(ordered_positions) < window_size:
            raise DataSplitError(
                f"Серия '{series_id}' слишком короткая для окна длины {window_size}: {len(ordered_positions)} строк"
            )

        train_row_count, val_row_count, test_row_count = _compute_split_counts(
            total_items=len(ordered_positions),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        row_boundaries = {
            "train": (0, train_row_count),
            "val": (train_row_count, train_row_count + val_row_count),
            "test": (train_row_count + val_row_count, len(ordered_positions)),
        }

        split_window_counts: dict[str, int] = {}
        for split_name, (left, right) in row_boundaries.items():
            split_positions = ordered_positions[left:right]
            if len(split_positions) < window_size:
                raise DataSplitError(
                    f"Серия '{series_id}' слишком короткая для непересекающегося split "
                    f"с окном длины {window_size}: в части '{split_name}' только {len(split_positions)} строк"
                )

            starts = list(range(0, len(split_positions) - window_size + 1, stride))
            split_window_counts[split_name] = len(starts)
            for start in starts:
                end = start + window_size
                window_positions = split_positions[start:end]
                splits[split_name].append(_build_window_record(series_id, window_positions, split_name))

        train_left, train_right = row_boundaries["train"]
        train_row_positions_for_fit.update(ordered_positions[train_left:train_right])

        series_summaries.append(
            {
                "series_id": series_id,
                "row_count": len(ordered_positions),
                "train_rows": train_row_count,
                "val_rows": val_row_count,
                "test_rows": test_row_count,
                "train_windows": split_window_counts["train"],
                "val_windows": split_window_counts["val"],
                "test_windows": split_window_counts["test"],
                "time_column_used": time_col,
            }
        )

    return {
        "window_size": int(window_size),
        "stride": int(stride),
        "random_state": int(random_state),
        "time_column": time_col,
        "series_column": series_col,
        "series_summaries": series_summaries,
        "splits": splits,
        "train_row_positions_for_fit": sorted(train_row_positions_for_fit),
    }


def build_inference_window_plan(df: pd.DataFrame, window_size: int, stride: int) -> dict[str, Any]:
    """Строит план окон для неразмеченного набора без разбиения на train/val/test."""
    if not isinstance(window_size, int) or window_size <= 1:
        raise DataSplitError("window_size должен быть целым числом больше 1")
    if not isinstance(stride, int) or stride <= 0:
        raise DataSplitError("stride должен быть положительным целым числом")

    time_col = detect_time_column(df.columns)
    series_col = detect_series_column(df.columns)
    windows: list[dict[str, Any]] = []

    for series_id, ordered_df in _iter_series_views(df, time_col=time_col, series_col=series_col):
        ordered_positions = [int(pos) for pos in ordered_df.index.tolist()]
        if len(ordered_positions) < window_size:
            continue

        starts = list(range(0, len(ordered_positions) - window_size + 1, stride))
        for start in starts:
            end = start + window_size
            windows.append(
                {
                    "series_id": series_id,
                    "row_positions": ordered_positions[start:end],
                    "start_position": int(ordered_positions[start]),
                    "end_position": int(ordered_positions[end - 1]),
                }
            )

    if not windows:
        raise DataSplitError("Не удалось построить ни одного окна для inference")

    return {
        "window_size": int(window_size),
        "stride": int(stride),
        "time_column": time_col,
        "series_column": series_col,
        "windows": windows,
    }


def _validate_feature_rows(features: np.ndarray) -> None:
    if features.ndim != 2:
        raise DataSplitError("Матрица признаков должна быть двумерной")
    if features.shape[0] == 0:
        raise DataSplitError("Матрица признаков не должна быть пустой")


def materialize_labeled_window_splits(
    features: np.ndarray,
    targets: np.ndarray,
    plan: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Преобразует план окон в массивы `X/y` для train/val/test."""
    _validate_feature_rows(features)
    if targets.ndim != 1:
        raise DataSplitError("Вектор targets должен быть одномерным")
    if len(targets) != features.shape[0]:
        raise DataSplitError("Количество строк в features должно совпадать с длиной targets")

    window_size = int(plan["window_size"])
    n_features = int(features.shape[1])

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in SPLIT_NAMES:
        records = list(plan["splits"][split_name])
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


def materialize_unlabeled_windows(features: np.ndarray, plan: dict[str, Any]) -> np.ndarray:
    """Преобразует план inference-окон в батч `(n_windows, T, n_features)`."""
    _validate_feature_rows(features)

    records = list(plan["windows"])
    window_size = int(plan["window_size"])
    n_features = int(features.shape[1])

    x_list = [features[np.asarray(record["row_positions"], dtype=np.int64)] for record in records]
    if not x_list:
        return np.empty((0, window_size, n_features), dtype=np.float32)
    return np.stack(x_list, axis=0).astype(np.float32)


def get_class_distribution(targets: np.ndarray) -> dict[int, int]:
    """Возвращает распределение классов по вектору целевых меток."""
    if targets.ndim != 1:
        raise DataSplitError("Вектор targets должен быть одномерным")
    if len(targets) == 0:
        raise DataSplitError("Вектор targets не должен быть пустым")

    unique, counts = np.unique(targets, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts, strict=True)}


def build_split_report(window_splits: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """Формирует краткую сводку window-based split по формам и классам."""
    report: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        x_split, y_split = window_splits[split_name]
        report[f"{split_name}_shape"] = [int(x_split.shape[0]), int(x_split.shape[1]), int(x_split.shape[2])]
        report[f"{split_name}_class_distribution"] = get_class_distribution(y_split)
    return report


def flatten_windows(x_windows: np.ndarray) -> np.ndarray:
    """Преобразует батч окон `(n, T, f)` в плоский вид `(n, T*f)` для baseline."""
    if x_windows.ndim != 3:
        raise DataSplitError("Для flatten ожидается массив размерности (n_windows, T, n_features)")
    if x_windows.shape[0] == 0:
        raise DataSplitError("Пакет окон не должен быть пустым")
    return x_windows.reshape(x_windows.shape[0], x_windows.shape[1] * x_windows.shape[2]).astype(np.float32)


def validate_no_index_leakage(plan: dict[str, Any]) -> None:
    """Проверяет, что строки исходного ряда не пересекаются между split."""
    row_sets = {
        split_name: {int(pos) for record in plan["splits"][split_name] for pos in record["row_positions"]}
        for split_name in SPLIT_NAMES
    }

    if row_sets["train"].intersection(row_sets["val"]):
        raise DataSplitError("Утечка данных: пересечение строк между train и val")
    if row_sets["train"].intersection(row_sets["test"]):
        raise DataSplitError("Утечка данных: пересечение строк между train и test")
    if row_sets["val"].intersection(row_sets["test"]):
        raise DataSplitError("Утечка данных: пересечение строк между val и test")


def save_split_artifact(plan: dict[str, Any], path: str) -> None:
    """Сохраняет JSON-артефакт с планом разбиения для воспроизводимости."""
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(plan, file_obj, ensure_ascii=False, indent=2)

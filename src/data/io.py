"""Загрузка и контрактная валидация табличных данных ТМИ."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data.validation import (
    DataContract,
    DataValidationError,
    normalize_missing_tokens,
    validate_dataframe_schema,
    validate_numeric_feature_columns,
)


class DataIOError(ValueError):
    """Исключение для ошибок ввода-вывода и формата датасета."""


def _read_dataset(source: Mapping[str, Any], null_tokens: list[object], stage_name: str) -> pd.DataFrame:
    path = Path(source["path"])
    if not path.exists():
        raise FileNotFoundError(f"[{stage_name}] Файл датасета не найден: {path}")

    source_format = source["format"]
    try:
        if source_format == "csv":
            df = pd.read_csv(
                path,
                sep=source["sep"],
                encoding=source["encoding"],
                na_values=null_tokens,
                keep_default_na=True,
            )
        elif source_format == "xls":
            df = pd.read_excel(
                path,
                sheet_name=source.get("sheet_name", 0),
                engine="xlrd",
                na_values=null_tokens,
            )
        else:
            raise DataIOError(f"[{stage_name}] Неподдерживаемый формат датасета: {source_format}")
    except pd.errors.EmptyDataError as exc:
        raise DataIOError(f"[{stage_name}] Датасет пуст: {path}") from exc

    df = normalize_missing_tokens(df, null_tokens)
    if df.empty:
        raise DataIOError(f"[{stage_name}] Датасет пуст: {path}")

    return df.reset_index(drop=True)


def _drop_all_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.dropna(axis=0, how="all")
    if filtered.empty:
        raise DataIOError("После удаления полностью пустых строк датасет оказался пустым")
    return filtered.reset_index(drop=True)


def load_labeled_dataset(config: Mapping[str, Any]) -> pd.DataFrame:
    """Загружает размеченный датасет и валидирует колонку с метками."""
    data_cfg = config["data"]
    task_cfg = config["task"]
    stage_name = "Этап 1 / labeled"

    labeled_df = _read_dataset(data_cfg["labeled"], data_cfg["null_tokens"], stage_name=stage_name)
    if data_cfg["drop_all_nan_rows"]:
        labeled_df = _drop_all_nan_rows(labeled_df)

    contract = DataContract(target_col=task_cfg["target_col"])
    validate_dataframe_schema(labeled_df, contract)

    feature_count = len([col for col in labeled_df.columns if col != task_cfg["target_col"]])
    if feature_count != data_cfg["expected_feature_count"]:
        raise DataValidationError(
            f"[{stage_name}] Ожидалось {data_cfg['expected_feature_count']} признаков, получено {feature_count}"
        )

    return labeled_df


def load_unlabeled_dataset(config: Mapping[str, Any], expected_features: tuple[str, ...]) -> pd.DataFrame:
    """Загружает неразмеченный датасет и проверяет совпадение признаков."""
    data_cfg = config["data"]
    task_cfg = config["task"]
    stage_name = "Этап 1 / unlabeled"

    unlabeled_df = _read_dataset(data_cfg["unlabeled"], data_cfg["null_tokens"], stage_name=stage_name)
    if data_cfg["drop_all_nan_rows"]:
        unlabeled_df = _drop_all_nan_rows(unlabeled_df)

    target_col = task_cfg["target_col"]
    if target_col in unlabeled_df.columns:
        raise DataValidationError(f"[{stage_name}] В неразмеченном датасете не должно быть колонки '{target_col}'")

    validate_numeric_feature_columns(unlabeled_df, unlabeled_df.columns)

    if len(unlabeled_df.columns) != data_cfg["expected_feature_count"]:
        raise DataValidationError(
            f"[{stage_name}] Ожидалось {data_cfg['expected_feature_count']} признаков, получено {len(unlabeled_df.columns)}"
        )

    missing_features = [name for name in expected_features if name not in unlabeled_df.columns]
    extra_features = [name for name in unlabeled_df.columns if name not in expected_features]
    if missing_features or extra_features:
        raise DataValidationError(
            f"[{stage_name}] Несовпадение признаков между размеченным и неразмеченным наборами: "
            f"missing={missing_features}, extra={extra_features}"
        )

    return unlabeled_df


def load_datasets(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает оба датасета и проверяет их совместимость по признакам."""
    labeled_df = load_labeled_dataset(config)

    target_col = config["task"]["target_col"]
    expected_features = tuple(col for col in labeled_df.columns if col != target_col)

    unlabeled_df = load_unlabeled_dataset(config, expected_features=expected_features)
    return labeled_df, unlabeled_df

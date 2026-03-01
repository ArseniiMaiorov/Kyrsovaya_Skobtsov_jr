"""Утилиты валидации датафрейма по контракту данных."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

ALLOWED_LABELS = {0, 1, 2}


class DataValidationError(ValueError):
    """Исключение для нарушений контракта данных."""


@dataclass(frozen=True)
class DataContract:
    target_col: str
    required_features: tuple[str, ...] = ()


def normalize_missing_tokens(df: pd.DataFrame, null_tokens: Iterable[object]) -> pd.DataFrame:
    """Возвращает копию датафрейма с заменой токенов пропусков на NaN."""
    return df.replace(list(null_tokens), pd.NA)


def validate_label_values(labels: Iterable[object]) -> None:
    """Проверяет, что метки непустые и входят только в ALLOWED_LABELS."""
    label_set = set(labels)
    if not label_set:
        raise DataValidationError("Целевые метки пусты")

    unknown = label_set - ALLOWED_LABELS
    if unknown:
        raise DataValidationError(f"Обнаружены неизвестные метки: {sorted(unknown)}")


def validate_numeric_feature_columns(df: pd.DataFrame, feature_columns: Iterable[str]) -> None:
    """Проверяет, что признаковые колонки являются числовыми после парсинга."""
    non_numeric: list[str] = []
    for column in feature_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        # Если после приведения появились NaN там, где исходно значение не было NaN, это нечисловой мусор.
        mask_invalid = converted.isna() & df[column].notna()
        if bool(mask_invalid.any()):
            non_numeric.append(column)

    if non_numeric:
        raise DataValidationError(f"Обнаружены нечисловые признаки: {non_numeric}")


def validate_dataframe_schema(df: pd.DataFrame, contract: DataContract) -> None:
    """Проверяет колонки датафрейма и значения целевой переменной."""
    if contract.target_col not in df.columns:
        raise DataValidationError(f"Отсутствует целевая колонка: '{contract.target_col}'")

    missing_features = [name for name in contract.required_features if name not in df.columns]
    if missing_features:
        missing = ", ".join(missing_features)
        raise DataValidationError(f"Отсутствуют обязательные признаки: {missing}")

    feature_columns = [name for name in df.columns if name != contract.target_col]
    validate_numeric_feature_columns(df, feature_columns)
    validate_label_values(df[contract.target_col].dropna().tolist())

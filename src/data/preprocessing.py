"""Подготовка raw/improved версий данных без утечки статистик между split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class PreprocessingError(ValueError):
    """Исключение для ошибок предобработки данных."""


@dataclass(frozen=True)
class RawPreprocessor:
    feature_names: tuple[str, ...]
    medians: dict[str, float]


@dataclass(frozen=True)
class ImprovedPreprocessor:
    feature_names: tuple[str, ...]
    medians: dict[str, float]
    clip_bounds: dict[str, tuple[float, float]]
    scaler: RobustScaler


def _ensure_target_exists(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise PreprocessingError(f"Отсутствует целевая колонка: '{target_col}'")


def _extract_feature_names(df: pd.DataFrame, target_col: str) -> tuple[str, ...]:
    feature_names = tuple(col for col in df.columns if col != target_col)
    if not feature_names:
        raise PreprocessingError("Не найдены признаки для предобработки")
    return feature_names


def _to_numeric_features(df: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    return df.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")


def _to_numeric_target(df: pd.DataFrame, target_col: str) -> np.ndarray:
    try:
        target = pd.to_numeric(df[target_col], errors="raise").to_numpy(dtype=np.int64)
    except Exception as exc:
        raise PreprocessingError("Целевая колонка должна быть числовой") from exc
    return target


def _compute_medians(features: pd.DataFrame) -> dict[str, float]:
    medians_series = features.median()
    if medians_series.isna().any():
        raise PreprocessingError("Невозможно вычислить медианы: есть полностью пустые признаки")
    return {col: float(val) for col, val in medians_series.to_dict().items()}


def _validate_quantiles(clip_quantiles: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(clip_quantiles, tuple) or len(clip_quantiles) != 2:
        raise PreprocessingError("clip_quantiles должен быть кортежем из двух чисел")

    lower_q, upper_q = clip_quantiles
    if not isinstance(lower_q, (int, float)) or not isinstance(upper_q, (int, float)):
        raise PreprocessingError("clip_quantiles должен содержать числа")
    if not (0 <= lower_q < upper_q <= 1):
        raise PreprocessingError("clip_quantiles должен удовлетворять условию 0 <= lower < upper <= 1")

    return float(lower_q), float(upper_q)


def _validate_feature_set(df: pd.DataFrame, expected_features: tuple[str, ...], target_col: str | None = None) -> None:
    present_features = [col for col in df.columns if col != target_col]
    missing = [name for name in expected_features if name not in present_features]
    extra = [name for name in present_features if name not in expected_features]
    if missing or extra:
        raise PreprocessingError(f"Несовпадение признаков: missing={missing}, extra={extra}")


def fit_raw_preprocessor(train_df: pd.DataFrame, target_col: str) -> RawPreprocessor:
    """Обучает raw-препроцессор: только median-импутация по train."""
    _ensure_target_exists(train_df, target_col)
    feature_names = _extract_feature_names(train_df, target_col)
    train_features = _to_numeric_features(train_df, feature_names)
    medians = _compute_medians(train_features)
    return RawPreprocessor(feature_names=feature_names, medians=medians)


def _transform_raw_features(df: pd.DataFrame, preprocessor: RawPreprocessor) -> np.ndarray:
    _validate_feature_set(df, preprocessor.feature_names, target_col=None)
    features = _to_numeric_features(df, preprocessor.feature_names)
    imputed = features.fillna(preprocessor.medians)
    return imputed.to_numpy(dtype=np.float32)


def transform_raw_labeled(
    df: pd.DataFrame,
    preprocessor: RawPreprocessor,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Применяет raw-препроцессор к размеченной выборке."""
    _ensure_target_exists(df, target_col)
    _validate_feature_set(df, preprocessor.feature_names, target_col=target_col)
    x = _transform_raw_features(df.drop(columns=[target_col]), preprocessor)
    y = _to_numeric_target(df, target_col)
    return x, y


def transform_raw_unlabeled(df: pd.DataFrame, preprocessor: RawPreprocessor) -> np.ndarray:
    """Применяет raw-препроцессор к неразмеченной выборке."""
    return _transform_raw_features(df, preprocessor)


def prepare_raw_data(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Формирует raw-версию одного датафрейма (fit+transform на нём самом)."""
    preprocessor = fit_raw_preprocessor(df, target_col=target_col)
    x, y = transform_raw_labeled(df, preprocessor, target_col=target_col)
    return x, y, preprocessor.feature_names


def prepare_raw_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any]:
    """Строит raw-версию train/val/test без утечки (fit только на train)."""
    preprocessor = fit_raw_preprocessor(train_df, target_col=target_col)

    x_train, y_train = transform_raw_labeled(train_df, preprocessor, target_col=target_col)
    x_val, y_val = transform_raw_labeled(val_df, preprocessor, target_col=target_col)
    x_test, y_test = transform_raw_labeled(test_df, preprocessor, target_col=target_col)

    return {
        "preprocessor": preprocessor,
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


def fit_improved_preprocessor(
    train_df: pd.DataFrame,
    target_col: str,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> ImprovedPreprocessor:
    """Обучает improved-препроцессор на train: median -> winsorize -> RobustScaler."""
    _ensure_target_exists(train_df, target_col)
    lower_q, upper_q = _validate_quantiles(clip_quantiles)
    feature_names = _extract_feature_names(train_df, target_col)

    train_features = _to_numeric_features(train_df, feature_names)
    medians = _compute_medians(train_features)
    imputed = train_features.fillna(medians)

    lower = imputed.quantile(lower_q)
    upper = imputed.quantile(upper_q)
    clip_bounds = {col: (float(lower[col]), float(upper[col])) for col in feature_names}

    clipped = imputed.copy()
    for col in feature_names:
        lo, hi = clip_bounds[col]
        clipped[col] = clipped[col].clip(lower=lo, upper=hi)

    scaler = RobustScaler()
    scaler.fit(clipped.to_numpy(dtype=np.float32))

    return ImprovedPreprocessor(
        feature_names=feature_names,
        medians=medians,
        clip_bounds=clip_bounds,
        scaler=scaler,
    )


def _transform_improved_features(df: pd.DataFrame, preprocessor: ImprovedPreprocessor) -> np.ndarray:
    _validate_feature_set(df, preprocessor.feature_names, target_col=None)
    features = _to_numeric_features(df, preprocessor.feature_names)
    imputed = features.fillna(preprocessor.medians)

    clipped = imputed.copy()
    for col in preprocessor.feature_names:
        lo, hi = preprocessor.clip_bounds[col]
        clipped[col] = clipped[col].clip(lower=lo, upper=hi)

    scaled = preprocessor.scaler.transform(clipped.to_numpy(dtype=np.float32))
    return scaled.astype(np.float32)


def transform_improved_labeled(
    df: pd.DataFrame,
    preprocessor: ImprovedPreprocessor,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Применяет improved-препроцессор к размеченной выборке."""
    _ensure_target_exists(df, target_col)
    _validate_feature_set(df, preprocessor.feature_names, target_col=target_col)
    x = _transform_improved_features(df.drop(columns=[target_col]), preprocessor)
    y = _to_numeric_target(df, target_col)
    return x, y


def transform_improved_unlabeled(df: pd.DataFrame, preprocessor: ImprovedPreprocessor) -> np.ndarray:
    """Применяет improved-препроцессор к неразмеченной выборке."""
    return _transform_improved_features(df, preprocessor)


def prepare_improved_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
) -> dict[str, Any]:
    """Строит improved-версию train/val/test без утечки (fit только на train)."""
    preprocessor = fit_improved_preprocessor(train_df, target_col=target_col, clip_quantiles=clip_quantiles)

    x_train, y_train = transform_improved_labeled(train_df, preprocessor, target_col=target_col)
    x_val, y_val = transform_improved_labeled(val_df, preprocessor, target_col=target_col)
    x_test, y_test = transform_improved_labeled(test_df, preprocessor, target_col=target_col)

    return {
        "preprocessor": preprocessor,
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }

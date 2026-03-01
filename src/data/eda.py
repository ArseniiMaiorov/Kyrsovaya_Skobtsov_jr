"""Базовые функции EDA для этапа первичного анализа ТМИ."""

from __future__ import annotations

from typing import Any

import pandas as pd


class EDAError(ValueError):
    """Исключение для ошибок расчета EDA-метрик."""


def get_numeric_feature_names(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Возвращает список числовых колонок, исключая заданные."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return [name for name in numeric_cols if name not in exclude]


def get_missing_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Считает пропуски по датафрейму и по колонкам."""
    per_column = df.isna().sum().astype(int).to_dict()
    row_count = int(df.shape[0])
    per_column_share = {
        str(name): (float(count / row_count) if row_count else 0.0)
        for name, count in per_column.items()
    }
    total_missing = int(sum(per_column.values()))
    total_cells = int(df.shape[0] * df.shape[1])
    missing_share = float(total_missing / total_cells) if total_cells else 0.0
    return {
        "total_missing": total_missing,
        "missing_share": missing_share,
        "per_column": per_column,
        "per_column_share": per_column_share,
    }


def get_class_distribution(df: pd.DataFrame, target_col: str) -> dict[int, int]:
    """Возвращает распределение классов в виде {метка: количество}."""
    if target_col not in df.columns:
        raise EDAError(f"Колонка '{target_col}' не найдена для расчета распределения классов")

    counts = df[target_col].value_counts().sort_index()
    return {int(label): int(count) for label, count in counts.items()}


def get_outlier_share_iqr(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Оценивает долю выбросов по правилу IQR для списка колонок."""
    result: dict[str, float] = {}
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            result[col] = 0.0
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            result[col] = 0.0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((series < lower) | (series > upper)).sum()
        result[col] = float(outliers / len(series))

    return result


def get_top_missing_features(
    df: pd.DataFrame,
    columns: list[str],
    top_n: int = 10,
) -> list[tuple[str, float, int]]:
    """Возвращает топ признаков по доле пропусков."""
    if top_n <= 0:
        raise EDAError("top_n должен быть положительным")

    if not columns:
        return []

    row_count = int(df.shape[0])
    result: list[tuple[str, float, int]] = []
    for col in columns:
        missing_count = int(df[col].isna().sum())
        missing_share = float(missing_count / row_count) if row_count else 0.0
        result.append((col, missing_share, missing_count))

    result.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return result[:top_n]


def get_basic_statistics(
    df: pd.DataFrame,
    columns: list[str],
) -> dict[str, dict[str, float | None]]:
    """Возвращает базовую статистику по числовым признакам."""
    result: dict[str, dict[str, float | None]] = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            result[col] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "q01": None,
                "q25": None,
                "q50": None,
                "q75": None,
                "q99": None,
            }
            continue

        result[col] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "q01": float(series.quantile(0.01)),
            "q25": float(series.quantile(0.25)),
            "q50": float(series.quantile(0.50)),
            "q75": float(series.quantile(0.75)),
            "q99": float(series.quantile(0.99)),
        }

    return result


def get_placeholder_counts(
    df: pd.DataFrame,
    columns: list[str],
    placeholders: tuple[int, ...] = (-999, -9999),
) -> dict[str, dict[str, int]]:
    """Считает вхождения типовых числовых заглушек по признакам."""
    result: dict[str, dict[str, int]] = {}
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        result[col] = {
            str(placeholder): int((series == placeholder).sum())
            for placeholder in placeholders
        }
    return result


def get_constant_features(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Возвращает список константных признаков."""
    constant_features: list[str] = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.nunique() <= 1:
            constant_features.append(col)
    return constant_features


def get_high_correlation_pairs(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.95,
    max_pairs: int = 20,
) -> list[tuple[str, str, float]]:
    """Возвращает пары признаков с высокой корреляцией Пирсона."""
    if not 0 <= threshold <= 1:
        raise EDAError("Порог корреляции должен быть в диапазоне [0, 1]")
    if max_pairs <= 0:
        raise EDAError("max_pairs должен быть положительным")

    if not columns:
        return []

    corr = df[columns].corr().abs()
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            right = columns[j]
            value = float(corr.iloc[i, j])
            if value >= threshold:
                pairs.append((left, right, value))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:max_pairs]


def build_eda_summary(labeled_df: pd.DataFrame, unlabeled_df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Формирует сводку EDA для размеченного и неразмеченного наборов."""
    class_distribution = get_class_distribution(labeled_df, target_col)
    numeric_features = get_numeric_feature_names(labeled_df, exclude=(target_col,))
    unlabeled_numeric_features = get_numeric_feature_names(unlabeled_df)

    labeled_missing = get_missing_summary(labeled_df)
    unlabeled_missing = get_missing_summary(unlabeled_df)
    labeled_top_missing_features = get_top_missing_features(labeled_df, numeric_features, top_n=10)
    unlabeled_top_missing_features = get_top_missing_features(unlabeled_df, unlabeled_numeric_features, top_n=10)
    labeled_basic_statistics = get_basic_statistics(labeled_df, numeric_features)
    unlabeled_basic_statistics = get_basic_statistics(unlabeled_df, unlabeled_numeric_features)
    labeled_placeholder_counts = get_placeholder_counts(labeled_df, numeric_features)
    unlabeled_placeholder_counts = get_placeholder_counts(unlabeled_df, unlabeled_numeric_features)
    constant_features = get_constant_features(labeled_df, numeric_features)

    outlier_share = get_outlier_share_iqr(labeled_df, numeric_features)
    top_outlier_features = sorted(outlier_share.items(), key=lambda x: x[1], reverse=True)[:10]

    high_corr_pairs = get_high_correlation_pairs(labeled_df, numeric_features, threshold=0.95, max_pairs=20)
    majority = max(class_distribution.values())
    minority = min(class_distribution.values())
    imbalance_ratio = float(majority / minority) if minority else float("inf")

    return {
        "labeled_shape": [int(labeled_df.shape[0]), int(labeled_df.shape[1])],
        "unlabeled_shape": [int(unlabeled_df.shape[0]), int(unlabeled_df.shape[1])],
        "numeric_feature_count": len(numeric_features),
        "class_distribution": class_distribution,
        "class_imbalance_ratio": imbalance_ratio,
        "labeled_missing": labeled_missing,
        "unlabeled_missing": unlabeled_missing,
        "labeled_top_missing_features": labeled_top_missing_features,
        "unlabeled_top_missing_features": unlabeled_top_missing_features,
        "labeled_basic_statistics": labeled_basic_statistics,
        "unlabeled_basic_statistics": unlabeled_basic_statistics,
        "labeled_placeholder_counts": labeled_placeholder_counts,
        "unlabeled_placeholder_counts": unlabeled_placeholder_counts,
        "constant_features": constant_features,
        "top_outlier_features": top_outlier_features,
        "high_correlation_pairs": high_corr_pairs,
    }

#!/usr/bin/env python3
"""Запуск EDA (этап 2) и сохранение отчета в reports/experiments."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eda import build_eda_summary
from src.data.io import load_datasets
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility


def _format_ratio(value: float) -> str:
    return f"{value:.4f}"


def _format_stats_block(stats: dict[str, dict[str, object]], limit: int = 5) -> list[str]:
    lines: list[str] = []
    for feature_name in list(stats.keys())[:limit]:
        feature_stats = stats[feature_name]
        lines.append(
            "- "
            f"{feature_name}: "
            f"mean={feature_stats['mean']}, std={feature_stats['std']}, "
            f"min={feature_stats['min']}, q01={feature_stats['q01']}, "
            f"q50={feature_stats['q50']}, q99={feature_stats['q99']}, max={feature_stats['max']}"
        )
    return lines


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    initialize_reproducibility(config, PROJECT_ROOT, stage_name="stage2_eda")
    labeled_df, unlabeled_df = load_datasets(config)
    target_col = config["task"]["target_col"]

    summary = build_eda_summary(labeled_df, unlabeled_df, target_col=target_col)

    out_dir = PROJECT_ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "eda_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    class_counts = summary["class_distribution"]
    top_outliers = summary["top_outlier_features"]
    corr_pairs = summary["high_correlation_pairs"]
    top_missing_labeled = summary["labeled_top_missing_features"]
    constant_features = summary["constant_features"]
    placeholder_totals = summary["labeled_placeholder_counts"]

    md_lines = [
        "# Этап 2: EDA",
        "",
        "## Размеры наборов",
        f"- Размеченный: {summary['labeled_shape'][0]} x {summary['labeled_shape'][1]}",
        f"- Неразмеченный: {summary['unlabeled_shape'][0]} x {summary['unlabeled_shape'][1]}",
        f"- Число числовых признаков: {summary['numeric_feature_count']}",
        "",
        "## Распределение классов",
    ]

    for label, count in sorted(class_counts.items(), key=lambda x: int(x[0])):
        md_lines.append(f"- Класс {label}: {count}")

    md_lines.extend(
        [
            f"- Дисбаланс (majority/minority): {_format_ratio(float(summary['class_imbalance_ratio']))}",
            "",
            "## Пропуски",
            f"- Размеченный: {summary['labeled_missing']['total_missing']}",
            f"- Неразмеченный: {summary['unlabeled_missing']['total_missing']}",
            "",
            "### Топ-10 признаков по доле пропусков (labeled)",
        ]
    )

    if int(summary["labeled_missing"]["total_missing"]) == 0:
        md_lines.append("- Пропуски по признакам отсутствуют")
    else:
        for feature_name, share, count in top_missing_labeled:
            md_lines.append(f"- {feature_name}: доля={_format_ratio(float(share))}, количество={int(count)}")

    total_neg999 = sum(int(value["-999"]) for value in placeholder_totals.values())
    total_neg9999 = sum(int(value["-9999"]) for value in placeholder_totals.values())

    md_lines.extend(
        [
            "",
            "## Базовая статистика (первые 5 признаков labeled)",
        ]
    )
    md_lines.extend(_format_stats_block(summary["labeled_basic_statistics"], limit=5))

    md_lines.extend(
        [
            "",
            "## Заглушки -999 / -9999 (labeled)",
        ]
    )

    if total_neg999 == 0 and total_neg9999 == 0:
        md_lines.append("- Не обнаружены")
    else:
        md_lines.append(f"- Всего -999: {total_neg999}")
        md_lines.append(f"- Всего -9999: {total_neg9999}")

    md_lines.extend(
        [
            "",
            "## Константные признаки",
        ]
    )

    if constant_features:
        for feature_name in constant_features:
            md_lines.append(f"- {feature_name}")
    else:
        md_lines.append("- Не обнаружены")

    md_lines.extend(
        [
            "",
            "## Топ признаков по доле выбросов (IQR)",
        ]
    )

    for feature, share in top_outliers:
        md_lines.append(f"- {feature}: {_format_ratio(float(share))}")

    md_lines.extend(["", "## Сильные корреляции (|r| >= 0.95)"])
    if corr_pairs:
        for left, right, corr in corr_pairs:
            md_lines.append(f"- {left} ↔ {right}: {_format_ratio(float(corr))}")
    else:
        md_lines.append("- Не обнаружены")

    md_lines.extend(
        [
            "",
            "## Выводы для предобработки",
            "- Разбиение выполнять только по временной оси, затем строить окна внутри каждого split-участка.",
            "- Для `raw` использовать только median-импутацию по train без масштабирования.",
            "- Для `improved` использовать median-импутацию, winsorize по q01/q99 и `RobustScaler` только по train.",
            "- Из-за дисбаланса классов считать `balanced_accuracy`, `macro-F1` и использовать `class_weight=balanced`.",
        ]
    )

    md_path = out_dir / "eda_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Сохранено: {json_path}")
    print(f"Сохранено: {md_path}")


if __name__ == "__main__":
    main()

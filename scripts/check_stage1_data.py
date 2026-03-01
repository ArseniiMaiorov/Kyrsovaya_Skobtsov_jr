#!/usr/bin/env python3
"""Проверка загрузки и базовой статистики датасетов для Этапа 1."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_datasets
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility


def main() -> None:
    config_path = Path("config.yaml")
    config = load_config(config_path)
    initialize_reproducibility(config, PROJECT_ROOT, stage_name="stage1_check_data")

    labeled_df, unlabeled_df = load_datasets(config)

    target_col = config["task"]["target_col"]
    class_counts = labeled_df[target_col].value_counts().sort_index()

    print("Этап 1: проверка датасетов")
    print(f"Размеченный набор: {labeled_df.shape[0]} строк, {labeled_df.shape[1]} колонок")
    print(f"Неразмеченный набор: {unlabeled_df.shape[0]} строк, {unlabeled_df.shape[1]} колонок")
    print("Распределение классов:")
    for label, count in class_counts.items():
        print(f"  класс {label}: {count}")

    print(f"Пропуски (размеченный): {int(labeled_df.isna().sum().sum())}")
    print(f"Пропуски (неразмеченный): {int(unlabeled_df.isna().sum().sum())}")


if __name__ == "__main__":
    main()

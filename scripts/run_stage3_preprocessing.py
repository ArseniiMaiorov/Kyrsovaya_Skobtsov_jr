#!/usr/bin/env python3
"""Этап 3: формирование raw/improved, окон и time-based split."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_datasets
from src.data.preprocessing import (
    fit_improved_preprocessor,
    fit_raw_preprocessor,
    transform_improved_labeled,
    transform_improved_unlabeled,
    transform_raw_labeled,
    transform_raw_unlabeled,
)
from src.data.splits import (
    build_inference_window_plan,
    build_split_report,
    build_window_split_plan,
    materialize_labeled_window_splits,
    materialize_unlabeled_windows,
    save_split_artifact,
    validate_no_index_leakage,
)
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility


def _compute_iqr_scale_deviation(x_train: np.ndarray) -> float:
    q75 = np.percentile(x_train, 75, axis=0)
    q25 = np.percentile(x_train, 25, axis=0)
    iqr = q75 - q25
    return float(np.abs(iqr - 1.0).mean())


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    initialize_reproducibility(config, PROJECT_ROOT, stage_name="stage3_preprocessing")
    labeled_df, unlabeled_df = load_datasets(config)

    target_col = config["task"]["target_col"]
    seq_cfg = config["sequence"]
    split_cfg = config["split"]
    prep_cfg = config["preprocessing"]["improved"]

    plan = build_window_split_plan(
        df=labeled_df,
        target_col=target_col,
        window_size=int(seq_cfg["T"]),
        stride=int(seq_cfg["stride"]),
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        random_state=int(split_cfg["random_state"]),
    )
    validate_no_index_leakage(plan)

    split_artifact_path = PROJECT_ROOT / "output" / "artifacts" / "window_split_plan.json"
    split_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    save_split_artifact(plan, str(split_artifact_path))

    train_fit_df = labeled_df.loc[plan["train_row_positions_for_fit"]].reset_index(drop=True)

    raw_preprocessor = fit_raw_preprocessor(train_fit_df, target_col=target_col)
    x_rows_raw, y_rows = transform_raw_labeled(labeled_df, raw_preprocessor, target_col=target_col)
    raw_window_splits = materialize_labeled_window_splits(x_rows_raw, y_rows, plan)
    x_unlabeled_rows_raw = transform_raw_unlabeled(unlabeled_df, raw_preprocessor)

    improved_preprocessor = fit_improved_preprocessor(
        train_fit_df,
        target_col=target_col,
        clip_quantiles=tuple(prep_cfg["clip_quantiles"]),
    )
    x_rows_improved, _ = transform_improved_labeled(labeled_df, improved_preprocessor, target_col=target_col)
    improved_window_splits = materialize_labeled_window_splits(x_rows_improved, y_rows, plan)
    x_unlabeled_rows_improved = transform_improved_unlabeled(unlabeled_df, improved_preprocessor)

    unlabeled_plan = build_inference_window_plan(
        df=unlabeled_df,
        window_size=int(seq_cfg["T"]),
        stride=int(seq_cfg["stride"]),
    )
    x_unlabeled_windows_raw = materialize_unlabeled_windows(x_unlabeled_rows_raw, unlabeled_plan)
    x_unlabeled_windows_improved = materialize_unlabeled_windows(x_unlabeled_rows_improved, unlabeled_plan)

    raw_report = build_split_report(raw_window_splits)
    improved_report = build_split_report(improved_window_splits)

    train_fit_positions = np.asarray(plan["train_row_positions_for_fit"], dtype=np.int64)
    x_train_rows_improved = x_rows_improved[train_fit_positions]
    train_median_max_abs = float(np.abs(np.median(x_train_rows_improved, axis=0)).max())
    train_iqr_mean_abs_diff = _compute_iqr_scale_deviation(x_train_rows_improved)

    summary = {
        "time_column": plan["time_column"],
        "series_column": plan["series_column"],
        "split_artifact": str(split_artifact_path.relative_to(PROJECT_ROOT)),
        "train_row_count_for_fit": int(len(plan["train_row_positions_for_fit"])),
        "raw": {
            "train": raw_report["train_shape"],
            "val": raw_report["val_shape"],
            "test": raw_report["test_shape"],
            "train_class_distribution": raw_report["train_class_distribution"],
            "val_class_distribution": raw_report["val_class_distribution"],
            "test_class_distribution": raw_report["test_class_distribution"],
            "unlabeled_windows_shape": [int(x_unlabeled_windows_raw.shape[0]), int(x_unlabeled_windows_raw.shape[1]), int(x_unlabeled_windows_raw.shape[2])],
        },
        "improved": {
            "train": improved_report["train_shape"],
            "val": improved_report["val_shape"],
            "test": improved_report["test_shape"],
            "train_class_distribution": improved_report["train_class_distribution"],
            "val_class_distribution": improved_report["val_class_distribution"],
            "test_class_distribution": improved_report["test_class_distribution"],
            "unlabeled_windows_shape": [int(x_unlabeled_windows_improved.shape[0]), int(x_unlabeled_windows_improved.shape[1]), int(x_unlabeled_windows_improved.shape[2])],
            "train_median_max_abs": train_median_max_abs,
            "train_iqr_mean_abs_diff": train_iqr_mean_abs_diff,
        },
    }

    out_dir = PROJECT_ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "stage3_preprocessing_summary.json"
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    md_lines = [
        "# Этап 3: предобработка, окна и time-based split",
        "",
        f"- Временная колонка: {plan['time_column'] or 'не найдена, используется порядок строк'}",
        f"- Колонка серии: {plan['series_column'] or 'не найдена, весь файл считается одной серией'}",
        f"- Артефакт разбиения: `{summary['split_artifact']}`",
        f"- Число строк train для fit статистик: {summary['train_row_count_for_fit']}",
        "",
        "## RAW-версия (окна)",
        f"- Train: {summary['raw']['train']}",
        f"- Val: {summary['raw']['val']}",
        f"- Test: {summary['raw']['test']}",
        f"- Классы train: {summary['raw']['train_class_distribution']}",
        f"- Классы val: {summary['raw']['val_class_distribution']}",
        f"- Классы test: {summary['raw']['test_class_distribution']}",
        f"- Окна unlabeled: {summary['raw']['unlabeled_windows_shape']}",
        "",
        "## IMPROVED-версия (окна)",
        f"- Train: {summary['improved']['train']}",
        f"- Val: {summary['improved']['val']}",
        f"- Test: {summary['improved']['test']}",
        f"- Классы train: {summary['improved']['train_class_distribution']}",
        f"- Классы val: {summary['improved']['val_class_distribution']}",
        f"- Классы test: {summary['improved']['test_class_distribution']}",
        f"- Окна unlabeled: {summary['improved']['unlabeled_windows_shape']}",
        f"- max(|median(train_scaled)|): {train_median_max_abs:.6f}",
        f"- mean(|IQR(train_scaled)-1|): {train_iqr_mean_abs_diff:.6f}",
        "",
        "## Проверка отсутствия утечки",
        "- Разбиение выполнено по временной оси на непересекающиеся участки, затем окна построены отдельно внутри каждого участка.",
        "- Статистики `raw/improved` (median, q01/q99, scaler) обучены только на train-части.",
        "- Для `improved` используется `RobustScaler`.",
    ]

    md_path = out_dir / "stage3_preprocessing_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Сохранено: {json_path}")
    print(f"Сохранено: {md_path}")


if __name__ == "__main__":
    main()

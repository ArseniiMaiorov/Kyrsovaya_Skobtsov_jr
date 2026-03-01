#!/usr/bin/env python3
"""Постфактум-диагностика сохранённых val-моделей без повторного обучения."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_datasets
from src.data.preprocessing import fit_improved_preprocessor, transform_improved_labeled
from src.data.splits import build_window_split_plan, materialize_labeled_window_splits
from src.metrics.metrics import evaluate_multiclass_classification, plot_multiclass_roc_curves
from src.models.hybrid import AttentionLayer
from src.utils.config import load_config


def _collect_improved_val(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    labeled_df, _ = load_datasets(config)

    target_col = str(config["task"]["target_col"])
    seq_cfg = config["sequence"]
    split_cfg = config["split"]

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
    train_fit_df = labeled_df.loc[plan["train_row_positions_for_fit"]].reset_index(drop=True)

    improved_preprocessor = fit_improved_preprocessor(
        train_fit_df,
        target_col=target_col,
        clip_quantiles=tuple(config["preprocessing"]["improved"]["clip_quantiles"]),
    )
    x_rows_improved, y_rows = transform_improved_labeled(labeled_df, improved_preprocessor, target_col=target_col)
    improved_splits = materialize_labeled_window_splits(x_rows_improved, y_rows, plan)
    x_val, y_val = improved_splits["val"]
    labels = tuple(int(label) for label in config["task"]["labels"])
    return x_val, y_val, labels


def _evaluate_saved_model(model_path: Path, x_val: np.ndarray, y_val: np.ndarray, labels: tuple[int, ...]) -> dict[str, Any]:
    model = tf.keras.models.load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer}, compile=False)
    y_proba = np.asarray(model.predict(x_val, verbose=0), dtype=np.float64)
    y_pred = np.argmax(y_proba, axis=1).astype(np.int64)
    metrics = evaluate_multiclass_classification(
        y_true=y_val,
        y_pred=y_pred,
        y_proba=y_proba,
        labels=labels,
    )
    roc_info = plot_multiclass_roc_curves(
        y_true=y_val,
        y_proba=y_proba,
        labels=labels,
        save_path=PROJECT_ROOT / "reports" / "figures" / f"{model_path.stem}_val_roc.png",
    )
    tf.keras.backend.clear_session()
    return {
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "val_metrics": metrics,
        "roc_curve": {
            "path": str(Path(roc_info["path"]).relative_to(PROJECT_ROOT)),
            "present_labels": list(roc_info["present_labels"]),
            "per_class_auc": dict(roc_info["per_class_auc"]),
        },
    }


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    x_val, y_val, labels = _collect_improved_val(config)

    candidates = {
        "stage6_ga_best": PROJECT_ROOT / "output" / "models" / "stage6_ga_best.keras",
        "stage7_hybrid_scratch": PROJECT_ROOT / "output" / "models" / "stage7_hybrid_scratch.keras",
        "stage7_hybrid_ae_finetuned": PROJECT_ROOT / "output" / "models" / "stage7_hybrid_ae_finetuned.keras",
    }

    report: dict[str, Any] = {
        "dataset_version": "improved",
        "split": "official_validation_only",
        "models": {},
    }

    for name, model_path in candidates.items():
        if model_path.exists():
            report["models"][name] = _evaluate_saved_model(model_path, x_val, y_val, labels)

    out_dir = PROJECT_ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "posthoc_validation_diagnostics.json"
    md_path = out_dir / "posthoc_validation_diagnostics.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Постфактум-диагностика validation-моделей",
        "",
        "Диагностика выполняется без повторного обучения и без использования `test`.",
    ]
    for model_name, payload in report["models"].items():
        metrics = payload["val_metrics"]
        roc_auc = metrics["roc_auc_ovr_macro"]
        roc_text = "n/a" if roc_auc is None else f"{float(roc_auc):.4f}"
        md_lines.extend(
            [
                "",
                f"## {model_name}",
                f"- model_path: `{payload['model_path']}`",
                f"- macro_f1: {float(metrics['macro_f1']):.4f}",
                f"- balanced_accuracy: {float(metrics['balanced_accuracy']):.4f}",
                f"- roc_auc_ovr_macro: {roc_text}",
                f"- roc_curve: `{payload['roc_curve']['path']}`",
            ]
        )
        if "roc_auc_note" in metrics:
            md_lines.append(f"- Примечание: {metrics['roc_auc_note']}")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Сохранено: {json_path}")
    print(f"Сохранено: {md_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Этап 4: baseline на flattened окнах (только validation)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_datasets
from src.data.preprocessing import fit_improved_preprocessor, fit_raw_preprocessor, transform_improved_labeled, transform_raw_labeled
from src.data.splits import build_window_split_plan, flatten_windows, materialize_labeled_window_splits
from src.models.baseline import run_baseline_experiment
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility


def _collect_window_versions(config: dict) -> dict[str, tuple[tuple[object, object], tuple[object, object]]]:
    labeled_df, _ = load_datasets(config)

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
    train_fit_df = labeled_df.loc[plan["train_row_positions_for_fit"]].reset_index(drop=True)

    raw_preprocessor = fit_raw_preprocessor(train_fit_df, target_col=target_col)
    x_rows_raw, y_rows = transform_raw_labeled(labeled_df, raw_preprocessor, target_col=target_col)
    raw_splits = materialize_labeled_window_splits(x_rows_raw, y_rows, plan)

    improved_preprocessor = fit_improved_preprocessor(
        train_fit_df,
        target_col=target_col,
        clip_quantiles=tuple(prep_cfg["clip_quantiles"]),
    )
    x_rows_improved, _ = transform_improved_labeled(labeled_df, improved_preprocessor, target_col=target_col)
    improved_splits = materialize_labeled_window_splits(x_rows_improved, y_rows, plan)

    raw_train_x, raw_train_y = raw_splits["train"]
    raw_val_x, raw_val_y = raw_splits["val"]
    improved_train_x, improved_train_y = improved_splits["train"]
    improved_val_x, improved_val_y = improved_splits["val"]

    return {
        "raw": ((flatten_windows(raw_train_x), raw_train_y), (flatten_windows(raw_val_x), raw_val_y)),
        "improved": ((flatten_windows(improved_train_x), improved_train_y), (flatten_windows(improved_val_x), improved_val_y)),
    }


def _metric_line(metrics: dict[str, object]) -> str:
    macro_f1 = float(metrics["macro_f1"])
    bal_acc = float(metrics["balanced_accuracy"])
    roc_auc = metrics["roc_auc_ovr_macro"]
    roc_text = "n/a" if roc_auc is None else f"{float(roc_auc):.4f}"
    return f"macro_f1={macro_f1:.4f}, balanced_acc={bal_acc:.4f}, roc_auc={roc_text}"


def _missing_labels_text(y_values: object, labels: tuple[int, ...]) -> str | None:
    present = {int(value) for value in y_values}
    missing = [label for label in labels if label not in present]
    if not missing:
        return None
    return ", ".join(str(label) for label in missing)


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    initialize_reproducibility(config, PROJECT_ROOT, stage_name="stage4_baseline")
    labels = tuple(int(x) for x in config["task"]["labels"])
    random_state = int(config["split"]["random_state"])

    dataset_versions = _collect_window_versions(config)

    report: dict[str, object] = {
        "model_name": "logistic_regression",
        "labels": list(labels),
        "random_state": random_state,
        "versions": {},
    }

    for version_name, payload in dataset_versions.items():
        (x_train, y_train), (x_val, y_val) = payload
        val_result = run_baseline_experiment(
            model_name="logistic_regression",
            x_train=x_train,
            y_train=y_train,
            x_eval=x_val,
            y_eval=y_val,
            labels=labels,
            random_state=random_state,
            class_weight="balanced",
        )
        report["versions"][version_name] = {
            "train_shape_flat": [int(x_train.shape[0]), int(x_train.shape[1])],
            "val_shape_flat": [int(x_val.shape[0]), int(x_val.shape[1])],
            "val_missing_labels": _missing_labels_text(y_val, labels),
            "val_metrics": val_result["metrics"],
        }

    best_version = max(
        report["versions"].items(),
        key=lambda item: float(item[1]["val_metrics"]["macro_f1"]),
    )[0]
    report["best_dataset_version_by_val_macro_f1"] = best_version

    out_dir = PROJECT_ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "stage4_baseline_summary.json"
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    md_lines = [
        "# Этап 4: baseline-модель",
        "",
        "Baseline по ТЗ: `LogisticRegression` на flattened временных окнах.",
        "Сравнение `raw` vs `improved` выполняется только на `val`.",
        "`test` на этом этапе не используется.",
    ]

    for version_name, version_payload in report["versions"].items():
        metrics = version_payload["val_metrics"]
        md_lines.extend(
            [
                "",
                f"## Версия данных: {version_name}",
                f"- Train(flat): {version_payload['train_shape_flat']}",
                f"- Val(flat): {version_payload['val_shape_flat']}",
                f"- Метрики val: {_metric_line(metrics)}",
            ]
        )
        if version_payload["val_missing_labels"] is not None:
            md_lines.append(f"- Важно: в `val` отсутствуют классы: {version_payload['val_missing_labels']}")
        if "roc_auc_note" in metrics:
            md_lines.append(f"- Примечание по ROC AUC: {metrics['roc_auc_note']}")
        md_lines.extend(
            [
                "- Classification report (val):",
                "```text",
                str(metrics["classification_report_text"]).rstrip(),
                "```",
            ]
        )

    md_lines.extend(
        [
            "",
            f"## Лучшая версия данных по val macro_f1: {best_version}",
            "- Главный ориентир выбора: `macro_f1`.",
            "- `balanced_accuracy` и `roc_auc_ovr_macro` используются как дополнительные контрольные метрики.",
        ]
    )

    md_path = out_dir / "stage4_baseline_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Сохранено: {json_path}")
    print(f"Сохранено: {md_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Дополнительная rolling-диагностика устойчивости внутри официального train-сегмента."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_datasets
from src.data.preprocessing import fit_improved_preprocessor, transform_improved_labeled
from src.data.rolling_validation import (
    build_train_rolling_window_plan,
    materialize_rolling_fold,
    save_rolling_plan_artifact,
    validate_rolling_no_index_leakage,
)
from src.data.splits import flatten_windows, get_class_distribution
from src.models.baseline import run_baseline_experiment
from src.training.hybrid_training import run_hybrid_experiment
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility

N_FOLDS = 3


def _missing_labels_text(y_values: np.ndarray, labels: tuple[int, ...]) -> str | None:
    present = {int(value) for value in y_values}
    missing = [label for label in labels if label not in present]
    if not missing:
        return None
    return ", ".join(str(label) for label in missing)


def _aggregate_fold_metrics(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [item for item in fold_results if item["status"] == "OK"]
    result: dict[str, Any] = {
        "folds_total": len(fold_results),
        "ok_folds": len(successful),
        "failed_folds": len(fold_results) - len(successful),
    }

    if not successful:
        result["macro_f1_mean"] = None
        result["macro_f1_std"] = None
        result["macro_f1_min"] = None
        result["macro_f1_max"] = None
        result["balanced_accuracy_mean"] = None
        result["balanced_accuracy_std"] = None
        result["balanced_accuracy_min"] = None
        result["balanced_accuracy_max"] = None
        result["folds_with_roc_auc"] = 0
        result["roc_auc_mean"] = None
        result["roc_auc_std"] = None
        return result

    macro_f1_values = np.asarray([float(item["val_metrics"]["macro_f1"]) for item in successful], dtype=np.float64)
    balanced_values = np.asarray([float(item["val_metrics"]["balanced_accuracy"]) for item in successful], dtype=np.float64)
    roc_values = [
        float(item["val_metrics"]["roc_auc_ovr_macro"])
        for item in successful
        if item["val_metrics"]["roc_auc_ovr_macro"] is not None
    ]

    result.update(
        {
            "macro_f1_mean": float(np.mean(macro_f1_values)),
            "macro_f1_std": float(np.std(macro_f1_values)),
            "macro_f1_min": float(np.min(macro_f1_values)),
            "macro_f1_max": float(np.max(macro_f1_values)),
            "balanced_accuracy_mean": float(np.mean(balanced_values)),
            "balanced_accuracy_std": float(np.std(balanced_values)),
            "balanced_accuracy_min": float(np.min(balanced_values)),
            "balanced_accuracy_max": float(np.max(balanced_values)),
            "folds_with_roc_auc": len(roc_values),
        }
    )
    if roc_values:
        roc_array = np.asarray(roc_values, dtype=np.float64)
        result["roc_auc_mean"] = float(np.mean(roc_array))
        result["roc_auc_std"] = float(np.std(roc_array))
    else:
        result["roc_auc_mean"] = None
        result["roc_auc_std"] = None
    return result


def _metric_line(metrics: dict[str, Any]) -> str:
    roc_auc = metrics["roc_auc_ovr_macro"]
    roc_text = "n/a" if roc_auc is None else f"{float(roc_auc):.4f}"
    return (
        f"macro_f1={float(metrics['macro_f1']):.4f}, "
        f"balanced_acc={float(metrics['balanced_accuracy']):.4f}, "
        f"roc_auc={roc_text}"
    )


def _summary_line(summary: dict[str, Any]) -> str:
    if summary["ok_folds"] == 0:
        return f"нет успешных фолдов; failed={summary['failed_folds']}"

    roc_mean = summary["roc_auc_mean"]
    roc_text = "n/a" if roc_mean is None else f"{float(roc_mean):.4f}"
    return (
        f"ok/failed={summary['ok_folds']}/{summary['failed_folds']}, "
        f"macro_f1 mean/std={float(summary['macro_f1_mean']):.4f}/{float(summary['macro_f1_std']):.4f}, "
        f"balanced_acc mean/std={float(summary['balanced_accuracy_mean']):.4f}/{float(summary['balanced_accuracy_std']):.4f}, "
        f"roc_auc mean={roc_text}"
    )


def _build_error_analysis(metrics: dict[str, Any]) -> list[str]:
    confusion = metrics["confusion_matrix"]
    labels = [int(label) for label in metrics["labels"]]
    lines: list[str] = []

    for row_idx, true_label in enumerate(labels):
        row = list(confusion[row_idx])
        support = int(sum(row))
        if support == 0:
            lines.append(f"- Класс {true_label}: отсутствует в текущем fold-val.")
            continue

        correct = int(row[row_idx])
        mistakes = {labels[col_idx]: int(value) for col_idx, value in enumerate(row) if col_idx != row_idx and int(value) > 0}
        if mistakes:
            dominant_target = max(mistakes.items(), key=lambda item: item[1])
            lines.append(
                f"- Класс {true_label}: верно {correct}/{support}, чаще всего путается с классом "
                f"{dominant_target[0]} ({dominant_target[1]} раз)."
            )
        else:
            lines.append(f"- Класс {true_label}: ошибок нет, верно {correct}/{support}.")

    return lines


def _run_baseline_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: tuple[int, ...],
    random_state: int,
) -> dict[str, Any]:
    result = run_baseline_experiment(
        model_name="logistic_regression",
        x_train=flatten_windows(x_train),
        y_train=y_train,
        x_eval=flatten_windows(x_val),
        y_eval=y_val,
        labels=labels,
        random_state=random_state,
        class_weight="balanced",
    )
    return result["metrics"]


def _run_hybrid_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    labels: tuple[int, ...],
    hybrid_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> dict[str, Any]:
    result = run_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=labels,
        hybrid_cfg=hybrid_cfg,
        training_cfg=training_cfg,
    )
    return {
        "metrics": result["metrics"],
        "history": result["history"],
    }


def _fold_has_multiple_classes(y_values: np.ndarray) -> bool:
    return int(np.unique(y_values).shape[0]) >= 2


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    initialize_reproducibility(config, PROJECT_ROOT, stage_name="rolling_diagnostics", use_tensorflow=True)

    labels = tuple(int(label) for label in config["task"]["labels"])
    target_col = str(config["task"]["target_col"])
    sequence_cfg = config["sequence"]
    split_cfg = config["split"]
    prep_cfg = config["preprocessing"]["improved"]
    hybrid_cfg = dict(config["training"]["hybrid"])
    training_cfg = {
        "early_stopping_patience": int(config["training"]["early_stopping_patience"]),
        "reduce_lr_patience": int(config["training"]["reduce_lr_patience"]),
        "reduce_lr_factor": float(config["training"]["reduce_lr_factor"]),
    }

    labeled_df, _ = load_datasets(config)
    plan = build_train_rolling_window_plan(
        df=labeled_df,
        target_col=target_col,
        window_size=int(sequence_cfg["T"]),
        stride=int(sequence_cfg["stride"]),
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        n_folds=N_FOLDS,
    )
    validate_rolling_no_index_leakage(plan)

    plan_path = PROJECT_ROOT / "output" / "artifacts" / "rolling_diagnostics_plan.json"
    save_rolling_plan_artifact(plan, plan_path)

    report: dict[str, Any] = {
        "kind": "rolling_diagnostics",
        "dataset_version": "improved",
        "n_folds": N_FOLDS,
        "note": (
            "Это дополнительная диагностическая проверка устойчивости внутри официального train-сегмента. "
            "Она не заменяет основной validation по ТЗ."
        ),
        "plan_artifact": str(plan_path.relative_to(PROJECT_ROOT)),
        "models": {
            "baseline_logistic_regression": {"folds": []},
            "hybrid_cnn_gru_dense": {"folds": []},
        },
    }

    for fold_plan in plan["folds"]:
        fold_index = int(fold_plan["fold_index"])
        train_fit_df = labeled_df.loc[fold_plan["train_row_positions_for_fit"]].reset_index(drop=True)
        improved_preprocessor = fit_improved_preprocessor(
            train_fit_df,
            target_col=target_col,
            clip_quantiles=tuple(prep_cfg["clip_quantiles"]),
        )
        x_rows_improved, y_rows = transform_improved_labeled(labeled_df, improved_preprocessor, target_col=target_col)
        fold_splits = materialize_rolling_fold(
            features=x_rows_improved,
            targets=y_rows,
            fold_plan=fold_plan,
            window_size=int(sequence_cfg["T"]),
        )

        x_train, y_train = fold_splits["train"]
        x_val, y_val = fold_splits["val"]

        common_payload = {
            "fold_index": fold_index,
            "train_shape": [int(x_train.shape[0]), int(x_train.shape[1]), int(x_train.shape[2])],
            "val_shape": [int(x_val.shape[0]), int(x_val.shape[1]), int(x_val.shape[2])],
            "train_class_distribution": get_class_distribution(y_train),
            "val_class_distribution": get_class_distribution(y_val),
            "val_missing_labels": _missing_labels_text(y_val, labels),
        }

        if not _fold_has_multiple_classes(y_train):
            error_text = "В train-фолде меньше двух классов; обучение модели пропущено"
            report["models"]["baseline_logistic_regression"]["folds"].append(
                {
                    **common_payload,
                    "status": "FAIL",
                    "error": error_text,
                }
            )
            report["models"]["hybrid_cnn_gru_dense"]["folds"].append(
                {
                    **common_payload,
                    "status": "FAIL",
                    "error": error_text,
                }
            )
            continue

        try:
            baseline_metrics = _run_baseline_fold(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                labels=labels,
                random_state=int(split_cfg["random_state"]),
            )
            report["models"]["baseline_logistic_regression"]["folds"].append(
                {
                    **common_payload,
                    "status": "OK",
                    "val_metrics": baseline_metrics,
                }
            )
        except Exception as exc:
            report["models"]["baseline_logistic_regression"]["folds"].append(
                {
                    **common_payload,
                    "status": "FAIL",
                    "error": str(exc),
                }
            )

        try:
            hybrid_result = _run_hybrid_fold(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                labels=labels,
                hybrid_cfg=hybrid_cfg,
                training_cfg=training_cfg,
            )
            report["models"]["hybrid_cnn_gru_dense"]["folds"].append(
                {
                    **common_payload,
                    "status": "OK",
                    "history": hybrid_result["history"],
                    "val_metrics": hybrid_result["metrics"],
                }
            )
        except Exception as exc:
            report["models"]["hybrid_cnn_gru_dense"]["folds"].append(
                {
                    **common_payload,
                    "status": "FAIL",
                    "error": str(exc),
                }
            )

    for model_payload in report["models"].values():
        model_payload["summary"] = _aggregate_fold_metrics(model_payload["folds"])

    out_dir = PROJECT_ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "rolling_diagnostics_summary.json"
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    md_lines = [
        "# Дополнительная rolling-диагностика",
        "",
        "- Диагностика выполнена только на версии данных `improved`.",
        f"- Число expanding-window фолдов: {N_FOLDS}.",
        "- Это дополнительная проверка устойчивости; основной `val` по ТЗ не заменяется.",
        f"- Артефакт плана: `{report['plan_artifact']}`",
    ]

    for model_name, model_payload in report["models"].items():
        md_lines.extend(
            [
                "",
                f"## Модель: {model_name}",
                f"- Сводка: {_summary_line(model_payload['summary'])}",
            ]
        )

        for fold_payload in model_payload["folds"]:
            md_lines.extend(
                [
                    "",
                    f"### Fold {fold_payload['fold_index']}",
                    f"- Train: {fold_payload['train_shape']}",
                    f"- Val: {fold_payload['val_shape']}",
                    f"- Классы train: {fold_payload['train_class_distribution']}",
                    f"- Классы val: {fold_payload['val_class_distribution']}",
                    f"- Статус: {fold_payload['status']}",
                ]
            )
            if fold_payload["val_missing_labels"] is not None:
                md_lines.append(f"- Важно: в `fold-val` отсутствуют классы: {fold_payload['val_missing_labels']}")
            if fold_payload["status"] != "OK":
                md_lines.append(f"- Причина: {fold_payload['error']}")
                continue

            metrics = fold_payload["val_metrics"]
            md_lines.append(f"- Метрики fold-val: {_metric_line(metrics)}")
            if "history" in fold_payload:
                md_lines.append(
                    f"- История обучения: лучшая эпоха {fold_payload['history']['best_epoch']}, "
                    f"всего эпох {fold_payload['history']['epochs_ran']}"
                )
            if "roc_auc_note" in metrics:
                md_lines.append(f"- Примечание по ROC AUC: {metrics['roc_auc_note']}")
            md_lines.extend(
                [
                    "- Confusion matrix (fold-val):",
                    "```text",
                    str(metrics["confusion_matrix"]),
                    "```",
                    "- Краткий анализ ошибок:",
                ]
            )
            md_lines.extend(_build_error_analysis(metrics))

    md_path = out_dir / "rolling_diagnostics_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Сохранено: {json_path}")
    print(f"Сохранено: {md_path}")


if __name__ == "__main__":
    main()

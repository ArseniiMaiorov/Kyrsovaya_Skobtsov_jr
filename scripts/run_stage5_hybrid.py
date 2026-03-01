#!/usr/bin/env python3
"""Этап 5: гибридные нейросетевые эксперименты и расширенный анализ validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation import augment_dataset
from src.data.io import load_datasets
from src.data.preprocessing import fit_improved_preprocessor, fit_raw_preprocessor, transform_improved_labeled, transform_raw_labeled
from src.data.splits import build_window_split_plan, materialize_labeled_window_splits
from src.metrics.metrics import plot_multiclass_roc_curves
from src.training.hybrid_training import (
    plot_attention_weights,
    run_hybrid_experiment,
    visualize_conv_filters,
    visualize_hidden_representations,
)
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Этап 5: гибридные эксперименты")
    parser.add_argument(
        "--rnn_type",
        choices=("gru", "lstm", "bi_gru", "bi_lstm"),
        default=None,
        help="Если задано, сравнение по типу RNN ограничивается одним вариантом.",
    )
    return parser.parse_args()


def _collect_window_versions(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
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

    return {
        "raw": {
            "train": raw_splits["train"],
            "val": raw_splits["val"],
        },
        "improved": {
            "train": improved_splits["train"],
            "val": improved_splits["val"],
        },
    }


def _missing_labels_text(y_values: object, labels: tuple[int, ...]) -> str | None:
    present = {int(value) for value in y_values}
    missing = [label for label in labels if label not in present]
    if not missing:
        return None
    return ", ".join(str(label) for label in missing)


def _metric_line(metrics: dict[str, Any]) -> str:
    macro_f1 = float(metrics["macro_f1"])
    bal_acc = float(metrics["balanced_accuracy"])
    roc_auc = metrics["roc_auc_ovr_macro"]
    roc_text = "n/a" if roc_auc is None else f"{float(roc_auc):.4f}"
    return f"macro_f1={macro_f1:.4f}, balanced_acc={bal_acc:.4f}, roc_auc={roc_text}"


def _build_change_summary(raw_metrics: dict[str, Any], improved_metrics: dict[str, Any]) -> str:
    raw_f1 = float(raw_metrics["macro_f1"])
    improved_f1 = float(improved_metrics["macro_f1"])
    delta = improved_f1 - raw_f1

    if delta > 0.01:
        return f"`improved` лучше `raw` по macro-F1 на {delta:.4f}; winsorize + RobustScaler стабилизировали признаки."
    if delta < -0.01:
        return f"`improved` хуже `raw` по macro-F1 на {abs(delta):.4f}; на текущем split преобразование ухудшило разделимость."
    return "`raw` и `improved` близки по macro-F1; выигрыш от предобработки на текущем split незначителен."


def _run_single_hybrid(
    *,
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    labels: tuple[int, ...],
    hybrid_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    artifacts_dir: Path | None = None,
    roc_path: Path | None = None,
) -> dict[str, Any]:
    result = run_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=labels,
        hybrid_cfg=hybrid_cfg,
        training_cfg=training_cfg,
        artifacts_dir=artifacts_dir,
    )

    roc_plot = None
    if roc_path is not None:
        roc_info = plot_multiclass_roc_curves(
            y_true=y_val,
            y_proba=result["y_proba"],
            labels=labels,
            save_path=roc_path,
        )
        roc_plot = {
            "path": str(Path(roc_info["path"]).relative_to(PROJECT_ROOT)),
            "present_labels": list(roc_info["present_labels"]),
            "per_class_auc": dict(roc_info["per_class_auc"]),
        }

    return {
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1]), int(x_train.shape[2])],
        "val_shape": [int(x_val.shape[0]), int(x_val.shape[1]), int(x_val.shape[2])],
        "history": result["history"],
        "history_artifacts": {
            key: str(Path(value).relative_to(PROJECT_ROOT)) for key, value in result["artifacts"].items()
        },
        "class_weight": {str(key): float(value) for key, value in result["class_weight"].items()},
        "val_metrics": result["metrics"],
        "roc_curve": roc_plot,
        "model": result["model"],
        "y_proba": result["y_proba"],
    }


def main() -> None:
    args = _parse_args()

    config = load_config(PROJECT_ROOT / "config.yaml")
    initialize_reproducibility(config, PROJECT_ROOT, stage_name="stage5_hybrid", use_tensorflow=True)

    labels = tuple(int(x) for x in config["task"]["labels"])
    base_hybrid_cfg = dict(config["training"]["hybrid"])
    training_cfg = {
        "early_stopping_patience": int(config["training"]["early_stopping_patience"]),
        "reduce_lr_patience": int(config["training"]["reduce_lr_patience"]),
        "reduce_lr_factor": float(config["training"]["reduce_lr_factor"]),
    }
    augmentation_cfg = dict(config.get("augmentation", {}))
    dataset_versions = _collect_window_versions(config)

    out_models_dir = PROJECT_ROOT / "output" / "models"
    out_models_dir.mkdir(parents=True, exist_ok=True)
    out_artifacts_dir = PROJECT_ROOT / "output" / "artifacts"
    out_artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "model_name": "hybrid_cnn_sequence_dense",
        "labels": list(labels),
        "hybrid_config": base_hybrid_cfg,
        "versions": {},
        "rnn_comparison": {},
        "augmentation_experiment": {},
        "attention_experiment": {},
    }

    for version_name, payload in dataset_versions.items():
        (x_train, y_train) = payload["train"]
        (x_val, y_val) = payload["val"]

        version_cfg = dict(base_hybrid_cfg)
        version_result = _run_single_hybrid(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            labels=labels,
            hybrid_cfg=version_cfg,
            training_cfg=training_cfg,
            artifacts_dir=out_artifacts_dir / f"stage5_{version_name}",
            roc_path=figures_dir / f"stage5_{version_name}_roc.png",
        )

        model_path = out_models_dir / f"stage5_hybrid_{version_name}.keras"
        version_result["model"].save(model_path)

        analysis_artifacts: dict[str, str] = {}
        if version_name == "improved":
            conv_plot_path = figures_dir / "stage5_improved_conv_filters.png"
            hidden_plot_path = figures_dir / "stage5_improved_hidden_repr.png"
            analysis_artifacts["conv_filters"] = str(
                Path(visualize_conv_filters(version_result["model"], conv_plot_path)).relative_to(PROJECT_ROOT)
            )
            analysis_artifacts["hidden_representations"] = str(
                Path(visualize_hidden_representations(version_result["model"], x_val, y_val, hidden_plot_path)).relative_to(
                    PROJECT_ROOT
                )
            )

        report["versions"][version_name] = {
            "train_shape": version_result["train_shape"],
            "val_shape": version_result["val_shape"],
            "val_missing_labels": _missing_labels_text(y_val, labels),
            "class_weight": version_result["class_weight"],
            "history": version_result["history"],
            "history_artifacts": version_result["history_artifacts"],
            "val_metrics": version_result["val_metrics"],
            "roc_curve": version_result["roc_curve"],
            "analysis_artifacts": analysis_artifacts,
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "rnn_type": version_cfg.get("rnn_type", "gru"),
            "use_attention": bool(version_cfg.get("use_attention", False)),
        }

    best_version = max(
        report["versions"].items(),
        key=lambda item: float(item[1]["val_metrics"]["macro_f1"]),
    )[0]
    report["best_dataset_version_by_val_macro_f1"] = best_version
    report["raw_vs_improved_summary"] = _build_change_summary(
        report["versions"]["raw"]["val_metrics"],
        report["versions"]["improved"]["val_metrics"],
    )

    # Сравнение по типам рекуррентного слоя на improved.
    (x_train_improved, y_train_improved) = dataset_versions["improved"]["train"]
    (x_val_improved, y_val_improved) = dataset_versions["improved"]["val"]
    rnn_types = [args.rnn_type] if args.rnn_type is not None else ["gru", "lstm", "bi_gru", "bi_lstm"]

    for rnn_type in rnn_types:
        rnn_cfg = dict(base_hybrid_cfg)
        rnn_cfg["rnn_type"] = rnn_type
        rnn_cfg["use_attention"] = False
        rnn_result = _run_single_hybrid(
            x_train=x_train_improved,
            y_train=y_train_improved,
            x_val=x_val_improved,
            y_val=y_val_improved,
            labels=labels,
            hybrid_cfg=rnn_cfg,
            training_cfg=training_cfg,
            artifacts_dir=None,
            roc_path=figures_dir / f"stage5_rnn_{rnn_type}_roc.png",
        )
        report["rnn_comparison"][rnn_type] = {
            "train_shape": rnn_result["train_shape"],
            "val_shape": rnn_result["val_shape"],
            "parameter_count": int(rnn_result["model"].count_params()),
            "history": rnn_result["history"],
            "val_metrics": rnn_result["val_metrics"],
            "roc_curve": rnn_result["roc_curve"],
        }

    report["best_rnn_type_by_val_macro_f1"] = max(
        report["rnn_comparison"].items(),
        key=lambda item: float(item[1]["val_metrics"]["macro_f1"]),
    )[0]

    # Аугментация train-окон на improved.
    if bool(augmentation_cfg.get("enabled", False)):
        params = {
            "std": float(augmentation_cfg.get("noise_std", 0.01)),
            "min_scale": float(augmentation_cfg.get("scale_min", 0.9)),
            "max_scale": float(augmentation_cfg.get("scale_max", 1.1)),
            "shift_min": int(augmentation_cfg.get("shift_min", -5)),
            "shift_max": int(augmentation_cfg.get("shift_max", 5)),
        }
        x_train_aug, y_train_aug = augment_dataset(
            x_train_improved,
            y_train_improved,
            aug_factor=int(augmentation_cfg.get("aug_factor", 2)),
            methods=tuple(str(value) for value in augmentation_cfg.get("methods", ["noise", "scale"])),
            params=params,
        )
        aug_cfg = dict(base_hybrid_cfg)
        augmented_result = _run_single_hybrid(
            x_train=x_train_aug,
            y_train=y_train_aug,
            x_val=x_val_improved,
            y_val=y_val_improved,
            labels=labels,
            hybrid_cfg=aug_cfg,
            training_cfg=training_cfg,
            artifacts_dir=out_artifacts_dir / "stage5_improved_augmented",
            roc_path=figures_dir / "stage5_improved_augmented_roc.png",
        )
        report["augmentation_experiment"] = {
            "enabled": True,
            "config": augmentation_cfg,
            "train_windows_before": int(x_train_improved.shape[0]),
            "train_windows_after": int(x_train_aug.shape[0]),
            "no_augmentation_val_metrics": report["versions"]["improved"]["val_metrics"],
            "with_augmentation_val_metrics": augmented_result["val_metrics"],
            "with_augmentation_history": augmented_result["history"],
            "with_augmentation_history_artifacts": augmented_result["history_artifacts"],
            "with_augmentation_roc_curve": augmented_result["roc_curve"],
        }
    else:
        report["augmentation_experiment"] = {"enabled": False}

    # Сравнение без attention и с attention на improved.
    attention_cfg = dict(base_hybrid_cfg)
    attention_cfg["use_attention"] = True
    attention_result = _run_single_hybrid(
        x_train=x_train_improved,
        y_train=y_train_improved,
        x_val=x_val_improved,
        y_val=y_val_improved,
        labels=labels,
        hybrid_cfg=attention_cfg,
        training_cfg=training_cfg,
        artifacts_dir=out_artifacts_dir / "stage5_attention",
        roc_path=figures_dir / "stage5_attention_roc.png",
    )
    attention_model_path = out_models_dir / "stage5_hybrid_attention.keras"
    attention_result["model"].save(attention_model_path)
    attention_artifacts: dict[str, str] = {}
    try:
        attention_plot_path = figures_dir / "stage5_attention_weights.png"
        attention_artifacts["attention_weights"] = str(
            Path(plot_attention_weights(attention_result["model"], x_val_improved[0], attention_plot_path)).relative_to(
                PROJECT_ROOT
            )
        )
    except Exception:
        attention_artifacts["attention_weights"] = "n/a"

    report["attention_experiment"] = {
        "baseline_no_attention_val_metrics": report["versions"]["improved"]["val_metrics"],
        "with_attention_val_metrics": attention_result["val_metrics"],
        "with_attention_history": attention_result["history"],
        "with_attention_history_artifacts": attention_result["history_artifacts"],
        "with_attention_roc_curve": attention_result["roc_curve"],
        "with_attention_artifacts": attention_artifacts,
        "with_attention_model_path": str(attention_model_path.relative_to(PROJECT_ROOT)),
    }

    out_dir = PROJECT_ROOT / "reports" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "stage5_hybrid_summary.json"
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    md_lines = [
        "# Этап 5: гибридные нейросетевые модели",
        "",
        "Основной протокол: сравнение `raw` и `improved` только на `val` без использования `test`.",
        "Дополнительно выполнены: ROC AUC-анализ, визуализация обучения, аугментация, сравнение типов RNN и сравнение с attention.",
        "",
        "## Базовый конфиг гибридной модели",
        f"- rnn_type: {base_hybrid_cfg.get('rnn_type', 'gru')}",
        f"- use_attention: {base_hybrid_cfg.get('use_attention', False)}",
        f"- n_conv_layers: {base_hybrid_cfg['n_conv_layers']}",
        f"- conv_filters: {base_hybrid_cfg['conv_filters']}",
        f"- conv_kernel_size: {base_hybrid_cfg['conv_kernel_size']}",
        f"- n_gru_layers: {base_hybrid_cfg['n_gru_layers']}",
        f"- gru_units: {base_hybrid_cfg['gru_units']}",
        f"- n_dense_layers: {base_hybrid_cfg['n_dense_layers']}",
        f"- dense_units: {base_hybrid_cfg['dense_units']}",
        f"- activation: {base_hybrid_cfg['activation']}",
        f"- optimizer: {base_hybrid_cfg['optimizer']}",
        f"- loss: {base_hybrid_cfg['loss']}",
        f"- batch_size: {base_hybrid_cfg['batch_size']}",
        f"- max_epochs: {base_hybrid_cfg['max_epochs']}",
    ]

    for version_name, version_payload in report["versions"].items():
        metrics = version_payload["val_metrics"]
        md_lines.extend(
            [
                "",
                f"## Версия данных: {version_name}",
                f"- Train: {version_payload['train_shape']}",
                f"- Val: {version_payload['val_shape']}",
                f"- Метрики val: {_metric_line(metrics)}",
                f"- Лучшая эпоха: {version_payload['history']['best_epoch']}",
                f"- Эпох выполнено: {version_payload['history']['epochs_ran']}",
                f"- Сохранённая модель: `{version_payload['model_path']}`",
            ]
        )
        if version_payload["roc_curve"] is not None:
            md_lines.append(f"- ROC-кривые: `{version_payload['roc_curve']['path']}`")
        if version_payload["val_missing_labels"] is not None:
            md_lines.append(f"- Важно: в `val` отсутствуют классы: {version_payload['val_missing_labels']}")
        if "roc_auc_note" in metrics:
            md_lines.append(f"- Примечание по ROC AUC: {metrics['roc_auc_note']}")
        for artifact_name, artifact_path in version_payload["history_artifacts"].items():
            md_lines.append(f"- Артефакт {artifact_name}: `{artifact_path}`")
        for artifact_name, artifact_path in version_payload["analysis_artifacts"].items():
            md_lines.append(f"- Артефакт {artifact_name}: `{artifact_path}`")

    md_lines.extend(
        [
            "",
            f"## Лучшая версия данных по val macro_f1: {best_version}",
            f"- Вывод по raw vs improved: {report['raw_vs_improved_summary']}",
            "",
            "## Сравнение типов рекуррентного слоя (improved)",
        ]
    )
    for rnn_type, payload in report["rnn_comparison"].items():
        md_lines.append(
            f"- {rnn_type}: {_metric_line(payload['val_metrics'])}, params={payload['parameter_count']}"
        )
    md_lines.append(f"- Лучший тип RNN: {report['best_rnn_type_by_val_macro_f1']}")

    md_lines.extend(["", "## Аугментация временных окон"])
    if report["augmentation_experiment"]["enabled"]:
        md_lines.extend(
            [
                f"- Train окон до аугментации: {report['augmentation_experiment']['train_windows_before']}",
                f"- Train окон после аугментации: {report['augmentation_experiment']['train_windows_after']}",
                f"- Без аугментации: {_metric_line(report['augmentation_experiment']['no_augmentation_val_metrics'])}",
                f"- С аугментацией: {_metric_line(report['augmentation_experiment']['with_augmentation_val_metrics'])}",
            ]
        )
    else:
        md_lines.append("- Аугментация отключена в config.")

    md_lines.extend(
        [
            "",
            "## Сравнение без attention и с attention",
            f"- Без attention: {_metric_line(report['attention_experiment']['baseline_no_attention_val_metrics'])}",
            f"- С attention: {_metric_line(report['attention_experiment']['with_attention_val_metrics'])}",
        ]
    )
    for artifact_name, artifact_path in report["attention_experiment"]["with_attention_artifacts"].items():
        md_lines.append(f"- Артефакт {artifact_name}: `{artifact_path}`")

    md_path = out_dir / "stage5_hybrid_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Сохранено: {json_path}")
    print(f"Сохранено: {md_path}")


if __name__ == "__main__":
    main()

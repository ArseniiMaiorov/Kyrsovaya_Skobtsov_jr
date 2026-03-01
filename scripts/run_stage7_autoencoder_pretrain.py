#!/usr/bin/env python3
"""Этап 7: AE-предобучение на неразмеченных окнах и fine-tuning классификатора."""

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
from src.data.preprocessing import fit_improved_preprocessor, transform_improved_labeled, transform_improved_unlabeled
from src.data.splits import (
    build_inference_window_plan,
    build_window_split_plan,
    materialize_labeled_window_splits,
    materialize_unlabeled_windows,
)
from src.models.autoencoder import extract_encoder_weights
from src.training.autoencoder_training import (
    run_autoencoder_pretraining,
    run_pretrained_hybrid_experiment,
)
from src.training.ga_search import GENE_NAMES
from src.training.hybrid_training import run_hybrid_experiment
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility, set_global_seed

AUTOENCODER_MODEL_PATH = PROJECT_ROOT / "output" / "models" / "stage7_autoencoder.keras"
SCRATCH_MODEL_PATH = PROJECT_ROOT / "output" / "models" / "stage7_hybrid_scratch.keras"
PRETRAINED_MODEL_PATH = PROJECT_ROOT / "output" / "models" / "stage7_hybrid_ae_finetuned.keras"
SUMMARY_JSON_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage7_autoencoder_pretrain_summary.json"
SUMMARY_MD_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage7_autoencoder_pretrain_summary.md"
STAGE6_BEST_GENOME_PATH = PROJECT_ROOT / "output" / "artifacts" / "stage6_best_genome.json"


def _collect_stage7_data(config: dict[str, Any]) -> dict[str, Any]:
    labeled_df, unlabeled_df = load_datasets(config)

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
    labeled_splits = materialize_labeled_window_splits(x_rows_improved, y_rows, plan)

    unlabeled_features = transform_improved_unlabeled(unlabeled_df, improved_preprocessor)
    unlabeled_plan = build_inference_window_plan(
        unlabeled_df,
        window_size=int(seq_cfg["T"]),
        stride=int(seq_cfg["stride"]),
    )
    x_unlabeled = materialize_unlabeled_windows(unlabeled_features, unlabeled_plan)

    return {
        "x_train": labeled_splits["train"][0],
        "y_train": labeled_splits["train"][1],
        "x_val": labeled_splits["val"][0],
        "y_val": labeled_splits["val"][1],
        "x_unlabeled": x_unlabeled,
        "window_counts": {
            "train": int(labeled_splits["train"][0].shape[0]),
            "val": int(labeled_splits["val"][0].shape[0]),
            "unlabeled": int(x_unlabeled.shape[0]),
        },
    }


def _resolve_stage7_hybrid_config(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    hybrid_cfg = dict(config["training"]["hybrid"])
    autoencoder_cfg = config["training"]["autoencoder"]

    if not bool(autoencoder_cfg["use_stage6_best_genome"]):
        return hybrid_cfg, "training.hybrid"

    if not STAGE6_BEST_GENOME_PATH.exists():
        return hybrid_cfg, "training.hybrid (stage6_best_genome не найден)"

    payload = json.loads(STAGE6_BEST_GENOME_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Артефакт stage6_best_genome.json должен содержать JSON-словарь")

    for gene_name in GENE_NAMES:
        if gene_name not in payload:
            raise ValueError(f"В stage6_best_genome.json отсутствует ген '{gene_name}'")
        hybrid_cfg[gene_name] = payload[gene_name]

    return hybrid_cfg, "stage6_best_genome"


def _metric_line(metrics: dict[str, Any]) -> str:
    roc_auc = metrics["roc_auc_ovr_macro"]
    roc_text = "n/a" if roc_auc is None else f"{float(roc_auc):.4f}"
    return (
        f"macro_f1={float(metrics['macro_f1']):.4f}, "
        f"balanced_acc={float(metrics['balanced_accuracy']):.4f}, "
        f"roc_auc={roc_text}"
    )


def _save_model(model: tf.keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    repro = initialize_reproducibility(config, PROJECT_ROOT, stage_name="stage7_autoencoder_pretrain", use_tensorflow=True)

    labels = tuple(int(label) for label in config["task"]["labels"])
    global_seed = int(repro["global_seed"])
    training_cfg = {
        "early_stopping_patience": int(config["training"]["early_stopping_patience"]),
        "reduce_lr_patience": int(config["training"]["reduce_lr_patience"]),
        "reduce_lr_factor": float(config["training"]["reduce_lr_factor"]),
    }
    autoencoder_cfg = dict(config["training"]["autoencoder"])

    data_bundle = _collect_stage7_data(config)
    hybrid_cfg, encoder_source = _resolve_stage7_hybrid_config(config)

    x_train = data_bundle["x_train"]
    y_train = data_bundle["y_train"]
    x_val = data_bundle["x_val"]
    y_val = data_bundle["y_val"]
    x_unlabeled = data_bundle["x_unlabeled"]

    scratch_seed = global_seed
    set_global_seed(scratch_seed, enable_tensorflow=True)
    scratch_result = run_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=labels,
        hybrid_cfg=hybrid_cfg,
        training_cfg=training_cfg,
    )
    _save_model(scratch_result["model"], SCRATCH_MODEL_PATH)
    tf.keras.backend.clear_session()

    ae_seed = global_seed + 10_000
    set_global_seed(ae_seed, enable_tensorflow=True)
    ae_result = run_autoencoder_pretraining(
        x_unlabeled=x_unlabeled,
        hybrid_cfg=hybrid_cfg,
        autoencoder_cfg=autoencoder_cfg,
        training_cfg=training_cfg,
    )
    _save_model(ae_result["model"], AUTOENCODER_MODEL_PATH)
    encoder_weights = extract_encoder_weights(ae_result["model"])
    tf.keras.backend.clear_session()

    finetune_seed = global_seed + 20_000
    set_global_seed(finetune_seed, enable_tensorflow=True)
    finetuned_result = run_pretrained_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=labels,
        hybrid_cfg=hybrid_cfg,
        training_cfg=training_cfg,
        encoder_weights=encoder_weights,
    )
    _save_model(finetuned_result["model"], PRETRAINED_MODEL_PATH)
    tf.keras.backend.clear_session()

    scratch_macro = float(scratch_result["metrics"]["macro_f1"])
    finetuned_macro = float(finetuned_result["metrics"]["macro_f1"])
    scratch_balanced = float(scratch_result["metrics"]["balanced_accuracy"])
    finetuned_balanced = float(finetuned_result["metrics"]["balanced_accuracy"])

    comparison = {
        "macro_f1_gain_vs_scratch": finetuned_macro - scratch_macro,
        "balanced_accuracy_gain_vs_scratch": finetuned_balanced - scratch_balanced,
        "better_model_by_macro_f1": "ae_pretrained" if finetuned_macro > scratch_macro else "scratch_or_equal",
    }

    report: dict[str, Any] = {
        "dataset_version": "improved",
        "encoder_source": encoder_source,
        "resolved_hybrid_config": hybrid_cfg,
        "window_counts": data_bundle["window_counts"],
        "ae_pretraining": {
            "seed": ae_seed,
            "split_summary": ae_result["split_summary"],
            "history": ae_result["history"],
            "reconstruction_metrics": ae_result["reconstruction_metrics"],
            "model_path": str(AUTOENCODER_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
        "scratch_classifier": {
            "seed": scratch_seed,
            "history": scratch_result["history"],
            "val_metrics": scratch_result["metrics"],
            "model_path": str(SCRATCH_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
        "ae_finetuned_classifier": {
            "seed": finetune_seed,
            "history": finetuned_result["history"],
            "val_metrics": finetuned_result["metrics"],
            "transferred_layers": finetuned_result["transferred_layers"],
            "model_path": str(PRETRAINED_MODEL_PATH.relative_to(PROJECT_ROOT)),
        },
        "comparison": comparison,
    }

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    md_lines = [
        "# Этап 7: AE-предобучение и fine-tuning",
        "",
        "- Предобучение выполняется на неразмеченных окнах `improved`.",
        "- Статистики `improved` обучаются на размеченном `train` и затем применяются к unlabeled.",
        "- Сравнение выполняется на одном и том же официальном `val`: `с нуля` vs `после AE`.",
        f"- Источник encoder-конфига: `{encoder_source}`",
        "",
        "## Окна",
        f"- Train (labeled): {report['window_counts']['train']}",
        f"- Val (labeled): {report['window_counts']['val']}",
        f"- Unlabeled (AE): {report['window_counts']['unlabeled']}",
        "",
        "## AE-предобучение",
        f"- Seed: {report['ae_pretraining']['seed']}",
        f"- Train windows: {report['ae_pretraining']['split_summary']['train_windows']}",
        f"- Val windows: {report['ae_pretraining']['split_summary']['val_windows']}",
        f"- mean_reconstruction_mse: {report['ae_pretraining']['reconstruction_metrics']['mean_reconstruction_mse']:.6f}",
        f"- Лучшая эпоха: {report['ae_pretraining']['history']['best_epoch']}",
        f"- Эпох выполнено: {report['ae_pretraining']['history']['epochs_ran']}",
        f"- Сохранённая модель: `{report['ae_pretraining']['model_path']}`",
        "",
        "## Классификатор с нуля",
        f"- Seed: {report['scratch_classifier']['seed']}",
        f"- Метрики val: {_metric_line(report['scratch_classifier']['val_metrics'])}",
        f"- Лучшая эпоха: {report['scratch_classifier']['history']['best_epoch']}",
        f"- Сохранённая модель: `{report['scratch_classifier']['model_path']}`",
        "",
        "## Классификатор после AE",
        f"- Seed: {report['ae_finetuned_classifier']['seed']}",
        f"- Метрики val: {_metric_line(report['ae_finetuned_classifier']['val_metrics'])}",
        f"- Лучшая эпоха: {report['ae_finetuned_classifier']['history']['best_epoch']}",
        f"- Перенесённые слои: {', '.join(report['ae_finetuned_classifier']['transferred_layers'])}",
        f"- Сохранённая модель: `{report['ae_finetuned_classifier']['model_path']}`",
        "",
        "## Сравнение",
        f"- Прирост macro-F1 после AE: {report['comparison']['macro_f1_gain_vs_scratch']:.4f}",
        f"- Прирост balanced_accuracy после AE: {report['comparison']['balanced_accuracy_gain_vs_scratch']:.4f}",
        f"- Лучшая версия по macro-F1: {report['comparison']['better_model_by_macro_f1']}",
    ]

    for section_name in ("scratch_classifier", "ae_finetuned_classifier"):
        metrics = report[section_name]["val_metrics"]
        if "roc_auc_note" in metrics:
            label = "с нуля" if section_name == "scratch_classifier" else "после AE"
            md_lines.append(f"- Примечание по ROC AUC ({label}): {metrics['roc_auc_note']}")

    with SUMMARY_MD_PATH.open("w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(md_lines) + "\n")

    print(f"Сохранено: {SUMMARY_JSON_PATH}")
    print(f"Сохранено: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()

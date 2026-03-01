#!/usr/bin/env python3
"""Этап 8: одноразовая финальная оценка выбранного пайплайна на test."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_JSON_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage8_final_eval_summary.json"
SUMMARY_MD_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage8_final_eval_summary.md"
FINAL_MODEL_PATH = PROJECT_ROOT / "output" / "models" / "stage8_final_selected.keras"
TEMP_MODEL_PATH = PROJECT_ROOT / "output" / "models" / "stage8_selected_candidate_tmp.keras"

if __name__ == "__main__" and SUMMARY_JSON_PATH.exists():
    print(
        "Этап 8 уже был выполнен: найден reports/experiments/stage8_final_eval_summary.json. "
        "Повторная оценка на test запрещена текущим протоколом."
    )
    raise SystemExit(0)

import numpy as np
import tensorflow as tf

from src.data.io import load_datasets
from src.data.preprocessing import fit_improved_preprocessor, transform_improved_labeled, transform_improved_unlabeled
from src.data.splits import (
    build_inference_window_plan,
    build_window_split_plan,
    materialize_labeled_window_splits,
    materialize_unlabeled_windows,
)
from src.models.autoencoder import extract_encoder_weights
from src.training.autoencoder_training import run_autoencoder_pretraining, run_pretrained_hybrid_experiment
from src.training.hybrid_training import evaluate_hybrid_classifier, run_hybrid_experiment
from src.utils.config import load_config
from src.utils.reproducibility import initialize_reproducibility, set_global_seed


def _collect_stage8_data(config: dict[str, Any]) -> dict[str, Any]:
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
        "x_test": labeled_splits["test"][0],
        "y_test": labeled_splits["test"][1],
        "x_unlabeled": x_unlabeled,
        "window_counts": {
            "train": int(labeled_splits["train"][0].shape[0]),
            "val": int(labeled_splits["val"][0].shape[0]),
            "test": int(labeled_splits["test"][0].shape[0]),
            "unlabeled": int(x_unlabeled.shape[0]),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден обязательный артефакт: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Ожидался JSON-словарь: {path}")
    return payload


def _choose_best_pipeline(project_root: Path) -> dict[str, Any]:
    stage6 = _load_json(project_root / "reports" / "experiments" / "stage6_ga_search_summary.json")
    stage7 = _load_json(project_root / "reports" / "experiments" / "stage7_autoencoder_pretrain_summary.json")

    stage6_candidate = {
        "pipeline_name": "stage6_ga_no_ae",
        "dataset_version": "improved",
        "use_autoencoder": False,
        "hybrid_config": dict(stage6["final_best_run"]["hybrid_config"]),
        "reference_val_metrics": dict(stage6["final_best_run"]["val_metrics"]),
        "reference_seed": int(stage6["final_best_run"]["seed"]),
    }
    stage7_candidate = {
        "pipeline_name": "stage7_ae_pretrained",
        "dataset_version": "improved",
        "use_autoencoder": True,
        "hybrid_config": dict(stage7["resolved_hybrid_config"]),
        "reference_val_metrics": dict(stage7["ae_finetuned_classifier"]["val_metrics"]),
        "reference_seed": int(stage7["ae_finetuned_classifier"]["seed"]),
    }

    candidates = [stage6_candidate, stage7_candidate]

    def rank(candidate: dict[str, Any]) -> tuple[float, float]:
        metrics = candidate["reference_val_metrics"]
        return (float(metrics["macro_f1"]), float(metrics["balanced_accuracy"]))

    return max(candidates, key=rank)


def _run_no_ae_pipeline(
    seed: int,
    labels: tuple[int, ...],
    hybrid_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    set_global_seed(seed, enable_tensorflow=True)
    return run_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=labels,
        hybrid_cfg=hybrid_cfg,
        training_cfg=training_cfg,
    )


def _run_ae_pipeline(
    seed: int,
    labels: tuple[int, ...],
    hybrid_cfg: dict[str, Any],
    autoencoder_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_unlabeled: np.ndarray,
) -> dict[str, Any]:
    ae_seed = seed + 10_000
    set_global_seed(ae_seed, enable_tensorflow=True)
    ae_result = run_autoencoder_pretraining(
        x_unlabeled=x_unlabeled,
        hybrid_cfg=hybrid_cfg,
        autoencoder_cfg=autoencoder_cfg,
        training_cfg=training_cfg,
    )
    encoder_weights = extract_encoder_weights(ae_result["model"])
    ae_info = {
        "seed": ae_seed,
        "history": ae_result["history"],
        "reconstruction_metrics": ae_result["reconstruction_metrics"],
        "split_summary": ae_result["split_summary"],
    }
    tf.keras.backend.clear_session()

    finetune_seed = seed + 20_000
    set_global_seed(finetune_seed, enable_tensorflow=True)
    classifier_result = run_pretrained_hybrid_experiment(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        labels=labels,
        hybrid_cfg=hybrid_cfg,
        training_cfg=training_cfg,
        encoder_weights=encoder_weights,
    )
    classifier_result["ae_pretraining"] = ae_info
    classifier_result["finetune_seed"] = finetune_seed
    return classifier_result


def _run_selected_pipeline(
    candidate: dict[str, Any],
    seed: int,
    labels: tuple[int, ...],
    autoencoder_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    data_bundle: dict[str, Any],
) -> dict[str, Any]:
    hybrid_cfg = dict(candidate["hybrid_config"])
    if candidate["use_autoencoder"]:
        return _run_ae_pipeline(
            seed=seed,
            labels=labels,
            hybrid_cfg=hybrid_cfg,
            autoencoder_cfg=autoencoder_cfg,
            training_cfg=training_cfg,
            x_train=data_bundle["x_train"],
            y_train=data_bundle["y_train"],
            x_val=data_bundle["x_val"],
            y_val=data_bundle["y_val"],
            x_unlabeled=data_bundle["x_unlabeled"],
        )
    return _run_no_ae_pipeline(
        seed=seed,
        labels=labels,
        hybrid_cfg=hybrid_cfg,
        training_cfg=training_cfg,
        x_train=data_bundle["x_train"],
        y_train=data_bundle["y_train"],
        x_val=data_bundle["x_val"],
        y_val=data_bundle["y_val"],
    )


def _seed_rank(metrics: dict[str, Any], seed: int) -> tuple[float, float, int]:
    return (
        round(float(metrics["macro_f1"]), 3),
        float(metrics["balanced_accuracy"]),
        -int(seed),
    )


def _build_error_analysis(metrics: dict[str, Any]) -> dict[str, Any]:
    labels = [int(label) for label in metrics["labels"]]
    conf = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    off_diag = conf.copy()
    np.fill_diagonal(off_diag, 0)

    max_error = int(np.max(off_diag))
    if max_error <= 0:
        return {
            "summary": "На test не обнаружено межклассовых ошибок по confusion matrix.",
            "top_confusion": None,
        }

    true_idx, pred_idx = np.argwhere(off_diag == max_error)[0]
    true_label = labels[int(true_idx)]
    pred_label = labels[int(pred_idx)]
    return {
        "summary": f"Чаще всего модель путает класс {true_label} с классом {pred_label}: {max_error} раз(а).",
        "top_confusion": {
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "count": max_error,
        },
    }


def _metric_line(metrics: dict[str, Any]) -> str:
    roc_auc = metrics["roc_auc_ovr_macro"]
    roc_text = "n/a" if roc_auc is None else f"{float(roc_auc):.4f}"
    return (
        f"macro_f1={float(metrics['macro_f1']):.4f}, "
        f"balanced_acc={float(metrics['balanced_accuracy']):.4f}, "
        f"roc_auc={roc_text}"
    )


def main() -> None:
    config = load_config(PROJECT_ROOT / "config.yaml")
    repro = initialize_reproducibility(config, PROJECT_ROOT, stage_name="stage8_final_eval", use_tensorflow=True)

    labels = tuple(int(label) for label in config["task"]["labels"])
    training_cfg = {
        "early_stopping_patience": int(config["training"]["early_stopping_patience"]),
        "reduce_lr_patience": int(config["training"]["reduce_lr_patience"]),
        "reduce_lr_factor": float(config["training"]["reduce_lr_factor"]),
    }
    autoencoder_cfg = dict(config["training"]["autoencoder"])
    baseline_ref = float(
        _load_json(PROJECT_ROOT / "reports" / "experiments" / "stage4_baseline_summary.json")["versions"]["improved"]["val_metrics"][
            "macro_f1"
        ]
    )

    data_bundle = _collect_stage8_data(config)
    candidate = _choose_best_pipeline(PROJECT_ROOT)

    stability_results: list[dict[str, Any]] = []
    best_seed = None
    best_seed_rank = None

    TEMP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TEMP_MODEL_PATH.exists():
        TEMP_MODEL_PATH.unlink()

    for seed in (0, 1, 2):
        run_result = _run_selected_pipeline(
            candidate=candidate,
            seed=seed,
            labels=labels,
            autoencoder_cfg=autoencoder_cfg,
            training_cfg=training_cfg,
            data_bundle=data_bundle,
        )
        val_metrics = dict(run_result["metrics"])
        stability_record = {
            "seed": int(seed),
            "val_metrics": val_metrics,
            "history": run_result["history"],
            "above_baseline": bool(float(val_metrics["macro_f1"]) >= baseline_ref),
        }
        if candidate["use_autoencoder"]:
            stability_record["ae_pretraining"] = run_result["ae_pretraining"]
            stability_record["transferred_layers"] = list(run_result["transferred_layers"])
            stability_record["finetune_seed"] = int(run_result["finetune_seed"])
        stability_results.append(stability_record)

        current_rank = _seed_rank(val_metrics, seed=seed)
        if best_seed_rank is None or current_rank > best_seed_rank:
            run_result["model"].save(TEMP_MODEL_PATH)
            best_seed_rank = current_rank
            best_seed = int(seed)

        tf.keras.backend.clear_session()

    if best_seed is None or best_seed_rank is None or not TEMP_MODEL_PATH.exists():
        raise RuntimeError("Не удалось выбрать лучший seed для финальной оценки")

    final_model = tf.keras.models.load_model(TEMP_MODEL_PATH)
    test_metrics = evaluate_hybrid_classifier(
        model=final_model,
        x_eval=data_bundle["x_test"],
        y_eval=data_bundle["y_test"],
        labels=labels,
    )
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(FINAL_MODEL_PATH)
    tf.keras.backend.clear_session()
    TEMP_MODEL_PATH.unlink(missing_ok=True)

    selected_stability = next(item for item in stability_results if int(item["seed"]) == best_seed)
    error_analysis = _build_error_analysis(test_metrics)
    present_test_labels = sorted(int(label) for label in np.unique(data_bundle["y_test"]))
    missing_test_labels = [label for label in labels if label not in present_test_labels]

    report: dict[str, Any] = {
        "selected_pipeline": {
            "pipeline_name": candidate["pipeline_name"],
            "dataset_version": candidate["dataset_version"],
            "use_autoencoder": bool(candidate["use_autoencoder"]),
            "hybrid_config": dict(candidate["hybrid_config"]),
            "selection_basis": "максимум macro-F1 на val; tie-break: balanced_accuracy",
            "reference_val_metrics": dict(candidate["reference_val_metrics"]),
            "reference_seed": int(candidate["reference_seed"]),
            "final_seed": int(best_seed),
        },
        "window_counts": data_bundle["window_counts"],
        "stability_check": {
            "baseline_improved_val_macro_f1": baseline_ref,
            "seeds_checked": [0, 1, 2],
            "results": stability_results,
            "stable_vs_baseline": bool(all(item["above_baseline"] for item in stability_results)),
            "selected_seed_metrics": dict(selected_stability["val_metrics"]),
        },
        "final_test_eval": {
            "seed": int(best_seed),
            "test_metrics": test_metrics,
            "model_path": str(FINAL_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "present_test_labels": present_test_labels,
            "missing_test_labels": missing_test_labels,
            "error_analysis": error_analysis,
        },
    }

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)

    md_lines = [
        "# Этап 8: финальная оценка на test",
        "",
        "- `test` в этом запуске используется один раз.",
        f"- Выбранный пайплайн: `{report['selected_pipeline']['pipeline_name']}`",
        f"- Версия данных: `{report['selected_pipeline']['dataset_version']}`",
        f"- AE-предобучение: {'да' if report['selected_pipeline']['use_autoencoder'] else 'нет'}",
        f"- Зафиксированный финальный seed: {report['selected_pipeline']['final_seed']}",
        "",
        "## Основание выбора",
        f"- Reference val: {_metric_line(report['selected_pipeline']['reference_val_metrics'])}",
        f"- Правило выбора: {report['selected_pipeline']['selection_basis']}",
        "",
        "## Проверка стабильности на val",
        f"- Baseline improved val macro-F1: {report['stability_check']['baseline_improved_val_macro_f1']:.4f}",
        f"- Конфигурация стабильна относительно baseline: {'да' if report['stability_check']['stable_vs_baseline'] else 'нет'}",
    ]
    for item in report["stability_check"]["results"]:
        md_lines.append(
            f"- Seed {item['seed']}: {_metric_line(item['val_metrics'])}; "
            f"{'не ниже baseline' if item['above_baseline'] else 'ниже baseline'}"
        )

    md_lines.extend(
        [
            "",
            "## Финальная оценка на test",
            f"- Метрики test: {_metric_line(report['final_test_eval']['test_metrics'])}",
            f"- Сохранённая модель: `{report['final_test_eval']['model_path']}`",
            f"- Классы, присутствующие в test: {report['final_test_eval']['present_test_labels']}",
            f"- Классы, отсутствующие в test: {report['final_test_eval']['missing_test_labels']}",
            f"- Анализ ошибок: {report['final_test_eval']['error_analysis']['summary']}",
            "",
            "## Classification Report (test)",
            "```text",
            str(report["final_test_eval"]["test_metrics"]["classification_report_text"]).rstrip(),
            "```",
            "",
            "## Confusion Matrix (test)",
            f"`{report['final_test_eval']['test_metrics']['confusion_matrix']}`",
        ]
    )
    if "roc_auc_note" in report["final_test_eval"]["test_metrics"]:
        md_lines.append(f"- Примечание по ROC AUC: {report['final_test_eval']['test_metrics']['roc_auc_note']}")

    with SUMMARY_MD_PATH.open("w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(md_lines) + "\n")

    print(f"Сохранено: {SUMMARY_JSON_PATH}")
    print(f"Сохранено: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()

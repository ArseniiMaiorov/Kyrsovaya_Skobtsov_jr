# Постфактум-диагностика validation-моделей

Диагностика выполняется без повторного обучения и без использования `test`.

## stage6_ga_best
- model_path: `output/models/stage6_ga_best.keras`
- macro_f1: 0.6667
- balanced_accuracy: 1.0000
- roc_auc_ovr_macro: 1.0000
- roc_curve: `reports/figures/stage6_ga_best_val_roc.png`
- Примечание: ROC AUC рассчитан по присутствующим классам (0, 2); отсутствующие классы исключены из усреднения

## stage7_hybrid_scratch
- model_path: `output/models/stage7_hybrid_scratch.keras`
- macro_f1: 0.6444
- balanced_accuracy: 0.9375
- roc_auc_ovr_macro: 0.9375
- roc_curve: `reports/figures/stage7_hybrid_scratch_val_roc.png`
- Примечание: ROC AUC рассчитан по присутствующим классам (0, 2); отсутствующие классы исключены из усреднения

## stage7_hybrid_ae_finetuned
- model_path: `output/models/stage7_hybrid_ae_finetuned.keras`
- macro_f1: 0.6444
- balanced_accuracy: 0.9375
- roc_auc_ovr_macro: 0.9375
- roc_curve: `reports/figures/stage7_hybrid_ae_finetuned_val_roc.png`
- Примечание: ROC AUC рассчитан по присутствующим классам (0, 2); отсутствующие классы исключены из усреднения

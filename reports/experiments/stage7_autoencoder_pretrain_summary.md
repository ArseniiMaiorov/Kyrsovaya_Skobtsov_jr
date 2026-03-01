# Этап 7: AE-предобучение и fine-tuning

- Предобучение выполняется на неразмеченных окнах `improved`.
- Статистики `improved` обучаются на размеченном `train` и затем применяются к unlabeled.
- Сравнение выполняется на одном и том же официальном `val`: `с нуля` vs `после AE`.
- Источник encoder-конфига: `stage6_best_genome`

## Окна
- Train (labeled): 55
- Val (labeled): 9
- Unlabeled (AE): 473

## AE-предобучение
- Seed: 10042
- Train windows: 403
- Val windows: 70
- mean_reconstruction_mse: 141.109283
- Лучшая эпоха: 6
- Эпох выполнено: 11
- Сохранённая модель: `output/models/stage7_autoencoder.keras`

## Классификатор с нуля
- Seed: 42
- Метрики val: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=n/a
- Лучшая эпоха: 28
- Сохранённая модель: `output/models/stage7_hybrid_scratch.keras`

## Классификатор после AE
- Seed: 20042
- Метрики val: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=n/a
- Лучшая эпоха: 19
- Перенесённые слои: conv1d_1, batch_norm_1, gru_1
- Сохранённая модель: `output/models/stage7_hybrid_ae_finetuned.keras`

## Сравнение
- Прирост macro-F1 после AE: 0.0000
- Прирост balanced_accuracy после AE: 0.0000
- Лучшая версия по macro-F1: scratch_or_equal
- Примечание по ROC AUC (с нуля): ROC AUC не вычислен: получено значение NaN
- Примечание по ROC AUC (после AE): ROC AUC не вычислен: получено значение NaN

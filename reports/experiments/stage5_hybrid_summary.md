# Этап 5: гибридные нейросетевые модели

Основной протокол: сравнение `raw` и `improved` только на `val` без использования `test`.
Дополнительно выполнены: ROC AUC-анализ, визуализация обучения, аугментация, сравнение типов RNN и сравнение с attention.

## Базовый конфиг гибридной модели
- rnn_type: gru
- use_attention: False
- n_conv_layers: 1
- conv_filters: 64
- conv_kernel_size: 5
- n_gru_layers: 1
- gru_units: 128
- n_dense_layers: 1
- dense_units: 128
- activation: relu
- optimizer: adam
- loss: sparse_categorical_crossentropy
- batch_size: 8
- max_epochs: 30

## Версия данных: raw
- Train: [55, 128, 49]
- Val: [9, 128, 49]
- Метрики val: macro_f1=0.3111, balanced_acc=0.4375, roc_auc=1.0000
- Лучшая эпоха: 9
- Эпох выполнено: 14
- Сохранённая модель: `output/models/stage5_hybrid_raw.keras`
- ROC-кривые: `reports/figures/stage5_raw_roc.png`
- Важно: в `val` отсутствуют классы: 1
- Примечание по ROC AUC: ROC AUC рассчитан по присутствующим классам (0, 2); отсутствующие классы исключены из усреднения
- Артефакт history_json: `output/artifacts/stage5_raw/training_history.json`
- Артефакт training_curves: `output/artifacts/stage5_raw/training_curves.png`

## Версия данных: improved
- Train: [55, 128, 49]
- Val: [9, 128, 49]
- Метрики val: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=1.0000
- Лучшая эпоха: 10
- Эпох выполнено: 15
- Сохранённая модель: `output/models/stage5_hybrid_improved.keras`
- ROC-кривые: `reports/figures/stage5_improved_roc.png`
- Важно: в `val` отсутствуют классы: 1
- Примечание по ROC AUC: ROC AUC рассчитан по присутствующим классам (0, 2); отсутствующие классы исключены из усреднения
- Артефакт history_json: `output/artifacts/stage5_improved/training_history.json`
- Артефакт training_curves: `output/artifacts/stage5_improved/training_curves.png`
- Артефакт conv_filters: `reports/figures/stage5_improved_conv_filters.png`
- Артефакт hidden_representations: `reports/figures/stage5_improved_hidden_repr.png`

## Лучшая версия данных по val macro_f1: improved
- Вывод по raw vs improved: `improved` лучше `raw` по macro-F1 на 0.3333; winsorize + RobustScaler стабилизировали признаки.

## Сравнение типов рекуррентного слоя (improved)
- gru: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=1.0000, params=107395
- lstm: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=1.0000, params=131715
- bi_gru: macro_f1=0.6667, balanced_acc=1.0000, roc_auc=1.0000, params=198275
- bi_lstm: macro_f1=0.6667, balanced_acc=1.0000, roc_auc=1.0000, params=246915
- Лучший тип RNN: bi_gru

## Аугментация временных окон
- Train окон до аугментации: 55
- Train окон после аугментации: 165
- Без аугментации: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=1.0000
- С аугментацией: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=0.9375

## Сравнение без attention и с attention
- Без attention: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=1.0000
- С attention: macro_f1=0.3137, balanced_acc=0.5000, roc_auc=0.0000
- Артефакт attention_weights: `reports/figures/stage5_attention_weights.png`

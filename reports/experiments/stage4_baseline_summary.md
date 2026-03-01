# Этап 4: baseline-модель

Baseline по ТЗ: `LogisticRegression` на flattened временных окнах.
Сравнение `raw` vs `improved` выполняется только на `val`.
`test` на этом этапе не используется.

## Версия данных: raw
- Train(flat): [55, 6272]
- Val(flat): [9, 6272]
- Метрики val: macro_f1=0.1818, balanced_acc=0.1875, roc_auc=0.5625
- ROC-кривые: `reports/figures/stage4_baseline_raw_roc.png`
- Важно: в `val` отсутствуют классы: 1
- Примечание по ROC AUC: ROC AUC рассчитан по присутствующим классам (0, 2); отсутствующие классы исключены из усреднения
- Classification report (val):
```text
              precision    recall  f1-score   support

           0       1.00      0.38      0.55         8
           1       0.00      0.00      0.00         0
           2       0.00      0.00      0.00         1

    accuracy                           0.33         9
   macro avg       0.33      0.12      0.18         9
weighted avg       0.89      0.33      0.48         9
```
- Confusion matrix (val):
```text
[[3, 2, 3], [0, 0, 0], [0, 1, 0]]
```
- Краткий анализ ошибок:
- Класс 0: верно 3/8, чаще всего путается с классом 2 (3 раз).
- Класс 1: отсутствует в текущем split.
- Класс 2: верно 0/1, чаще всего путается с классом 1 (1 раз).

## Версия данных: improved
- Train(flat): [55, 6272]
- Val(flat): [9, 6272]
- Метрики val: macro_f1=0.3137, balanced_acc=0.5000, roc_auc=0.1875
- ROC-кривые: `reports/figures/stage4_baseline_improved_roc.png`
- Важно: в `val` отсутствуют классы: 1
- Примечание по ROC AUC: ROC AUC рассчитан по присутствующим классам (0, 2); отсутствующие классы исключены из усреднения
- График важности признаков: `reports/figures/stage4_baseline_improved_feature_importance.png`
- Топ признаков по коэффициентам (сводно):
  - class_0: Ipt2,A[t115] (coef=0.0130)
  - class_1: Ipt2,A[t115] (coef=-0.0067)
  - class_2: Ipt2,A[t115] (coef=-0.0064)
- Classification report (val):
```text
              precision    recall  f1-score   support

           0       0.89      1.00      0.94         8
           1       0.00      0.00      0.00         0
           2       0.00      0.00      0.00         1

    accuracy                           0.89         9
   macro avg       0.30      0.33      0.31         9
weighted avg       0.79      0.89      0.84         9
```
- Confusion matrix (val):
```text
[[8, 0, 0], [0, 0, 0], [1, 0, 0]]
```
- Краткий анализ ошибок:
- Класс 0: ошибок нет, верно 8/8.
- Класс 1: отсутствует в текущем split.
- Класс 2: верно 0/1, чаще всего путается с классом 0 (1 раз).

## Лучшая версия данных по val macro_f1: improved
- Главный ориентир выбора: `macro_f1`.
- `balanced_accuracy` и `roc_auc_ovr_macro` используются как дополнительные контрольные метрики.

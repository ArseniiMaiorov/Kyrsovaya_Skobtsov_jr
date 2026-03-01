# Этап 4: baseline-модель

Baseline по ТЗ: `LogisticRegression` на flattened временных окнах.
Сравнение `raw` vs `improved` выполняется только на `val`.
`test` на этом этапе не используется.

## Версия данных: raw
- Train(flat): [55, 6272]
- Val(flat): [9, 6272]
- Метрики val: macro_f1=0.1818, balanced_acc=0.1875, roc_auc=n/a
- Важно: в `val` отсутствуют классы: 1
- Примечание по ROC AUC: ROC AUC не вычислен: получено значение NaN
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

## Версия данных: improved
- Train(flat): [55, 6272]
- Val(flat): [9, 6272]
- Метрики val: macro_f1=0.4706, balanced_acc=0.5000, roc_auc=n/a
- Важно: в `val` отсутствуют классы: 1
- Примечание по ROC AUC: ROC AUC не вычислен: получено значение NaN
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

## Лучшая версия данных по val macro_f1: improved
- Главный ориентир выбора: `macro_f1`.
- `balanced_accuracy` и `roc_auc_ovr_macro` используются как дополнительные контрольные метрики.

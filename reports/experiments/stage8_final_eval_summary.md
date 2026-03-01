# Этап 8: финальная оценка на test

- `test` в этом запуске используется один раз.
- Выбранный пайплайн: `stage6_ga_no_ae`
- Версия данных: `improved`
- AE-предобучение: нет
- Зафиксированный финальный seed: 1

## Основание выбора
- Reference val: macro_f1=0.6667, balanced_acc=1.0000, roc_auc=n/a
- Правило выбора: максимум macro-F1 на val; tie-break: balanced_accuracy

## Проверка стабильности на val
- Baseline improved val macro-F1: 0.3137
- Конфигурация стабильна относительно baseline: нет
- Seed 0: macro_f1=0.6444, balanced_acc=0.9375, roc_auc=n/a; не ниже baseline
- Seed 1: macro_f1=0.6667, balanced_acc=1.0000, roc_auc=n/a; не ниже baseline
- Seed 2: macro_f1=0.2917, balanced_acc=0.4375, roc_auc=n/a; ниже baseline

## Финальная оценка на test
- Метрики test: macro_f1=0.6667, balanced_acc=1.0000, roc_auc=n/a
- Сохранённая модель: `output/models/stage8_final_selected.keras`
- Классы, присутствующие в test: [0, 1]
- Классы, отсутствующие в test: [2]
- Анализ ошибок: На test не обнаружено межклассовых ошибок по confusion matrix.

## Classification Report (test)
```text
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         6
           1       1.00      1.00      1.00         3
           2       0.00      0.00      0.00         0

    accuracy                           1.00         9
   macro avg       0.67      0.67      0.67         9
weighted avg       1.00      1.00      1.00         9
```

## Confusion Matrix (test)
`[[6, 0, 0], [0, 3, 0], [0, 0, 0]]`
- Примечание по ROC AUC: ROC AUC не вычислен: получено значение NaN

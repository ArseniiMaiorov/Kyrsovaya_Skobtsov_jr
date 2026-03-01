# Спецификация baseline-моделей (Этап 4)

## Цель
Сформировать воспроизводимую нулевую точку качества перед гибридной моделью `1D-CNN -> GRU`.

## Модель baseline
- `logistic_regression`
- Используется `class_weight=balanced`.
- Вход: flattened окна `(batch, T, n_features) -> (batch, T*n_features)`.
- Конфиг модели:
  - `solver="saga"`
  - `max_iter=2000`
  - мультиклассовый режим определяется текущей версией `scikit-learn` автоматически

## Протокол сравнения
- Сравнение проводится отдельно для двух версий данных:
  - `raw`
  - `improved`
- Выбор лучшей baseline-модели выполняется только по `val macro_f1`.
- `test` на этом этапе не используется.

## Метрики
Единый расчет через `src/metrics/metrics.py`:
- `accuracy`
- `balanced_accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1` (основная)
- `weighted_f1`
- `roc_auc_ovr_macro` (если доступны вероятности)
- `confusion_matrix`

## Артефакты этапа
- `reports/experiments/stage4_baseline_summary.json`
- `reports/experiments/stage4_baseline_summary.md`

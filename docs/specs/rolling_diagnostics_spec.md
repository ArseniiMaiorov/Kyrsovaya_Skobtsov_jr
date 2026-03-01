# Спецификация: дополнительная rolling-диагностика

## Назначение
Это дополнительный инженерный контур для проверки устойчивости модели перед этапом GA.
Он не заменяет официальный `train/val/test` по ТЗ и не меняет правило "не трогать test".

## Область применения
- используется только официальный train-сегмент;
- официальный `val` остаётся единственным формальным источником выбора модели по ТЗ;
- `test` в диагностике не используется.

## Схема фолдов
- тип: expanding-window
- число фолдов: `3`
- train-сегмент делится на `4` последовательных временных блока

Фолды:
1. fold 1: train = блок 1, val = блок 2
2. fold 2: train = блоки 1-2, val = блок 3
3. fold 3: train = блоки 1-3, val = блок 4

## Защита от утечки
- окна строятся отдельно внутри `fold-train` и `fold-val`;
- строки не пересекаются между `fold-train` и `fold-val`;
- все статистики preprocessing (`median`, `q01/q99`, `RobustScaler`) обучаются только на `fold-train`.
- если `fold-train` содержит меньше двух классов, такой фолд считается неинформативным и логируется как `FAIL`, но не останавливает общий прогон.

## Данные и модели
- версия данных: `improved`
- модели:
  - `baseline_logistic_regression`
  - `hybrid_cnn_gru_dense`

## Артефакты
- `reports/experiments/rolling_diagnostics_summary.json`
- `reports/experiments/rolling_diagnostics_summary.md`
- `output/artifacts/rolling_diagnostics_plan.json`

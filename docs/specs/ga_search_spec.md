# Спецификация GA-поиска (Этап 6)

## Цель
Подобрать гиперпараметры гибридной модели `1D-CNN -> GRU -> Dense` поверх базовой конфигурации этапа 5.

## Объект оптимизации
- Используется только версия данных `improved`.
- Fitness считается только на официальном `val`.
- `test` на этапе 6 не используется.
- Rolling-диагностика остаётся отдельным инженерным контуром и не участвует в отборе.

## Гены
- `n_conv_layers`
- `conv_filters`
- `conv_kernel_size`
- `n_gru_layers`
- `gru_units`
- `n_dense_layers`
- `dense_units`
- `optimizer`
- `activation`

## Алгоритм
- `population_size = 12`
- `generations = 8`
- `tournament_size = 3`
- `mutation_probability = 0.2`
- `elite_size = 1`
- Кроссовер: `two-point crossover`
- Мутация: независимая по каждому гену

## Протокол обучения
- Для fitness используется `max_epochs_fitness = 10`.
- Для финального дообучения лучшего генома используется `max_epochs_final = 50`.
- Обязательные callbacks:
  - `EarlyStopping(patience=5, restore_best_weights=True)`
  - `ReduceLROnPlateau(patience=3, factor=0.5)`

## Артефакты
- `reports/experiments/stage6_ga_search_summary.json`
- `reports/experiments/stage6_ga_search_summary.md`
- `output/logs/ga_population_log.jsonl`
- `output/artifacts/stage6_best_genome.json`
- `output/models/stage6_ga_best.keras`

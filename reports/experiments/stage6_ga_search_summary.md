# Этап 6: GA-поиск гиперпараметров

- Датасет для поиска: `improved`.
- Модель для поиска: `hybrid_cnn_gru_dense`.
- Fitness считается только на официальном `val`.
- Rolling-диагностика остаётся отдельным вспомогательным контуром и не участвует в отборе.
- Лог популяции: `output/logs/ga_population_log.jsonl`
- Артефакт лучшего генома: `output/artifacts/stage6_best_genome.json`

## Конфиг GA
- population_size: 12
- generations: 8
- max_epochs_fitness: 10
- max_epochs_final: 50
- tournament_size: 3
- mutation_probability: 0.2
- elite_size: 1

## Лучший индивид по fitness
- Поколение: 5
- Индивид: 11
- Статус: OK
- Метрики fitness-val: macro_f1=0.6667, balanced_acc=1.0000, roc_auc=n/a
- Лучшая эпоха fitness: 10
- Число параметров: 15267
- Геном:
  - n_conv_layers: 1
  - conv_filters: 16
  - conv_kernel_size: 5
  - n_gru_layers: 1
  - gru_units: 32
  - n_dense_layers: 2
  - dense_units: 64
  - optimizer: adam
  - activation: relu

## Поколения
- Поколение 1: best_macro_f1=0.6444, mean_macro_f1=0.5154, ok=12, fail=0
- Поколение 2: best_macro_f1=0.6667, mean_macro_f1=0.5945, ok=12, fail=0
- Поколение 3: best_macro_f1=0.6444, mean_macro_f1=0.5962, ok=12, fail=0
- Поколение 4: best_macro_f1=0.6444, mean_macro_f1=0.5939, ok=12, fail=0
- Поколение 5: best_macro_f1=0.6667, mean_macro_f1=0.6222, ok=12, fail=0
- Поколение 6: best_macro_f1=0.6444, mean_macro_f1=0.6074, ok=12, fail=0
- Поколение 7: best_macro_f1=0.6667, mean_macro_f1=0.6389, ok=12, fail=0
- Поколение 8: best_macro_f1=0.6444, mean_macro_f1=0.5683, ok=12, fail=0

## Финальное дообучение лучшего генома
- Seed: 1000041
- Метрики final-val: macro_f1=0.6667, balanced_acc=1.0000, roc_auc=n/a
- Лучшая эпоха: 39
- Эпох выполнено: 44
- Число параметров: 15267
- Сохранённая модель: `output/models/stage6_ga_best.keras`
- Примечание по ROC AUC: ROC AUC не вычислен: получено значение NaN

## Сравнение с предыдущими этапами
- Baseline improved (этап 4): 0.3137
- Прирост относительно baseline: 0.3529
- Цель по ТЗ (+0.03) выполнена.
- Базовый hybrid improved (этап 5): 0.6444
- Прирост относительно базового hybrid: 0.0222

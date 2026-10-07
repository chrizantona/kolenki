# EXP-OF-002: attention против mean+max

**Подготовлен и проверен локально. На Kaggle ещё не запущен: ждёт успешного полного EXP-OF-001.**

Сравним две небольшие study-level головы на одинаковых frozen OrthoFoundation features. Attention учится выбирать значимые tokens для каждого диагноза. Mean+max соединяет средний контекст и самый выраженный сигнал. Такая проверка дешёвая: encoder повторно не запускаем.

- Те же 4407 studies, labels, report-grouped weak split, seed 42 и фиксированные 12 эпох; те же optimizer, learning rate и sampling.
- Отдельные weak-holdout и production heads; 58 Gold исключены из градиентов и выбора checkpoint.
- До обучения проверяем все features, geometry/identity, manifest/source hashes и bytes/SHA исходных head checkpoints.
- Сравниваем macro/per-class AUC, runtime и память. Weak AUC отражает согласие с teachers; Gold58 — небольшая диагностическая выборка.

Seed и настройки одинаковы. Порядок minibatches может различаться: инициализация разных голов расходует RNG по-разному.

[Конфигурация](../../configs/orthofoundation_meanmax.yaml) · [Запуск и проверки](../../scripts/compare_frozen_heads.py) · [Генератор notebook](../../scripts/build_head_comparison_notebook.py) · [Готовый приватный notebook](../../notebooks/orthofoundation/meanmax/compare.ipynb).

Полный encoder feature bank пока отсутствует; качество и GPU runtime этого сравнения не измерены. Подготовленный notebook использует приватные inputs и сам по себе не открывает другу доступ к ним.

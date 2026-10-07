# EXP-OF-002: attention против mean+max на sharded features

**Подготовлено и проверено; не запускалось. Полного feature bank и измеренных AUC ещё нет.** [Notebook](../../notebooks/orthofoundation/sharded_meanmax) намеренно отказывает до разрешения inputs, пока не передан настоящий SHA завершённого attention-bank.

После EXP-OF-001 root отдельно фиксирует SHA полного банка из завершённого неизменяемого attention output, включая все 4407 studies. Builder получает этот pin явно; он не рассчитывает ожидаемый SHA из данных самого сравнения. Проверяются все семь cache pins, выбранные series, K4 positions, полный mask, finite values и исходные checkpoints/splits. Затем оба attention-head воспроизводят исходные weak-holdout/Gold probabilities с объявленным FP32 допуском. Это связывает сохранённые baseline metrics с теми же features. Исходный run экспортирует probabilities, не raw logits; replay logits проверяются отдельно.

Только `head_kind` и имя эксперимента меняются. Encoder не вызывается: mean+max обучается на том же банке, с тем же weak-fold0, seed42, AdamW и двумя фиксированными 12-epoch fits. Gold58 исключён из градиентов и выбора checkpoint. Одинаковый seed не гарантирует одинаковый порядок minibatches: разные heads расходуют RNG по-разному при initialization.

Private/offline CPU notebook подключает только маленький global metadata Dataset и завершённый feature/head output. Полный Python-stage cap 1500 s: подготовка до 300 s, unchanged head fit до 1200 s. CUDA — явно выбранный отдельный режим; actual CUDA replay ещё не проверен. Сохраняются per-label/macro AUC differences, checkpoint/CSV/bank SHA, replay device/tolerance/runtime; predictions и UID tables остаются приватными.

Independent review: 44 tests PASS за 11.451 s; root повторил 7 новых contracts за 2.362 s. Ревью воспроизвело старую ошибку проверки и подтвердило отказ до fit при изменении scored Gold и unscored training features, CSV, checkpoint с обновлёнными internal receipts и неправильном/отсутствующем pin. Эти fixtures не подтверждают качество модели или время на полном банке. [Verification receipt](verification_receipt.json).

Код: [adapter](../../scripts/compare_sharded_heads.py), [builder](../../scripts/build_sharded_head_comparison_notebook.py), [tests](../../tests/test_sharded_head_comparison.py). Основной MRI pilot, core7, frozen config и sharded orchestrator не изменены.

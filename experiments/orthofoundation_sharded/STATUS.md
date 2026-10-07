# OrthoFoundation: текущий запуск по одному архиву

**MRI pilot PASS; проверены 7/7 extraction-частей: 4 413 study partials / 85 336срезов.** Запуск сборки и обучения attention принят Kaggle; веса/AUC ещё не проверены. Полный банк, production training и AUC пока не готовы. Текущие actual receipts, Kaggle-ссылки и прогноз — в [HTTP V2 журнале](../orthofoundation_http_sharded_v2/STATUS.md). [Машинный статус](status.json).

На Kaggle pilot действительно обработал 16 weak studies / 308 MRI-срезов и сделал одно обновление пробной головы; Gold в градиентах — 0. Encoder: 72.6 images/s, peak1.33GB, owned runtime217.8s, cleanup подтверждён. [Независимо проверенный pilot](../orthofoundation_http_cache0_pilot_v2/root_MRI_validation.json). Smoke-обновление проверяет работоспособность, а не заменяет полное обучение.

## Как исполняем

1. Extract0…6: каждый job скачивает один проверяемый MRI-архив по временному READ-доступу, пересчитывает SHA и сохраняет только принадлежащие ему признаки. Источник0 — приватный byte-identical mirror, остальные — исходные kernel outputs; source5 получен отдельным декодированием DICOM, для него нет заявления о равенстве прежних NPY.
2. ROOT и отдельный reviewer проверяют каждый фактический NPZ: SHA, shape, dtype, конечность, global selection, positions и ownership. Прежде чем перейти к следующему job, учитываем свежую GPU-квоту и резерв на оставшуюся работу.
3. Merge собирает 4 407 studies, включая шесть studies с сериями в разных архивах. Проверяет полное покрытие и отсутствие повторного владения slots, затем обучает две attention-головы по 12 фиксированных эпох: weak-fold и production.
4. Отдельный [V2 mean/max adapter](../orthofoundation_sharded_meanmax_http_v2/STATUS.md) сравнивает головы на том же завершённом банке. Для запуска нужны внешняя проверка полного банка, его SHA pin и replay обоих attention-checkpoints. Encoder повторно не запускается.

Глобальный выбор: 21 334 slot-series, 85 336 изображений при K=4. Encoder frozen, `author_no_rope`, FP16, вход224×224, grayscale slice повторяется в RGB, FOV0.92. Это одиночные срезы, а не соседние срезы в трёх каналах. Gold58 исключён из градиентов и выбора checkpoint. [Модель и обучение подробнее](../../docs/ORTHO_CURRENT.md).

Owned Python caps для V2: pilot900s; extract1204/1231/1236/1213/1216/1228/1069s, вместе9297s при guard9300s. Merge/head job отдельно1500s, fit максимум1200s. Pre-Python Kaggle startup этими Python-таймерами не ограничен. Текущая оценка использует минимум реально измеренной скорости, опубликованный в [owner measurements](../orthofoundation_http_sharded_v2/owner_measurements.json); это прогноз, не гарантированное время.

## История transport-попыток

- Подключение семи MRI-cache inputs зависало до Python; большой probe отменён. [Cancellation receipt](cache_probe_cancellation_receipt.json).
- Sharded notebook-input pilot137491819/V1 и Dataset pilot137498512/V1 завершились ERROR без подтверждённого MRI/head обучения. [Первый failure receipt](pilot_v1_failure_receipt.json), [Dataset receipt](dataset_pilot_10min_checkpoint.json).
- HTTP pilot137503798/V1 скачал полный10.242GB архив с совпавшим SHA за581.26s, но cap600s остановил повторный SHA до encoder. [Terminal receipt](../orthofoundation_http_cache0_pilot/terminal_receipt.json).
- Отдельная V2 с pilot900s прошла реальные MRI и smoke-head проверки. После code review началось полное последовательное извлечение; старые V1 scientific modules не менялись.

[HTTP V2 notebooks](../../notebooks/orthofoundation/http_sharded_v2), [source-index pins](../orthofoundation_http_sharded_v2/source_index.json), [независимое ревью полного кода](../orthofoundation_http_sharded_v2/independent_review.json). В Git публикуем код и агрегатные receipts; приватные MRI, clinical rows, predictions, weights и scoped READ URLs туда не входят.

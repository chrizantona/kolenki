# OrthoFoundation frozen features → 12-label head

**MRI pilot V2 PASS; полное извлечение идёт по одному архиву.** Проверены 5/7 частей: 3 208 study partials / 62 044изображений; Shard5 запущен. Production training и Ortho AUC пока не готовы. Актуальные запуски и actual receipts: [HTTP V2 журнал](../orthofoundation_http_sharded_v2/STATUS.md), [машинный статус](status.json).

Взяли опубликованный OrthoFoundation-L на основе DINOv3 ViT-L/16. Encoder заморожен, выдаёт 1024-мерный CLS-вектор с каждого среза. Checkpoint1 213 056 638bytes, SHA256 `385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`. [Asset provenance](asset_manifest.json). Первый эксперимент сохраняет авторский `author_no_rope`; canonical RoPE — отдельная будущая абляция.

Pilot на T4 реально обработал16weak studies/308MRI-срезов, сохранил конечные признаки и сделал одно обновление smoke-head, Gold gradient0. Encoder72.6images/s, peak1.33GB. Это проверка работоспособности; качество новой ветки определяется после полного fixed-epoch обучения. [Фактическая проверка](../orthofoundation_http_cache0_pilot_v2/root_MRI_validation.json).

Полный план: frozen features всех4349weak+58Gold, K4 в каждой из6plane/FSslots; объединение семи partial banks; attention weak-fold head и production head по12эпох, затем отдельное mean/max сравнение. Gold58 не участвует в градиентах или выборе checkpoint. [Модель и данные подробно](../../docs/ORTHO_CURRENT.md), [сравнение с ConvNeXt](../orthofoundation_http_sharded_v2/COMPARISON_PLAN.md), [V2 mean/max adapter](../orthofoundation_sharded_meanmax_http_v2/STATUS.md).

Исходные семь модулей [src/orthofoundation](../../src/orthofoundation) сохраняются. Текущий запуск использует отдельный [V2 orchestrator](../../scripts/orthofoundation_http_v2/sharded_frozen_extract.py) и [V2 config](../../configs/orthofoundation_frozen_http_v2.yaml); их source pins записаны в receipts. Это versioned execution route с собственной feature identity.

Ранние notebook-input попытки137481721/V1–V2 завершилисьERROR без пользовательского Python и MRI-head результатов. [V1 failure](pilot_v1_failure_summary.json), [V2 failure](pilot_v2_failure_summary.json). Успешный [model-assets probe](asset_gpu_smoke_report.json) отдельно доказал strict checkpoint load и synthetic forward на T4, а последующий MRI pilot — работу на реальных срезах. История остальных transport-попыток: [sharded journal](../orthofoundation_sharded/STATUS.md).

Kaggle inputs приватные: ссылка на notebook сама по себе не даёт другу доступ. В Git есть код, описание и агрегатные проверяемые результаты; нет MRI, clinical text/UID rows, prediction CSV, весов и scoped READ URLs. Нового public score Ortho пока нет.

# OrthoFoundation HTTP pilot V2

**Подготовлен и прошёл независимое ревью; фактическое MRI/head выполнение ещё не подтверждено.** [Notebook](../../notebooks/orthofoundation/http_cache0_pilot_v2/train.ipynb) разрешён только для одного pilot после проверок private READ input, wire930, fresh quota ≥11427 s и отсутствия active jobs. [Review receipt](review_receipt.json).

[V1](../orthofoundation_http_cache0_pilot/terminal_receipt.json) скачал полный 10.24 GB архив с правильным SHA за 581.26 s, но лимит600 остановил его до MRI. V2 даёт shared Python deadline900 s и server request930 s для завершения/cleanup; earliest timer включает download, повторный SHA scan, preprocessing, encoder, head smoke и export. Server wire пока описан как request-only в [launch settings](launch_settings.json); фактический accepted source/wire фиксируется отдельным receipt.

Core7 и V1 orchestrator не менялись. [Versioned runner](../../scripts/orthofoundation_http_v2/sharded_frozen_extract.py) содержит ровно четыре замены числовых pins: pilot600→900, total9000→9300, строгий config guard9300 и default projection9300. В [V2 config](../../configs/orthofoundation_frozen_http_v2.yaml) меняются только имя experiment и execution budget. Model checkpoint, 224px, grayscale RGB, K4/FOV0.92, FP16, longest-series policy, split/seed/optimizer и две12-epoch головы сохранены.

Новые pipeline provenance/feature identity отличаются вследствие SHA versioned orchestrator: `7f0e16dc1c34e8c0920f567db2ebb68f55c470c09ce77dd7c0aa238cd5ec6c77`. Она независимо вычислена на всех проверенных global metadata до запуска; это ещё не доказательство полученных GPU features. Семь caps прежние:1204/1231/1236/1213/1216/1228/1069 s; сумма8397 +pilot900 =9297 ≤9300 s. Merge/head reserve1500 s отдельно, head fit1200 s.

Независимые23 localhost/process fixtures PASS9.457 s: строгие job/config versions, точные4diffs, старые pins, общие timeout/cleanup и неизменность V1/core7. Их результат не доказывает качество MRI модели. Full7 extraction/merge notebooks готовятся отдельно и запускаются только после настоящих finite features, optimizer update и подходящего прогноза времени. Текущий V1 mean+max adapter несовместим с V2 config9300/new identity; для него требуется отдельная V2 адаптация и review.

URL/READ job, weights, MRI, UID и медицинские таблицы остаются private. Никакого нового submission или измеренного AUC V2 пока нет.

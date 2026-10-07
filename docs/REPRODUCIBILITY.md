# Воспроизводимость и доступ

Этот репозиторий — handoff для участника команды: описание, код, notebooks/configs и небольшие проверяемые receipts. Исходные DICOM, cache archives, checkpoints, отчёты, UID-таблицы и credentials не должны попадать в Git. Данные и weights подключаются как Kaggle inputs или скачиваются из лицензированных источников вне репозитория.

## Подтверждённые запуски

| Run | Kaggle reference | Результат |
|---|---|---|
| Baseline | `alanchoo/rsna-knee-public-944-v2`, run `355816679`, submission `56886017` | Scoring COMPLETE, public 0.943 |
| ConvNeXt training | `alanchoo/rsna-knee-cnxt-finetune`, version 1 | Две полные эпохи, 1086 успешных updates |
| Candidate inference / submit | `alanchoo/rsna-knee-944-finetuned`, run `355883025`, submission `56891310` | Scoring COMPLETE, public **0.944** |

[GoodPJ source version](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944?scriptVersionId=355300588) закреплена как исходный public baseline. [Наш candidate](https://www.kaggle.com/code/alanchoo/rsna-knee-944-finetuned?scriptVersionId=355883025) сейчас приватный; другу понадобится отдельный доступ к notebooks/inputs либо самостоятельный запуск из опубликованного здесь кода.

Runtime обучения: Python 3.12.13, PyTorch 2.10.0+cu128, NumPy 2.0.2, pandas 2.3.3; Kaggle T4, fp16. Локальная версия PyTorch отличается и не является точным training environment. GPU quota изменяется; текущий остаток проверяется перед новым запуском.

Опубликованные исходники и доказательства:

- [knee.py](../src/convnext_reader/knee.py): model, study dataset и sampling.
- [preprocess.py](../src/convnext_reader/preprocess.py): DICOM → physical crop/cache.
- [train.py](../src/convnext_reader/train.py): наш resumable warm-start training port, fp16, accumulation и EMA.
- [zip_cache_loader.py](../src/convnext_reader/zip_cache_loader.py): чтение sharded caches.
- [infer.py](../src/convnext_reader/infer.py): reader inference; полный public ensemble — в закреплённом GoodPJ notebook.
- [make_labels_reference.py](../src/convnext_reader/make_labels_reference.py): исходный labels recipe с путями автора `/mnt/...`; перед запуском адаптировать пути к своим inputs.
- [training_report.json](../experiments/convnext_fullweak_2ep/training_report.json), [results.json](../experiments/convnext_fullweak_2ep/results.json), [submission_receipt.json](../experiments/convnext_fullweak_2ep/submission_receipt.json): небольшие receipts реальных запусков.
- [Third-party attribution](../third_party/NOTICE.md): источник reader-кода и применимые лицензии. GoodPJ reader имеет Apache-2.0; MIT корневого репозитория не меняет её.

Это ещё не запуск «одной командой»: private Kaggle inputs, cache paths и labels reference нужно подключить в своей среде. Source manifest фиксирует происхождение исходных файлов; опубликованный training port также содержит наши изменения для Kaggle.

## Что сохранять для каждого нового эксперимента

1. `config` и git commit: encoder/head, image size, sampling, splits, seed, labels version, optimizer, epochs.
2. Provenance inputs: официальный dataset/version, cache manifests, URL/version/hash внешних weights.
3. Training report: реально обработанные studies, completed epochs, successful/skipped updates, loss, runtime, peak memory.
4. Checkpoint hash и predictions на фиксированной validation; macro/per-class AUC и ограниченная Gold-диагностика.
5. Kaggle kernel version, status и submission receipt. `RUNNING`/`COMPLETE` kernel не равны успешному hidden scoring; score записываем только после подтверждения.

Для OrthoFoundation актуальный журнал — [STATUS.md](../experiments/orthofoundation_frozen/STATUS.md). Отсутствующий score означает, что результат ещё не измерен, а не нулевой AUC.

## Как повторить текущий подход

1. Присоединиться к соревнованию и получить доступ к официальным train inputs и публичным weak labels.
2. Взять закреплённую исходную версию GoodPJ и проверить доступность всех её model dependencies.
3. Подготовить cache с совместимыми геометрией и sampling. Наш полный cache: 4407 studies / 24 371 series / 819 078 source slices; семь shards, все manifests проверены перед training.
4. Дообучить публичный fold0 по настройкам из [CURRENT_SOLUTION.md](CURRENT_SOLUTION.md), исключив 58 Gold. Не смешивать частичный pilot с финальным checkpoint.
5. Подставить собственный EMA вместо fold0 в reader; fold1/fold2 и остальные public branches сохранить. Проверить strict state-dict load, finite logits, masks и offline input paths.
6. Проверить schema/order `submission.csv` на visible demo, затем отправить notebook и дождаться полного hidden rerun.

Visible demo содержит 3 исследования и служит проверкой исполнения. Его AUC и совпадение финальных рангов не говорят о качестве на скрытом test.

## Пределы текущей проверки

- Итоговый ConvNeXt EMA — `123057035` bytes; SHA256 `2557782563a0b416cf85c7bf98a0caaf685dc4f991409b184b8b118730dfa25a`.
- Независимый аудит от 6 октября подтвердил реальное обучение, изменение всех model tensors и запуск собственного checkpoint. На момент того аудита hidden score ещё был неизвестен.
- Более поздний submission receipt от **7 октября 08:46 UTC** подтвердил `COMPLETE` и public **0.944**. При конфликте старого snapshot и позднего receipt учитываем время и область проверки.
- Gold macro AUC 0.911160 — диагностика одной reader-модели на 58 исследованиях, не чистый OOF и не метрика финального ансамбля.
- Нет подтверждённого private score, сравнения нескольких seeds или доказанного постоянного выигрыша +0.001.
- Лицензия репозитория не переоформляет сторонние code/data/weights; сохраняем авторство и проверяем их условия отдельно.

Для диагностики Kaggle bootstrap первый code cell сохраняет `bootstrap_start.json` до ML-импортов. Последующие этапы пишутся в `bootstrap_progress.json`; Python-ошибка сохраняет фазу и traceback в `bootstrap_failure.json`, после чего запуск завершается ошибкой. Отсутствие этих файлов ограничивает диагностику: оно само по себе не раскрывает причину сбоя инфраструктуры.

## OrthoFoundation по одному архиву

[Sharded journal](../experiments/orthofoundation_sharded/STATUS.md) описывает подготовленные pilot, семь extract jobs и merge/head job. Builder читает приватные global metadata и asset manifest, проверяет pins и встраивает только код. Используются те же семь основных модулей, но отдельный orchestrator и отдельная feature identity. Для повторения потребуются собственный доступ к двум приватным datasets и MRI-cache outputs; одних публичных notebooks в Git недостаточно.

Не меняйте выбор series отдельно в каждом архиве: шесть studies пересекают границы архивов. Merge принимает только проверенное полное глобальное покрытие. Caps ограничивают Python-stage; задержка Kaggle до первого cell ими не ограничена. Existing mean/max comparator требует adapter для нового schema.

Альтернативный transport использует `--cache-transport datasets --cache-dataset-map <private-map.json>`. Map содержит shard → dataset ID и путь к локальному mirror manifest. Архив хранится как `.zip.bin`, полностью проверяется по прежнему SHA и подключается к неизменённому loader через временный symlink под `/tmp`; внешний cleanup guard очищает его при ошибках до и после child launch. Это обход подключения inputs; модель, labels, sampling и расписание обучения сохраняются. Подготовка artifact не означает, что удалённый Dataset готов или MRI pilot прошёл.

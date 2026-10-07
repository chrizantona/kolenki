# OrthoFoundation: извлечение признаков по одному архиву

**HTTP transport проверен на Kaggle: скачаны все 10 241 860 213 bytes, полный SHA совпал за 581.26 s. Сам MRI pilot V1 не прошёл: лимит 600 s остановил его при повторном SHA scan, до подтверждённых features/head updates.** [Terminal receipt](../orthofoundation_http_cache0_pilot/terminal_receipt.json). Предыдущий Dataset pilot тоже завершился `ERROR`, оба задания terminal, reserved quota = 0. Готовим отдельный versioned pilot с Python900 / server930 s; V1 core7 и orchestrator остаются неизменными. Новый общий frozen budget9300 s позволит сохранить прежние семь extraction caps; полное извлечение всё ещё зависит от настоящего MRI/head smoke и свежей квоты.

**Dataset pilot [137498512 / V1](https://www.kaggle.com/code/alanchoo/rsna-knee-ortho-dataset-pilot) завершился `ERROR`, без подтверждённого Python.** Историческая [проверка через десять минут](dataset_pilot_10min_checkpoint.json) зафиксировала ожидание и временное GPU allocation; после terminal состояние quota было пересчитано. Runtime600 был передан через `sessionTimeoutSeconds`, но ожидание до Python им не ограничивалось.

**Первый pilot [137491819 / V1](https://www.kaggle.com/code/alanchoo/rsna-knee-ortho-sharded-pilot) завершился `ERROR` до первого Python cell. GPU-квота не списалась; фактического MRI/head обучения нет.** Это отдельный способ исполнения того же первого эксперимента, выбранный после зависания Kaggle при подключении всех семи MRI-архивов.

## Что уже проверено

- OrthoFoundation-L действительно загружен на Tesla T4: strict checkpoint load, конечный 1024-мерный вектор и замороженные веса. Это был один синтетический image, не MRI-эксперимент.
- Отдельный CPU notebook прочитал официальные таблицы: 4407 studies и 24 371 series; SHA256 совпали с локальными исходниками. Это проверяет таблицы, а не работу MRI-cache на GPU.
- [Metadata+assets-only GPU probe](https://www.kaggle.com/code/alanchoo/rsna-knee-ortho-metadata-assets-probe) завершился `COMPLETE`: Tesla T4, конечная CUDA-операция, все 49 metadata members и семь fingerprint pins совпали. MRI он не декодировал; encoder load отдельно доказан model-assets probe. [Receipt](metadata_assets_gpu_probe_report.json).
- Global metadata подготовлены в отдельном **приватном** Kaggle dataset: архив 6 272 628 bytes, без MRI и весов. Его удалённая копия проверена по SHA256. Медицинские отчёты и UID-таблицы остаются вне Git.
- Выбор серии считается по всему датасету до разделения jobs: 21 334 серии в шести plane/FS slots, 85 336 изображений при K=4. У шести studies выбранные серии находятся в разных архивах; merge явно собирает их вместе.
- CPU contract tests проверяют глобальный выбор, ownership, неполное/перекрывающееся покрытие, fingerprint tampering и исключение Gold из градиентов. Эти тесты не заменяют фактический GPU pilot.

## Как исполняется

1. **Pilot:** один MRI-архив + assets OrthoFoundation + маленькие global metadata. Проверяем 16 weak studies, конечные признаки, настоящее обновление пробной головы, скорость и память.
2. **Extract 0…6:** каждый job читает один MRI-архив и сохраняет только принадлежащие ему признаки. Для shard 0 предусмотрено использование уже посчитанных pilot features.
3. **Merge + heads:** подключаются семь небольших outputs с features. Перед обучением проверяются полное покрытие 4407 studies, ownership, выбранные series, positions, SHA256 и отсутствие повторов. Затем неизменённый training code обучает weak-fold head и production head по 12 фиксированных эпох.

Encoder остаётся frozen; вход — 224×224, один grayscale slice повторён в RGB, FOV 0.92, K=4, `author_no_rope`, FP16. Gold58 не участвует в градиентах или выборе checkpoint. Исходные семь модулей `src/orthofoundation` и базовая конфигурация не менялись; новый orchestrator имеет собственную identity. Не утверждаем побайтовое равенство features при другом GPU batching.

## Ограничения времени и следующие решения

Для неизменённой V1 Python-stage caps: pilot 600 s; семь extraction jobs — 1204/1231/1236/1213/1216/1228/1069 s. Сумма — **8997 s**. Для merge/head job отдельно зарезервировано 1500 s, из них не более 1200 s на обучение. Перед каждым GPU job проверяем живую квоту; продолжение зависит от фактической скорости pilot.

Подключение inputs до первого Python cell не ограничивается этим таймером. Большой cache diagnostic превысил час, не показав пользовательского Python; он отменён через UI, официальный статус `CANCEL_ACKNOWLEDGED`, за ожидание GPU-квота не списалась. Точная причина не раскрыта Kaggle API. `RUNNING` само по себе не означает, что модель учится. Если подключение одного архива тоже не работает, нужен другой проверенный способ чтения, а не повторение полного запуска.

Существующее сравнение attention/mean+max рассчитано на исходный feature schema. Для sharded bank подготовлен и прошёл ревью [отдельный adapter](../orthofoundation_sharded_meanmax/STATUS.md): явный SHA полного завершённого attention-bank и checkpoint replay обязательны. Пока нет bank pin, notebook не исполняется; реальное сравнение ещё не запущено. Нового public score и подтверждённого выигрыша OrthoFoundation нет.

Следующий проверяемый обход — byte-identical mirror одного cache-архива в приватном dataset с расширением `.zip.bin`, аналогично рабочему model asset. Это гипотеза о способе подключения, а не установленная причина сбоя. До успешного MRI pilot остальные шесть архивов не зеркалируем. [Failure receipt](pilot_v1_failure_receipt.json).

Source0 целиком скачан и проверен: **10 241 860 213 bytes**, SHA256 `31b3dd3934e79931fa78195315bef1b58d8432bae841b7920c6d4497262e764c`, все 3511 ZIP members согласованы с прежним manifest, неизменённый loader принимает архив. Приватный Dataset теперь `ready`, V1: точные три файла, storage MD5 всего объекта, первые/последние 64 KiB и SHA полных маленьких файлов совпали. Начальный 403 был временным; HTTP pilot затем пересчитал полный архив на GPU host и подтвердил тот же SHA за 581.26 s. [Remote verification](cache0_dataset_remote_verification.json). Новый [Dataset pilot](../../notebooks/orthofoundation/dataset_sharded/rsna-knee-ortho-dataset-pilot) подготовлен и прошёл code review и 37 тестов. Он подключает три datasets, без MRI notebook inputs; перед encoder потоково считает SHA всего архива. Hashing входит в прежний лимит 600 s. [Static verification](dataset_transport_verification.json).

Первый [verification receipt](verification_receipt.json) фиксирует подготовку исходного notebook-output маршрута на commit `40e05f6`; указанный в нём builder SHA относится к [той версии builder](https://github.com/chrizantona/kolenki/blob/40e05f657ef6bccb126b47d0c2ec9af0edebc1fa/scripts/build_sharded_notebooks.py). Позже builder расширен Dataset-маршрутом; восемь vendored source-файлов и прежние notebooks не менялись.

Код: [orchestrator](../../scripts/sharded_frozen_extract.py), [builder](../../scripts/build_sharded_notebooks.py), [tests](../../tests/test_sharded_fallback.py). Подготовленные notebooks: [sharded](../../notebooks/orthofoundation/sharded). Машинный журнал: [status.json](status.json); проверка подготовки: [verification_receipt.json](verification_receipt.json). Предыдущие попытки и model provenance: [основной журнал OrthoFoundation](../orthofoundation_frozen/STATUS.md).

Dataset pilot [137498512 / V1](https://www.kaggle.com/code/alanchoo/rsna-knee-ortho-dataset-pilot) принят, private/offline T4, cap 600 s. Этот Dataset run завершился ERROR без подтверждённого Python/MRI/head; принятие run не доказывает обучение. Подготовленный [CPU cloud transport](../../scripts/cache_transfer) прошёл отдельное ревью и 17 protocol tests, но production transfer ещё не запускался.

HTTP transport подключает только model assets, небольшие global metadata и отдельный private input с временной file-scoped READ ссылкой на один архив. Scoped URL выпущен и проверен; принятый HTTP pilot 137503798/V1 уже запустил Python. Скачивание, полный SHA и MRI pilot ограничены общим hard timeout 600 s; ключ аккаунта в облако не передаётся. Полный download SHA доказан, но MRI/head не успели до timeout; [terminal receipt](../orthofoundation_http_cache0_pilot/terminal_receipt.json) отделяет эти результаты.

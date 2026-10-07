# OrthoFoundation: извлечение признаков по одному архиву

**Первый pilot [137491819 / V1](https://www.kaggle.com/code/alanchoo/rsna-knee-ortho-sharded-pilot) завершился `ERROR` до первого Python cell. GPU-квота не списалась; фактического MRI/head обучения нет.** Это отдельный способ исполнения того же первого эксперимента, выбранный после зависания Kaggle при подключении всех семи MRI-архивов.

## Что уже проверено

- OrthoFoundation-L действительно загружен на Tesla T4: strict checkpoint load, конечный 1024-мерный вектор и замороженные веса. Это был один синтетический image, не MRI-эксперимент.
- Отдельный CPU notebook прочитал официальные таблицы: 4407 studies и 24 371 series; SHA256 совпали с локальными исходниками. Это проверяет таблицы, а не работу MRI-cache на GPU.
- Global metadata подготовлены в отдельном **приватном** Kaggle dataset: архив 6 272 628 bytes, без MRI и весов. Его удалённая копия проверена по SHA256. Медицинские отчёты и UID-таблицы остаются вне Git.
- Выбор серии считается по всему датасету до разделения jobs: 21 334 серии в шести plane/FS slots, 85 336 изображений при K=4. У шести studies выбранные серии находятся в разных архивах; merge явно собирает их вместе.
- CPU contract tests проверяют глобальный выбор, ownership, неполное/перекрывающееся покрытие, fingerprint tampering и исключение Gold из градиентов. Эти тесты не заменяют фактический GPU pilot.

## Как исполняется

1. **Pilot:** один MRI-архив + assets OrthoFoundation + маленькие global metadata. Проверяем 16 weak studies, конечные признаки, настоящее обновление пробной головы, скорость и память.
2. **Extract 0…6:** каждый job читает один MRI-архив и сохраняет только принадлежащие ему признаки. Для shard 0 предусмотрено использование уже посчитанных pilot features.
3. **Merge + heads:** подключаются семь небольших outputs с features. Перед обучением проверяются полное покрытие 4407 studies, ownership, выбранные series, positions, SHA256 и отсутствие повторов. Затем неизменённый training code обучает weak-fold head и production head по 12 фиксированных эпох.

Encoder остаётся frozen; вход — 224×224, один grayscale slice повторён в RGB, FOV 0.92, K=4, `author_no_rope`, FP16. Gold58 не участвует в градиентах или выборе checkpoint. Исходные семь модулей `src/orthofoundation` и базовая конфигурация не менялись; новый orchestrator имеет собственную identity. Не утверждаем побайтовое равенство features при другом GPU batching.

## Ограничения времени и следующие решения

Python-stage caps: pilot 600 s; семь extraction jobs — 1204/1231/1236/1213/1216/1228/1069 s. Сумма — **8997 s**. Для merge/head job отдельно зарезервировано 1500 s, из них не более 1200 s на обучение. Перед каждым GPU job проверяем живую квоту; продолжение зависит от фактической скорости pilot.

Подключение inputs до первого Python cell не ограничивается этим таймером. Большой cache diagnostic превысил час, не показав пользовательского Python; он отменён через UI, официальный статус `CANCEL_ACKNOWLEDGED`, за ожидание GPU-квота не списалась. Точная причина не раскрыта Kaggle API. `RUNNING` само по себе не означает, что модель учится. Если подключение одного архива тоже не работает, нужен другой проверенный способ чтения, а не повторение полного запуска.

Существующее сравнение attention/mean+max рассчитано на исходный feature schema. Для этого нового sharded bank нужен отдельный проверенный adapter; оно пока не запущено. Нового public score и подтверждённого выигрыша OrthoFoundation нет.

Следующий проверяемый обход — byte-identical mirror одного cache-архива в приватном dataset с расширением `.zip.bin`, аналогично рабочему model asset. Это гипотеза о способе подключения, а не установленная причина сбоя. До успешного MRI pilot остальные шесть архивов не зеркалируем. [Failure receipt](pilot_v1_failure_receipt.json).

Код: [orchestrator](../../scripts/sharded_frozen_extract.py), [builder](../../scripts/build_sharded_notebooks.py), [tests](../../tests/test_sharded_fallback.py). Подготовленные notebooks: [sharded](../../notebooks/orthofoundation/sharded). Машинный журнал: [status.json](status.json); проверка подготовки: [verification_receipt.json](verification_receipt.json). Предыдущие попытки и model provenance: [основной журнал OrthoFoundation](../orthofoundation_frozen/STATUS.md).

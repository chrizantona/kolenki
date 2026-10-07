# OrthoFoundation frozen features → 12-label head

**Этап: настоящий MRI pilot V2 PASS —16 studies/308 images, finite features, одно реальное обновление головы, Gold исключён.** [Verified pilot](../orthofoundation_http_cache0_pilot_v2/STATUS.md). Полное извлечение всех4407 studies и12-epoch головы ещё не запущены; свежий журнал — [HTTP sharded route](../orthofoundation_sharded/STATUS.md).

[Notebook pilot](https://www.kaggle.com/code/alanchoo/rsna-knee-orthofoundation-pilot), kernel `137481721`. В обеих версиях output содержит только log `[]`, traceback и наши run-файлы отсутствуют, GPU-квота не списалась. Точная причина не раскрыта API. [V1](pilot_v1_failure_summary.json) · [V2](pilot_v2_failure_summary.json). Образ и model assets затем проверены отдельным успешным тестом ниже. Отдельный probe с семью MRI-cache inputs и competition input не показал пользовательского Python за час и был отменён; [receipt отмены](../orthofoundation_sharded/cache_probe_cancellation_receipt.json).

Отдельный [model-assets probe](https://www.kaggle.com/code/alanchoo/rsna-knee-ortho-asset-startup-probe) завершился `COMPLETE`: Tesla T4, torch 2.10.0+cu128, строгая загрузка checkpoint за 14.61 s, frozen forward `author_no_rope` дал конечный `[1,1024]`, peak allocated 1.237 GB. Это синтетический один-image smoke test: скорость на MRI, обучение головы и точность ещё не проверены. [GPU receipt](asset_gpu_smoke_report.json).

[Competition-only CPU probe](https://www.kaggle.com/code/alanchoo/rsna-knee-ortho-competition-cpu-probe) также завершился `COMPLETE`: официальные CSV доступны и совпадают по SHA256. Семь cache inputs отдельно проверены через API: текущие версии доступны, небольшие manifests совпадают с прежними pins, начало каждого архива читается через HTTP Range. Причина зависания их подключения в Kaggle notebook пока не установлена.

Взяли опубликованный OrthoFoundation-L, основанный на DINOv3 ViT-L/16. Checkpoint скачан: **1 213 056 638 bytes**, SHA256 `385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`. Строгая загрузка всех 368 ключей и CPU forward прошли; выход — конечный 1024-мерный CLS-вектор. Это проверка совместимости, не проверка точности.

Первый вариант воспроизводит авторский проход **без RoPE**, использованный в опубликованном OrthoFoundation wrapper. Канонический DINOv3 forward с RoPE — отдельная будущая абляция.

План первого запуска:

1. GPU pilot: реальные MRI из проверенного cache, скорость, память, конечные признаки и фактическое обновление тестовой головы.
2. При достаточной скорости — признаки всех 4349 weak и 58 Gold studies: 224×224, одиночный grayscale slice повторяется в RGB, до 4 срезов на каждую из 6 категорий серии.
3. Обучение новой головы с фиксированным числом эпох. Gold58 исключён из градиентов и выбора checkpoint. Отдельная weak-fold диагностика и production-head на всех weak.
4. Оценка по каждому диагнозу; затем отдельное решение о включении в ансамбль. Нового leaderboard score пока нет.

Код: [src/orthofoundation](../../src/orthofoundation). Конфигурация: [orthofoundation_frozen.yaml](../../configs/orthofoundation_frozen.yaml). Фактические статусы и время последней проверки: [status.json](status.json). Provenance модели: [asset_manifest.json](asset_manifest.json).

Kaggle assets и MRI-cache приватные; ссылка на notebook не открывает другу приватные inputs. В Git лежат код, описание и проверяемые результаты, без MRI, report text, labels и весов.

Для исходного feature schema подготовлено [сравнение attention и mean+max](../orthofoundation_meanmax/STATUS.md), которое повторно encoder не запускает. Для нового sharded bank [adapter уже подготовлен и проверен](../orthofoundation_sharded_meanmax/STATUS.md); запуск ждёт реального завершённого attention-bank и отдельно зафиксированного SHA.

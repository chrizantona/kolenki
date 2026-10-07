# OrthoFoundation frozen features → 12-label head

**Этап: pilot V1 и неизменённый retry V2 завершились с `ERROR` до наблюдаемого исполнения Python. Model-assets GPU probe прошёл; проверяем подключение MRI-cache. Полный эксперимент ещё не запущен.**

[Notebook pilot](https://www.kaggle.com/code/alanchoo/rsna-knee-orthofoundation-pilot), kernel `137481721`. В обеих версиях output содержит только log `[]`, traceback и наши run-файлы отсутствуют, GPU-квота не списалась. Точная причина не раскрыта API. [V1](pilot_v1_failure_summary.json) · [V2](pilot_v2_failure_summary.json). Образ и model assets затем проверены отдельным успешным тестом ниже. Теперь короткий probe проверяет семь MRI-cache inputs вместе с competition input.

Отдельный [model-assets probe](https://www.kaggle.com/code/alanchoo/rsna-knee-ortho-asset-startup-probe) завершился `COMPLETE`: Tesla T4, torch 2.10.0+cu128, строгая загрузка checkpoint за 14.61 s, frozen forward `author_no_rope` дал конечный `[1,1024]`, peak allocated 1.237 GB. Это синтетический один-image smoke test: скорость на MRI, обучение головы и точность ещё не проверены. [GPU receipt](asset_gpu_smoke_report.json).

Взяли опубликованный OrthoFoundation-L, основанный на DINOv3 ViT-L/16. Checkpoint скачан: **1 213 056 638 bytes**, SHA256 `385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`. Строгая загрузка всех 368 ключей и CPU forward прошли; выход — конечный 1024-мерный CLS-вектор. Это проверка совместимости, не проверка точности.

Первый вариант воспроизводит авторский проход **без RoPE**, использованный в опубликованном OrthoFoundation wrapper. Канонический DINOv3 forward с RoPE — отдельная будущая абляция.

План первого запуска:

1. GPU pilot: реальные MRI из проверенного cache, скорость, память, конечные признаки и фактическое обновление тестовой головы.
2. При достаточной скорости — признаки всех 4349 weak и 58 Gold studies: 224×224, одиночный grayscale slice повторяется в RGB, до 4 срезов на каждую из 6 категорий серии.
3. Обучение новой головы с фиксированным числом эпох. Gold58 исключён из градиентов и выбора checkpoint. Отдельная weak-fold диагностика и production-head на всех weak.
4. Оценка по каждому диагнозу; затем отдельное решение о включении в ансамбль. Нового leaderboard score пока нет.

Код: [src/orthofoundation](../../src/orthofoundation). Конфигурация: [orthofoundation_frozen.yaml](../../configs/orthofoundation_frozen.yaml). Фактические статусы и время последней проверки: [status.json](status.json). Provenance модели: [asset_manifest.json](asset_manifest.json).

Kaggle assets и MRI-cache приватные; ссылка на notebook не открывает другу приватные inputs. В Git лежат код, описание и проверяемые результаты, без MRI, report text, labels и весов.

После первого полного извлечения готово [контролируемое сравнение attention и mean+max](../orthofoundation_meanmax/STATUS.md), которое повторно encoder не запускает.

# OrthoFoundation frozen features → 12-label head

**Этап: pilot V1 завершился с `ERROR`; выполняем диагностику запуска. Полный эксперимент ещё не запущен.**

[Notebook pilot](https://www.kaggle.com/code/alanchoo/rsna-knee-orthofoundation-pilot), версия **1**, kernel `137481721`. Статус `ERROR` подтверждён. В output нет notebook, traceback или наших run-файлов; log содержит только `[]`. GPU-квота не списалась. По имеющимся данным, сбой произошёл на подготовке запуска; точная причина ещё проверяется. [Наблюдения](pilot_v1_failure_summary.json).

Взяли опубликованный OrthoFoundation-L, основанный на DINOv3 ViT-L/16. Checkpoint скачан: **1 213 056 638 bytes**, SHA256 `385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9`. Строгая загрузка всех 368 ключей и CPU forward прошли; выход — конечный 1024-мерный CLS-вектор. Это проверка совместимости, не проверка точности.

Первый вариант воспроизводит авторский проход **без RoPE**, использованный в опубликованном OrthoFoundation wrapper. Канонический DINOv3 forward с RoPE — отдельная будущая абляция.

План первого запуска:

1. GPU pilot: реальные MRI из проверенного cache, скорость, память, конечные признаки и фактическое обновление тестовой головы.
2. При достаточной скорости — признаки всех 4349 weak и 58 Gold studies: 224×224, одиночный grayscale slice повторяется в RGB, до 4 срезов на каждую из 6 категорий серии.
3. Обучение новой головы с фиксированным числом эпох. Gold58 исключён из градиентов и выбора checkpoint. Отдельная weak-fold диагностика и production-head на всех weak.
4. Оценка по каждому диагнозу; затем отдельное решение о включении в ансамбль. Нового leaderboard score пока нет.

Код: [src/orthofoundation](../../src/orthofoundation). Конфигурация: [orthofoundation_frozen.yaml](../../configs/orthofoundation_frozen.yaml). Фактические статусы и время последней проверки: [status.json](status.json). Provenance модели: [asset_manifest.json](asset_manifest.json).

Kaggle assets и MRI-cache приватные; ссылка на notebook не открывает другу приватные inputs. В Git лежат код, описание и проверяемые результаты, без MRI, report text, labels и весов.

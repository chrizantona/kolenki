# Следующие эксперименты после первых Ortho результатов

Опора для общего submission — существующий ансамбль **0.944**. Frozen Ortho
с K4/CLS обучен полностью: attention/meanmax/Ridge получили Gold0.736/0.735/0.772.
Эти результаты не дают оснований добавлять текущую ветку. Все задачи ниже —
**планы**, новые дорогие обучения ещё не запускались.

| Порядок | Эксперимент | Что меняем и как оцениваем |
|---|---|---|
| 1 | Сопоставить ветки двух участников | Одинаковые IDs, labels version, fold и per-class predictions. Сначала фиксируем независимую validation; public warm start с неизвестной supervised history не считаем чистым OOF |
| 2 | EXP-OF-004: partial fine-tuning | Ortho encoder, разморозить только последние2blocks; тот же K4/input/weak fold0, Gold0градиентов. Pilot на фиксированных weak studies измерит память, скорость и реальные updates; число epochs и бюджет фиксируем до полного run |
| 3 | EXP-OF-005: sampling | При том же frozen encoder/head сравнить K4→K12; один заранее выбранный вариант. Это новый bank, не повторное обучение уже проверенных heads. Выигрыш сначала на870 weak holdout, Gold один раз после фиксированного fit |
| 4 | Local reader | Отдельные meniscus/ACL crops плюс full-knee context, физический FOV/mm и согласованный train/inference. Контроль с тем же ImageNet/MRI encoder и split; не выбирать ROI по test labels |
| 5 | Иерархический pooling | Slice→series→study; сохранять повторные серии. Gated MIL/DSMIL или12cross-attention queries только после появления более содержательных slice/patch features |
| 6 | Внешние данные | fastMRI+ для локальных pathology annotations, MRNet дляACL/meniscus, OAI дляOA, SKM-TEA дляанатомии. Сначала доступ/условия и mappinglabels; отсутствующие цели маскировать. Дополнительный encoder pretraining нельзя автоматически объявлять новым сигналом: Ortho уже pretrained на OAI/fastMRI |

Перед EXP-OF-004 нужен **один технический Linux replay** выбранного до просмотра
labels study. На Mac с OpenCV5.0 обнаружили редкие pixel deltas±1 относительно
исходного Linux4.12.0.88 cache; точная parity в этом новом эталоне не пройдена.
Используем прежние Linux wheel versions, затем проверяем canonical volumes,
выбор series, K4 positions, FP16 CLS и production logits. Tolerances не расширяем
по факту. Это отдельный незавершённый технический шаг, а не объяснение низкого AUC.

Для fine-tuning pilot сохраняем microbatch, gradient accumulation, precision,
число размороженных blocks, failed/skipped updates и memory. Сначала считаем
полный training/inference budget из фактической скорости; непроверенную оценку
не превращаем в многочасовой job. CPU Ridge не требует дальнейшего alpha search.
Если обучение улучшает только train loss, а weak holdout нет, расширение не запускаем.

Заимствуемые схемы: localizer→2.5D ROI из [RSNA2024 spine,5th](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification),
anatomical crops+sequence aggregation из [RSNA2023 abdominal,2nd](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma).
Переносим устройство pipeline; их веса и clinical labels не являются прямой
заменой модели колена. [Внешние источники и условия](RESEARCH.md).

Полное fine-tuning соответствует типу downstream-оценки авторов
[OrthoFoundation](https://github.com/ytrsk/OrthoFoundation/blob/4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470/README.md);
разморозить последние2blocks — наша более дешёвая гипотеза, её результат пока неизвестен.

Общий один submission собираем после validation и сравнения с результатами
тиммейта. Низкая корреляция сама по себе не аргумент за blend; веса выбираем
на допустимой held-out validation, а не перебором по58Gold/publicLB.

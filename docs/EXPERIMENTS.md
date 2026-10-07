# План экспериментов

Текущая опорная система — завершённый ансамбль **0.944**. Следующий выбранный encoder — **OrthoFoundation**. Фактический статус подготовки, загрузки весов, Kaggle run и обучения ведётся в [эксперименте OrthoFoundation](../experiments/orthofoundation_frozen/STATUS.md); этот документ задаёт гипотезы, а не подтверждает их успех.

## Первый эксперимент: frozen OrthoFoundation

Гипотеза: knee-specific pretraining даст признаки, полезные для нашей задачи и достаточно отличающиеся от готовых DINO/ConvNeXt/CoAtNet, чтобы улучшить ансамбль.

1. Закрепить исходный repository commit, модель, checkpoint URL/hash и условия лицензий. Проверить загрузку весов, shape выходов и memory на небольшом batch.
2. Прогнать замороженный encoder по нашим MRI-представлениям и сохранить компактные features. Не обновлять его веса на первом этапе.
3. Обучить на weak labels собственную study-level голову на **12 целей**, с plane/series metadata и missing masks. Сначала использовать простую сравнимую агрегацию.
4. Сохранить predictions и per-class метрики на заранее фиксированной validation; отдельно оценить Gold как ограниченную диагностику.
5. Сравнить ошибки и пробный blend с существующим ансамблем. Только перспективный вариант переносить в полный offline submission pipeline.

Полное дообучение большого DINOv3-L сразу увеличивает стоимость и смешивает две гипотезы: полезность признаков и полезность fine-tuning. Поэтому сначала учим маленькую голову; позже можно разморозить верхние блоки.

Из-за зависания Kaggle при подключении семи cache-архивов подготовлен [запуск по одному архиву](../experiments/orthofoundation_sharded/STATUS.md): глобальный выбор series сохраняется, partial features строго объединяются перед обучением. Продолжение зависит от фактического MRI pilot и GPU-квоты.

Следом подготовлен [EXP-OF-002: attention против mean+max](../experiments/orthofoundation_meanmax/STATUS.md). Он использует те же 4407 сохранённых feature-файлов, labels/split/seed/optimizer и фиксированные 12 эпох. Меняется только тип головы; извлечение признаков не повторяется. Пока полного feature bank нет, запуск остаётся в очереди. Для нового sharded bank потребуется отдельно проверенный schema adapter.

Для оценки именно выигрыша медицинского предобучения нужен отдельный контроль с обычным DINOv3-L при одинаковых input/head/split. Сравнение только с ConvNeXt смешивает архитектуру и pretraining.

## Очередь задач

| Приоритет | Эксперимент | Конкретная проверка |
|---|---|---|
| P0 | OrthoFoundation frozen features + 12-label head | Скорость, качество самостоятельной ветки и польза её blend |
| P0 | Зафиксировать splits и experiment ledger | Ни один held-out study не участвует в обучении оцениваемой собственной модели |
| P1 | Иерархическая голова: windows → series → study | Сохранять повторные серии вместо единственной longest-series; контроль с тем же encoder |
| P1 | Консенсус без двойного учёта Lixin | Сменить только labels recipe, оставить training settings |
| P1 | Meniscus bag: medial/lateral specialist | В [публичном комбинированном stack 0.945](https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-tri-specialist-superstack?scriptVersionId=355737240) есть такая ветка; контролируемый retraining на нашем split, готовые fullfit predictions не OOF |
| P1 | Физический sampling в mm | Сравнить индексные и физические offsets при согласованном train/inference |
| P2 | Query2Label-подобные 12 queries | Cross-attention к tokens вместо текущего attention pooling |
| P2 | DSMIL / gated MIL / mean+max | Выбор локальной находки плюс глобальный контекст; недорогие головы на одинаковых features |
| P2 | Loss с учётом unknown/silence | Не считать отсутствие упоминания уверенным negative; особенно Synovitis |
| P2 | Локальные crops мениск / ACL / cartilage | Local reader вместе с full-knee branch, а не потеря глобального контекста |
| P3 | OrthoDiffusion как 3D encoder | Проверить preprocessing и inference budget до большого обучения |
| P3 | fastMRI+ / MRNet / OAI | Доступ, условия, совпадение целей и дополнительный сигнал до скачивания объёмных данных |

## Что считаем корректным сравнением

- Одинаковые study splits, labels version, sampling, preprocessing и seed для ablations одного компонента.
- Разделяем **validation на weak labels**, **диагностику на 58 Gold** и **public leaderboard**. Эти источники отвечают на разные вопросы.
- Frozen encoder можно использовать в clean validation, если его предобучение не включало наш held-out RSNA supervised split. Для неизвестного публичного knee-trained warm start это нельзя автоматически гарантировать.
- Пишем macro AUC, все 12 per-class AUC, runtime, peak memory, число обученных studies и predictions. Для маленького Gold — paired bootstrap и число positives/negatives.
- Для ансамбля важен выигрыш на одинаковых held-out IDs, а не только score самостоятельной модели. Низкая корреляция прогнозов сама по себе тоже не гарантирует полезный blend.
- Фиксируем blend weights до окончательной оценки; не перебираем десятки настроек на 58 Gold и public LB.

Каждый запуск должен иметь config, provenance, данные о старте/завершении, checkpoint hash, predictions и receipt. Если run не стартовал или упал, записываем именно это.

## Research backlog

- [ ] Разобрать локализацию и 2.5D ROI heads из [RSNA 2024, 5-е место](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification).
- [ ] Разобрать organ crops и sequence aggregation из [RSNA 2023, 2-е место](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma).
- [ ] Прочитать [Query2Label](https://github.com/SlongLiu/query2labels) и спроектировать 12 queries для MRI tokens.
- [ ] Прочитать [DSMIL](https://github.com/binli123/dsmil-wsi) и проверить адаптацию к локальным MRI-находкам.
- [ ] Сопоставить определения внешних labels с 12 RSNA-целями; отсутствующие цели маскировать, не заменять нулями.
- [ ] Обновить snapshot discussions перед следующими дорогими экспериментами.
- [ ] До final submit проверить offline inputs, время на полном hidden scale, native dtype GPU и число fallback-прогнозов. В [discussion 744230](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744230) участник связывает сильное замедление с BF16 на T4; это его наблюдение, не измеренный эффект для нашего ансамбля. Новый OrthoFoundation extraction использует FP16.
- [ ] Проверить отсутствие скрытых приватных dependencies и фактический hidden scoring.

Дедлайн entry/team merge — **15 октября 23:59 UTC**, final submissions — **22 октября 23:59 UTC**; в Москве это 16 и 23 октября в 02:59 соответственно. План ориентирован на доступные Kaggle GPU: pilot с оценкой стоимости обязателен перед большим extraction/training. Актуальную quota проверяем перед запуском, статическую оценку часов не считаем гарантией. [Официальный overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview).

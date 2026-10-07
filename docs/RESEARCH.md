# Внешние модели, datasets и прошлые соревнования

Проверено **7 октября 2026**. Приоритет — опубликованный knee-specific encoder, который можно использовать без нового большого pretraining. Скачиваемые веса проверены отдельно от наличия статьи или обещания авторов; новый score ни для одного ресурса ещё не получен.

## Готовые модели

| Ресурс | Что доступно | Что пробуем / ограничения |
|---|---|---|
| [OrthoFoundation](https://github.com/ytrsk/OrthoFoundation) | DINOv3-L; опубликован checkpoint около 1.2 GB. По описанию авторов pretraining на 1.25M knee images, включая ~894k MRI slices | **Выбран для первого frozen-feature эксперимента.** В repo есть MIT для кода; это не отменяет условий исходных DINOv3 weights |
| OrthoFoundation-MSK-5M | Более крупный набор pretraining и доступные weights | LICENSE в проверенной версии отсутствовал. Не считаем автоматически покрытым MIT исходного проекта |
| [OrthoDiffusion](https://github.com/lt-0123/OrthoDiffusion), [weights](https://huggingface.co/lanstat0123/orthodiffusion) | Опубликованы pretrained spatial backbones; авторы описывают обучение на 15 948 knee MRI | Альтернатива с 3D-признаками. Нужно проверить архитектуру, preprocessing, условия weights и T4 budget |
| [KneePreM / fuse-med-ml](https://github.com/BiomedSciAI/fuse-med-ml) | Код и метод self-supervised 3D pretraining | В проверенном README checkpoints не распространялись. Пока research reference, а не готовый checkpoint для запуска |

Head внешней модели не обязан совпадать с нашей задачей. Используем encoder и обучаем новую 12-label head; переносим существующие внешние heads только при совпадении определения классов.

## Внешние данные

| Dataset | Полезный сигнал | Что решить до использования |
|---|---|---|
| [fastMRI+](https://github.com/microsoft/fastmri-plus) | MRI annotations и bounding boxes патологий; полезен для encoder/localizer | Аннотации открыты отдельно, изображения fastMRI требуют отдельного доступа. Сопоставить классы и проверить data-use terms |
| [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) | 1370 knee exams; abnormality, ACL tear, meniscus tear | Нужен Stanford access/DUA. Общий meniscus tear не заменяет отдельные medial/lateral RSNA labels |
| [OAI](https://nda.nih.gov/oai) | Продольная knee MRI-когорта; cartilage и osteoarthritis | Выбрать MRI/labels, проверить применимый access agreement; разрешение host условное, не снимает условия владельца данных |
| [SKM-TEA](https://github.com/StanfordMIMI/skm-tea) | 3D knee MRI, segmentation и количественные tissue measurements | Может помочь анатомической локализации; отличается от наших clinical targets, доступ и лицензия требуют проверки |
| [KneeCoT — host ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109) | Knee MRI с дополнительным клиническим сигналом | **Организатор запретил использование** из-за institutional data agreement; в experiments не включаем |

Если внешний dataset размечает ACL и общий мениск, это не повод выставлять остальные 10 RSNA-целей в ноль. Для непокрытых целей нужен loss mask или отдельное дополнительное обучение энкодера.

## Правила, которые уже уточнили

Organizers допускают доступные внешние данные и pretrained models, но доступность и лицензия — отдельные проверки. Простая бесплатная регистрация/click-through agreement обычно допустима; обязательное institutional review, legal team или специальное согласование может нарушать равный доступ. [Ответы host](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965).

По OAI host дал условное разрешение при общем бесплатном доступе без сложного institutional sign-off, сохранив ответственность участника за условия источника. Публичный mirror или производный dataset не получает автоматического разрешения только потому, что файл можно скачать. [OAI clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819).

Некоммерческая лицензия сама по себе не означает запрет prize competition по разъяснению host; всё равно проверяем фактический текст применимого соглашения и требования к распространению результатов. MIT нашего репозитория распространяется на наш код, а не автоматически на чужие weights, исходные datasets или перенесённый сторонний код.

Hosted LLM/API для обработки отчётов организатор разрешил при соблюдении доступности и остальных правил. Внешний paid API в текущем pipeline не запускаем; используем уже опубликованные weak labels. [Official discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965).

## Было ли это раньше

По [официальной истории RSNA](https://www.rsna.org/artificial-intelligence/ai-image-challenge), **2026 — первая задача RSNA именно про knee abnormalities**. В 2025 была intracranial aneurysm detection, в 2024 lumbar spine MRI, в 2023 abdominal trauma CT. Старого победителя «RSNA knee 2025» нет. Отдельный [Stanford MRNet challenge](https://stanfordmlgroup.github.io/competitions/mrnet/) проходил раньше; задача на три класса уже закрыта.

| Решение | Что исследуем для переноса |
|---|---|
| [RSNA 2024, 5-е место](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification) | Localizer → 2.5D ROI crops → classification; локальный reader мениска/ACL/хряща вместе с full-knee context |
| [RSNA 2023, 2-е место](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma) | Анатомические crops, отдельные ветки и sequence aggregation; перенос устройства pipeline, не классов CT |
| [Query2Label](https://github.com/SlongLiu/query2labels) | Queries по диагнозам и cross-attention к spatial/slice tokens; собственное обучение на MRI |
| [DSMIL](https://github.com/binli123/dsmil-wsi) | MIL с наиболее подозрительным instance и общим контекстом; проверить применимость локальным повреждениям |

Наш текущий reader уже содержит Transformer и 12 attention pools. Поэтому «добавить attention» не отдельная новая идея. Содержательные изменения — иерархия повторных серий, другие label queries, локализация и другой источник pretraining.

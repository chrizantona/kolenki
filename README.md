# kolenki — RSNA Knee Abnormality Detection 2026

Рабочий репозиторий команды **alanchoo**: исследование данных, воспроизводимый публичный baseline и собственные эксперименты по распознаванию 12 патологий на MRI колена.

**Подтверждённый результат: public score 0.944.** Мы воспроизвели публичный ансамбль GoodPJ, дообучили один ConvNeXt-reader и отправили ансамбль с нашими весами. Обучили также **замороженный OrthoFoundation → признаки срезов → наша голова на 12 целей**; первые результаты этой ветки пока слабее ConvNeXt.

| Запуск | Public score | Что подтверждено |
|---|---:|---|
| GoodPJ, авторский baseline V2 | 0.944 | Исторический результат автора; не наш запуск |
| Наше воспроизведение baseline | 0.943 | Завершённое Kaggle scoring |
| Наш ансамбль с дообученным ConvNeXt | **0.944** | Submission `56891310`, статус `COMPLETE` |
| Frozen OrthoFoundation: attention / meanmax / Ridge | — | Полный bank и все readout обучены и проверены: Gold58 AUC **0.736 / 0.735 / 0.772**. [Результаты](docs/ORTHO_RESULTS.md), [следующие эксперименты](docs/NEXT_EXPERIMENTS.md) |

Результаты выше проверены **7 октября 2026**. Разница между двумя сабмитами — один отображаемый шаг leaderboard; это ещё не доказательство устойчивого улучшения на независимой выборке.

Начать знакомство с проектом можно здесь:

- [Что уже работает: модель, обучение и весь ансамбль](docs/CURRENT_SOLUTION.md).
- [EDA: наблюдения, которые влияют на решение](docs/EDA.md).
- [Эксперименты: приоритеты, критерии и задачи](docs/EXPERIMENTS.md).
- [Обзор discussions и датированный реестр 161 тем](docs/DISCUSSIONS.md).
- [Внешние модели, данные и прошлые соревнования](docs/RESEARCH.md).
- [Как воспроизводить запуски и понимать результаты](docs/REPRODUCIBILITY.md).

[Соревнование](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) · [Исходный GoodPJ V2](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944?scriptVersionId=355300588) · [Наш сабмит](https://www.kaggle.com/code/alanchoo/rsna-knee-944-finetuned?scriptVersionId=355883025)

Наши Kaggle notebooks и подготовленные caches сейчас приватные: наличие ссылки не даёт другому аккаунту доступ к inputs. Здесь публикуются описание и код; большие MRI, caches, checkpoints, токены и тексты медицинских отчётов в Git не хранятся.

[Как устроена OrthoFoundation-ветка](docs/ORTHO_CURRENT.md): данные → замороженный MRI encoder → attention или mean/max голова.

[Отдельная Ortho inference-ветка](experiments/orthofoundation_inference/STATUS.md) подготовлена и
[прошла независимое ревью](experiments/orthofoundation_inference/independent_inference_preparation_review.json):
80 CPU-тестов проходят. Она требует проверенных production-весов и ROOT seal;
реальные test-прогнозы этой веткой ещё не получены.

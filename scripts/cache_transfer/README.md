# CPU transport для оставшихся cache-архивов

Подготовленная utility передаёт один уже зафиксированный ZIP без изменения байтов между Kaggle/CDN и приватным Dataset upload. Она нужна из-за сбоя подключения больших notebook outputs. Это инфраструктура первого OrthoFoundation эксперимента, а не изменение модели.

`build_notebook.py` только генерирует private CPU notebook; remote API он не вызывает. `transfer.py` проверяет размеры, точный Content-Range и полный SHA256, затем выполняет HTTPS PUT по ссылке на один файл. `supervise.py` ограничивает общее время дочернего процесса и удаляет принадлежащий ему scratch под `/tmp`, включая timeout. Cookies, redirects, посторонние hosts и credential fields отвергаются; необработанные URLs/исключения не печатаются.

Account API credential и opaque blob token остаются на Mac. В облако допускаются лишь временные file-scoped read/PUT URLs через отдельный private input. Такой input и реальные URLs запрещены в Git. Проверки production upload/finalization выполняются отдельно; локальные fixtures их не заменяют.

Independent review: 17 protocol fixtures PASS (12.200 s), включая Set-Cookie, поздний upload probe, blocking/trickle response, hard timeout, cleanup и успешную точную передачу. На момент публикации ни одного реального cloud transfer не запускали. Следующие шесть архивов допускаются только после успешного MRI pilot и проверки actual scoped URL host/TTL.

Локальная проверка из этой папки: `python -m unittest -v test_transfer.py`. Зависимости transport — `requests`; генерация notebook дополнительно требует `nbformat`. Fixtures используют localhost и не обращаются к Kaggle API.

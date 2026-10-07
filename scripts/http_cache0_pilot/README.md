# HTTP cache0 pilot transport

The immutable [notebook](../../notebooks/orthofoundation/http_cache0_pilot/train.ipynb) embeds all transport and scientific sources. Its SHA256 is `444a375eb5cd7a085617d4a4ec0b067709ed4f01b654b7477728ba3c986fa796`.

`download.py` and `parent_guard.py` are byte-identical to the reviewed preparation. `transfer_reviewed.py` is the exact existing [Range implementation](../cache_transfer/transfer.py), SHA256 `7212063963481c5fe35ee2d27c61ef101d506a93372326f5501f6771615206ed`. The HTTP wrapper executes only its verified download block. The parent owns the whole process group and temporary files and enforces one 600-second deadline across preparation, HTTP download and the scientific pilot.

The original local `build.py` uses absolute paths on the author's Mac and is intentionally omitted. The notebook already contains the generated sources and configuration; its approved bytes are the reproducible execution artifact. Changes require rebuilding and reviewing a new artifact rather than silently editing this one.

Private inputs remain necessary: global metadata, model assets and a read-only job input. The temporary file-scoped URL belongs only in `cache_read_job.json`; that file is ignored by Git and is never part of this directory. Account credentials are not passed to the child. Download receipts and parent failures redact URL/credential data; raw child output stays in owned temporary scratch and is removed.

The fixture harness is adapted only to repository paths and uses the existing `cache_transfer/test_transfer.py` localhost fixtures. To run it, from this directory:

```sh
python -m unittest -v test_http.py
```

These tests create a localhost HTTP server and owned subprocesses. They do not contact Kaggle. The approved local review recorded 17 passing tests; see the [preparation receipt](../../experiments/orthofoundation_http_cache0_pilot/preparation_receipt.json). Public copying/static verification did not rerun network fixtures. Preparation/review evidence is separate from GPU execution and model-quality results.

# Isolated Ortho inference preparation

Prepared for independent review. The production attention head and its complete
training-bank seal are still future inputs; no Ortho test predictions, blend or
submission have been produced by this branch.

The [separate CLI](../../scripts/run_orthofoundation_inference.py) follows the
[reviewed recipe](../orthofoundation_http_sharded_v2/INFERENCE_PLAN.md) and imports
the pinned canonical decoder and frozen training modules. It accepts only
`head.pt` bound to an externally supplied SHA and the SHA of a separately
validated ROOT full-merge seal. That proof must certify the fixed 12-epoch,
4,349-weak-study production head, with all 58 Gold studies excluded from
gradients and checkpoint selection. Smoke and weak-holdout heads are refused.
Pins are never generated from the checkpoint being admitted.

Each test study is decoded into owned temporary storage. Series are ranked by
their decoded depth capped at 64, then series UID; six plane/fat-suppression
slots, K=4 centers, positions and masks reuse the training code. The literal
353-pixel crop, antialiased 224 resize, replicated grayscale RGB, `author_no_rope`
FP16 encoder and FP16 feature-storage rounding match the frozen experiment.
The attention head runs in FP32/eval, with one sigmoid application. Only one
study's canonical volumes are kept at a time.

The output is independent `_ortho.csv` plus `_ortho_provenance.json`. The former
contains private study predictions and must not be committed. Aggregate
provenance records a separate test identity referencing training identity
`7f0e16dc1c34e8c0920f567db2ebb68f55c470c09ce77dd7c0aa238cd5ec6c77`;
it does not claim the training cache's 4,407-study coverage for test data.
The CLI preserves `_own.csv` and performs no blending or submission.
CSV and receipt are staged together in an owned temporary directory, with
deadline checks after CSV export/hash, receipt write and each exclusive
publication. Any caught failure or overrun removes this attempt's published
links and temporary exports; existing results are never overwritten. The
future parent must also require a successful owned-child receipt, because a
forced process kill cannot run Python cleanup.

CPU fixtures use a stub encoder and tiny synthetic heads. Actual canonical
decode parity was checked on **one** existing private DICOM series, 27 slices,
against four local copies of its canonical cache: exact uint8 equality and
maximum absolute difference 0. This is a small preprocessing check; no real
encoder forward, production-head replay or all-test inference has run.

## Future use after ROOT validation

Keep the reviewed repository layout, the offline encoder asset release and an
explicit test manifest (`test.csv`, `test_series.csv`,
`test_series/<study>/<series>/`). Obtain both external pins from ROOT's completed
merge validation, then invoke in a fresh, CUDA-enabled child:

```sh
python scripts/run_orthofoundation_inference.py \
  --production-head /private/verified-merge/run/head.pt \
  --production-head-sha256 ROOT_SUPPLIED_HEAD_SHA256 \
  --ROOT-training-seal /private/ROOT-full-merge-seal.json \
  --ROOT-training-seal-sha256 ROOT_SUPPLIED_SEAL_SHA256 \
  --data-dir /private/competition-data --split test \
  --assets-dir /private/orthofoundation-l-assets-v1 \
  --output-dir /private/fresh-ortho-inference --max-seconds 3600
```

The pin placeholders deliberately cannot execute. `--prepare-placeholder`
also fails before weights or DICOM reads. The future parent must supply an
owned-process hard timeout and fresh quota checks. The CLI's cooperative timer
starts after production-head admission and includes encoder setup, decoding,
feature/head inference and export/publication checks; imports/admission and
Kaggle mount time are outside that timer. The duration recorded inside the
staged receipt ends after CSV export/hash, before receipt write/publication;
those later operations are checked against the same deadline. Actual runtime,
memory, exact requested study coverage,
finite 12-label probabilities and production replay still require validation
before integration.

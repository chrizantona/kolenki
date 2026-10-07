# EXP-OF-002 on the HTTP V2 sharded feature bank — preparation only

The separate V2 adapter uses the versioned 9300-second extraction orchestrator.
The expected global feature identity is
`7f0e16dc1c34e8c0920f567db2ebb68f55c470c09ce77dd7c0aa238cd5ec6c77`.
Its future source is `alanchoo/rsna-knee-ortho-http-merge-heads-v2`; no completed
attention bank or sealed full-bank SHA exists in this preparation.

The generated CPU notebook is a non-executable placeholder: `source_bank_sha256`
is null and `comparison_launch_allowed` is false. It refuses before data input
resolution. After attention training actually completes, the owner must seal
the immutable full source-bank SHA independently and rebuild the notebook.
Both original attention checkpoints must also replay their saved weak-holdout
and Gold probabilities within the declared FP32 CPU/CUDA tolerances.

The source-bank checks retain all seven cache pins, full global coverage,
ownership, selected series and positions. Only `head_kind` and `experiment`
differ between V2 attention and V2 mean/max configs. Training reuses the same
bank and unchanged core code: seed 42, weak fold 0 and 12 fixed epochs for both
heads. Gold is excluded from gradients and checkpoint selection. The job keeps
the 1500-second Python cap, with 300 seconds for validation and 1200 for fitting.
The same seed does not guarantee identical minibatch order after different head
initialization. This preparation claims no AUC, leaderboard result or trained
V2 head comparison.

The original V1 adapter, builder, configs, core7 and orchestrator remain unchanged.
The V2 comparator changes the runner import and hashes the actual nested runner
and executed comparison alias. The notebook vendors ten files: core7, the nested
V2 orchestrator, unchanged common helper and V2 comparison alias; it contains no
duplicate unversioned runner. Inputs are only small private global metadata and
the future completed feature/head output, without MRI or encoder weights.

The reviewed prepared artifact is under
`artifacts/kaggle/sharded_meanmax_http_v2/cpu/compare.ipynb`. Its static verification
receipt is in the same artifact parent directory; the source-bank pin remains
absent. Independent preparation review passed ([receipt](independent_review_receipt.json));
ROOT verified the exact source/config/notebook hashes and placeholder gate before
publication. This grants no comparison launch until the full attention bank exists.

For local preparation, run the builder from the repository with `PYTHONPATH=src`.
The standalone comparator also requires an explicit V2 `--config` path; the
notebook supplies its vendored V2 config automatically. Neither builder contacts
Kaggle nor invokes the encoder.

Offline verification passed all 53 repository tests: the original 44 tests,
seven V2 adapter fixtures and two V2 provenance/namespace fixtures. The latter
run a fresh vendored import and synthetic CPU comparison without encoder calls.

Sources: [V2 comparator](../../scripts/compare_sharded_heads_http_v2.py),
[V2 builder](../../scripts/build_sharded_head_comparison_notebook_http_v2.py),
[V2 mean/max config](../../configs/orthofoundation_meanmax_http_v2.yaml).

# Full HTTP OrthoFoundation V2

**Полное извлечение и attention training завершены на Kaggle и независимо проверены.** Банк: 4 407 исследований / 85 336 срезов. Weak holdout AUC **0.760328**, Gold58 AUC **0.736061**. Нового leaderboard score нет; ветка пока уступает нашему отдельному ConvNeXt на Gold и в ансамбль не добавлена.

[ROOT full-bank/checkpoint seal](ROOT_actual_full_merge_seal.json), [independent full audit](merge_independent_full_validation.json), [actual execution receipt](merge_execution_receipt.json), [Gold diagnostic](gold_attention_diagnostics/Gold_diagnostics.json). Нижние записи — датированная история запуска; заявления о pending относятся к состоянию на момент соответствующего события.

The separate [pilot V2 journal](../orthofoundation_http_cache0_pilot_v2/STATUS.md)
and [independent MRI proof](../orthofoundation_http_cache0_pilot_v2/root_MRI_validation.json)
are maintained by the launch owner. Root and its independent watcher confirmed
the actual pilot: 16 weak studies, 308 finite MRI features, one finite head update
(loss 0.6678), Gold0, and owned scratch cleanup. This is pilot execution evidence,
not a full feature bank, trained production head, AUC or submission.

[Seven extraction notebooks and merge](../../notebooks/orthofoundation/http_sharded_v2/)
contain the exact scientific8/config of that pilot. Feature identity is
`7f0e16dc1c34e8c0920f567db2ebb68f55c470c09ce77dd7c0aa238cd5ec6c77`.
Selection stays global: 4,407 studies, 24,371 cached series, 21,334 selected
slot-series, 85,336 images; Gold58 excluded from gradient/checkpoint selection.

The seven exact source ZIPs total 70,494,551,381 bytes. Shard0 reads the existing
Dataset0 V1 bytes; shards1–6 read original ZIP kernel outputs V1. Source5 retains
fresh-DICOM provenance and has no claim of identity with the previous NPY output.
The [transport sources](../../scripts/http_sharded_v2/) stream one full SHA per
ZIP into owned `/tmp`, beside exact globally pinned small receipts. The immutable
loader still checks receipt fingerprints and ZIP index/CRC.

Extraction caps remain 1204/1231/1236/1213/1216/1228/1069 seconds, including HTTP,
hashing, setup and export; wire requests add30 seconds. Pilot900 + caps8397 =
9297 ≤ total9300. Merge1500 includes prepare300 + fixed head fit1200, wire1530.
Pre-Python Kaggle preparation lies outside the Python timers.

Each extraction needs a private read job and owner gate with sealed actual pilot
receipt hashes, matching source/identity pins, a measured whole-route forecast
that fits total9300 and every shard cap, and fresh quota. Pilot second-SHA time
stays inside startup conservatively. Shard0 resumes the completed V2 pilot; other
shards mount no MRI kernel outputs. Merge mounts seven complete compact partial
banks and global metadata, without MRI or encoder inputs.

Private preparation passed25 localhost/CPU fixtures in7.313 seconds. Public
copies are verified statically and byte-for-byte, without rerunning those tests.
Their checks cover source pins, precise lookup/resume, honest provenance, budget
gates, cookies/auth/redaction, timeout/cleanup and synthetic complete/missing/
overlapping merges. [Independent review](independent_review.json) passed25 fresh fixtures with no
Critical/Important findings. The private approval is a trusted ROOT assertion:
ROOT independently checked the three actual pilot report SHAs and measurements
before authorizing shard0. [Owner decision](owner_decision.json) records this
boundary. Actual launch and completion receipts are recorded separately.

Future merge ID: `alanchoo/rsna-knee-ortho-http-merge-heads-v2`.
Partials: `alanchoo/rsna-knee-ortho-http-shard-{0..6}-v2`.
The original V1 mean/max comparator needs the separately prepared V2 adapter.

Actual shard0: full10.24 GB archive SHA verified in241.77 seconds; parent
completed and cleaned scratch in401.32 seconds. All632 FP16 feature files
passed ROOT and independent checks of SHA/size, masks, ownership, selected
series/positions and finite nonzero vectors.16 pilot features were resumed
exactly;616 new studies/11,932 new images were encoded. The smoke head made
one finite update on16weak/Gold0. This is not the production12-epoch head.
[Execution receipt](shard0_execution_receipt.json),
[ROOT validation](shard0_ROOT_partial_validation.json),
[independent actual validation](shard0_independent_partial_validation.json).
ROOT authorized shard1 after these checks; Kaggle accepted V1/kernel137512753
at16:41:26 UTC. [Actual shard1 launch receipt](shard1_launch_receipt.json).

Actual shard1:648 independently verified feature files,3,132slots/12,528images;
parent269.21seconds with cleanup; original ZIP10.275GB/fullSHA verified
in105.28seconds. One finite smoke update/Gold0; production heads still untrained.
[Execution receipt](shard1_execution_receipt.json),
[ROOT validation](shard1_ROOT_partial_validation.json),
[independent validation](shard1_independent_partial_validation.json).
ROOT authorized shard2 after actual shard1 checks; Kaggle accepted
V1/kernel137514112 at16:54:31 UTC. [Actual launch receipt](shard2_launch_receipt.json).

Actual shard2:653 independently verified feature files,3,145slots/12,580images;
parent433.11seconds/cleanup, original ZIP10.302GB/fullSHA verified
in214.95seconds. One finite smoke update/Gold0; production heads still untrained.
[Execution receipt](shard2_execution_receipt.json),
[ROOT validation](shard2_ROOT_partial_validation.json),
[independent validation](shard2_independent_partial_validation.json).
The [then-current short-pilot measurements](owner_measurements_short_pilot_v1.json) use the minimum
actual same-identity short-pilot GPU rate58.7047images/s. Faster steady-state
102.7images/s is recorded separately. Conservative full-route forecast7015.12s
still fits total9300 and every shard cap. ROOT authorized shard3 after both actual validations; Kaggle accepted
V1/kernel137515960 at17:12:26 UTC. [Actual launch receipt](shard3_launch_receipt.json).

The fixed-model [inference plan](INFERENCE_PLAN.md) describes the separate Ortho
branch and preprocessing parity checks. Aggregate Gold58 comparison code and its
[independent review](independent_gold_diagnostics_review.json) are ready; actual
Ortho probability CSVs and training receipts do not exist yet.

Actual shard3 COMPLETE:635 independently verified partials,3,083slots/12,332images;
parent1144.438s/exit0/cleanup. Full10.278GB ZIPSHA matched; download978.243s,
0retries. Encoding the remaining12,036images took120.066s. Source8 was also
[downloaded from the actual immutable outputs and hash-checked](shard3_source8_remote_audit.json).
[Execution receipt](shard3_execution_receipt.json), [ROOT validation](shard3_ROOT_partial_validation.json),
[independent validation](shard3_independent_partial_validation.json).

The [legacy model with this slower HTTP measurement](shard3_slower_HTTP_projection.json)
projects9724.195s and fails the9300s/full-route and future shard caps.
ROOT explicitly revised the operational model after four complete measured jobs:
minimum full-stage decode/load/encoder/NPZ rate100.245images/s, minimum observed
HTTP10.506MB/s, and maximum startup residual rounded up to99.0s. The residual
retains imports, weights, initial16-study pilot/cold CUDA warmup, smoke head,
checks, export and cleanup. Projecting all plannedimages also doublecounts pilot
images. [Current owner measurements](owner_measurements.json) and
[independent phase review](independent_phase_forecast_review.json) document the
new empirical model and historical policy without changing scientific settings.

Rounded forecast9154.145s fits9300s and all original caps. Remaining4/5/6 margins
are17.073/17.826/27.247s: another2%HTTP slowdown would break4/5. The model is
an estimate from actual jobs, not a guarantee. Hard watchdogs, fresh quota,
sequential actual verification and a separate ROOT gate remain mandatory.
Shard4 was accepted V1/kernel137521166 at17:59:59UTC; actualPython confirmed.
[Actual launch receipt](shard4_launch_receipt.json). Production heads/AUC remain pending.

Actual shard4 COMPLETE:640 independently verified partials,3,091slots/12,364images;
parent413.632s/exit0/cleanup. Full10.260GB ZIPSHA matched, HTTP210.551s;
new12,064images in113.275s. [Execution receipt](shard4_execution_receipt.json),
[ROOT validation](shard4_ROOT_partial_validation.json), [independent validation](shard4_independent_partial_validation.json).
[Actual timing bounds still cover this job](shard4_ROOT_owner_bounds_revalidation.json),
so ownerphasepolicyb71e remains unchanged. ROOT then approved shard5 only;
accepted V1/kernel137523281 at18:21:01UTC, actualPython confirmed.
[Launch receipt](shard5_launch_receipt.json). This source uses separatelydecoded
DICOM; no original-NPY byte-identity claim. Two extractionparts were incomplete at that launch.

Actual shard5 COMPLETE:645 independently verified partials,3,124slots/12,496images;
parent390.501s/exit0/cleanup. Full10.364GB direct-ZIP SHA matched, HTTP207.795s;
new12,192images in115.784s. [Execution receipt](shard5_execution_receipt.json),
[ROOT validation](shard5_ROOT_partial_validation.json), [independent validation](shard5_independent_partial_validation.json).
[Actual timing bounds cover this job](shard5_ROOT_owner_bounds_revalidation.json);
ownerphasepolicyb71e is unchanged. Source5 preserves its fresh-DICOM provenance
and original-NPY-byte-identity=false. Six completed parts contain3,853study
partials/74,540images, with disjoint slot ownership; these are not unique merged
study counts. Last shard6 was accepted as V1/kernel137525106 at18:40:17UTC; actualPython observed.
[Launch receipt](shard6_launch_receipt.json). Full merged bank and production heads remain pending.

All eight scientific source files were also downloaded from the actual immutable
shard0–2 outputs, closing the earlier local .py inventory omission.
[Aggregate remote-source audit](prior_shards012_remote_source8_aggregate.json)
records24 exactsize/SHA matches; no local source copy is presented as remote evidence.

Actual shard6 COMPLETE:560 independently verified partials,2,699slots/10,796images;
parent254.887s/exit0/cleanup, full8.773GB ZIP SHA matched, HTTP113.256s;
new10,496images in89.900s. [Execution receipt](shard6_execution_receipt.json),
[ROOT validation](shard6_ROOT_partial_validation.json), [independent validation](shard6_independent_partial_validation.json).
[Actual timing bounds cover the final job](shard6_ROOT_owner_bounds_revalidation.json);
ownerphasepolicyb71e remains unchanged. [All-seven extraction receipt](extraction_completion_receipt.json)
confirms4,413partialstudy files representing exactly4,407unique studies,
21,334slots/85,336images. Merge and fixed12epoch heads are the next unexecuted stage.

Merge+attention accepted V1/kernel137526746 at18:56:55UTC, actualPython observed.
[Launch receipt](merge_launch_receipt.json) binds the exact reviewed sources, all7
actual partialmanifest seals, offline pinnedT4/image and Python1500/wire1530 caps.
[Repaired launcher review](independent_merge_launcher_review.json) passed17fixtures
and refused an independentlysealed mismatched manifest across all7actualsources.
Production gradient start/completion and AUC remain unverified at this receipt.


Actual merge+attention COMPLETE: exact union of all7 sealed banks, 4,407 unique studies, 21,334 slots /85,336 images. Both fixed12epoch heads trained: CV3,479 weak studies/660updates, production4,349 weak/816updates, Gold gradients/selection0. Parent139.581s, training39.490s, scratch cleaned. ROOT independently reconstructed every merged array and replayed both heads; a separate reviewer confirmed bank/state/predictions/AUC. No new submission. CPU meanmax executable now independently reviewed with actual full-bank pin; execution requires a separate one-job ROOT authority.


The completed [CPU meanmax comparison](../orthofoundation_sharded_meanmax_http_v2/result_receipt.json) gives weak0.768476/Gold0.735397; the separately completed [fixed localCPU Ridge](../orthofoundation_ridge_probe/independent_actual_Ridge_validation.json) gives0.775881/0.771916. Both independently validated. [Final same-Gold diagnostics](gold_final_diagnostics/Gold_diagnostics.json) retain the sole predeclared50/50attention blend; no meanmax/Ridge blend search. [Current strategy](../../docs/NEXT_EXPERIMENTS.md). No new submit or running job.

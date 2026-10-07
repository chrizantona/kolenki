# Full HTTP OrthoFoundation V2

**Actual extraction 3/7 verified: shards0–2 COMPLETE, 1,933 study partials / 37,348 images.
Merge and production training have not run; AUC remains unmeasured.** [Preparation receipt](preparation_receipt.json) and
[source-index pins](source_index.json) contain only aggregate counts, references,
file sizes and hashes, without medical rows or scoped access URLs.

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
The [updated owner measurements](owner_measurements.json) use the minimum
actual same-identity short-pilot GPU rate58.7047images/s. Faster steady-state
102.7images/s is recorded separately. Conservative full-route forecast7015.12s
still fits total9300 and every shard cap. ROOT authorized shard3 after both actual validations; Kaggle accepted
V1/kernel137515960 at17:12:26 UTC. [Actual launch receipt](shard3_launch_receipt.json).

The fixed-model [inference plan](INFERENCE_PLAN.md) describes the separate Ortho
branch and preprocessing parity checks. Aggregate Gold58 comparison code and its
[independent review](independent_gold_diagnostics_review.json) are ready; actual
Ortho probability CSVs and training receipts do not exist yet.

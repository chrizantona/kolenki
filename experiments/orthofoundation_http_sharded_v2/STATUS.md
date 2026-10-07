# Full HTTP OrthoFoundation V2

**Independent static review PASS; first full extraction shard authorized.
Kaggle accepted shard0 V1/kernel137510617 at16:22:26 UTC.
Python execution and complete bank are not yet independently verified; merge
and production training have not run. [Actual launch receipt](shard0_launch_receipt.json).** [Preparation receipt](preparation_receipt.json) and
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

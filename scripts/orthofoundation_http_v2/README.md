# HTTP cache0 pilot V2

This is a versioned copy of the original sharded orchestrator. It changes four
exact budget pins: pilot 600 → 900 seconds, total extraction 9000 → 9300 seconds,
the strict config guard 9000 → 9300, and the projection default 9000 → 9300.
The original seven core modules, original orchestrator and V1 notebooks stay
unchanged. The matching config is `configs/orthofoundation_frozen_http_v2.yaml`.

The remaining extraction pool stays 7980 seconds. Seven shard caps remain
1204, 1231, 1236, 1213, 1216, 1228 and 1069 seconds; with the V2 pilot they total
9297 seconds. Head fitting keeps its 1200-second budget within a 1500-second
head job. Download, the second full archive SHA, preparation, MRI extraction,
head smoke and export share the pilot's single 900-second Python deadline.
Kaggle's requested server timeout is separately 930 seconds.

The notebook vendors this runner beside `orthofoundation/`. For a local import
of this nested source, put the repository's `src/` on `PYTHONPATH`.
The changed orchestrator SHA creates a distinct feature identity; existing
V1 feature banks cannot be resumed under V2.

Preparation alone proves neither an MRI pilot PASS nor trained model metrics.
Full extraction remains gated on finite pilot features, a finite head update,
a forecast fitting the 9300-second total and original shard caps, and fresh
available GPU quota.

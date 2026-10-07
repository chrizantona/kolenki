#!/usr/bin/env python3
"""Explicit partial-feature fallback; original all-seven full runner stays intact.

Global series selection uses verified metadata from every shard. Each extraction
job owns only specified shards. Partial features cannot enter training until an
exact coverage merge has succeeded. No CLI accepts relaxed competition counts.

This bank has an explicit sharded provenance schema. The existing monolithic
mean/max comparison script needs a separately reviewed schema adapter before
using it; that adapter is planned, not implemented here.
"""
import os
import time

# The private launcher exports its earliest Python-cell monotonic start. This
# includes vendoring/imports; Kaggle's pre-Python mount preparation is excluded.
_MODULE_START = float(os.environ.get("RSNA_SHARDED_BOOTSTRAP_START", time.monotonic()))

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import traceback

LOCAL_SRC = Path(__file__).resolve().parents[1] / "src"
if LOCAL_SRC.is_dir():
    sys.path.insert(0, str(LOCAL_SRC))

import numpy as np
import pandas as pd
import torch

from orthofoundation.cache import cache_receipt_fingerprint, load_cache_mixed, MixedVolumeLoader
from orthofoundation.data import LABELS, build_table, ids_hash, slice_indices, validated_labels
from orthofoundation.encoder import file_sha, load_encoder
from orthofoundation.run import extract, extraction_identity, write_json
from orthofoundation.train import fit, run_training

REQUIRED_COUNTS = {"studies": 4407, "weak": 4349, "gold": 58, "series": 24371,
                   "selected_series": 21334, "images": 85336}
PILOT_BUDGET_SECONDS = 900
STARTUP_RESERVE_SECONDS = 60
TOTAL_EXTRACTION_BUDGET_SECONDS = 9300
HEAD_FIT_BUDGET_SECONDS = 1200
HEAD_JOB_RESERVE_SECONDS = 1500


def allocated_extraction_cap(images):
    if not isinstance(images, int) or not 0 < images <= REQUIRED_COUNTS["images"]:
        raise RuntimeError("Invalid owned image count for budget allocation")
    pool = TOTAL_EXTRACTION_BUDGET_SECONDS - PILOT_BUDGET_SECONDS - 7 * STARTUP_RESERVE_SECONDS
    return math.floor(pool * images / REQUIRED_COUNTS["images"]) + STARTUP_RESERVE_SECONDS


def check_elapsed(start, limit):
    elapsed = time.monotonic() - start
    if elapsed > limit:
        raise RuntimeError("Python orchestrator walltime budget exceeded; no completed-stage claim")
    return elapsed


def verify_global_members(root, manifest, expected_manifest_sha):
    root = Path(root)
    if file_sha(root / "global_metadata_manifest.json") != expected_manifest_sha:
        raise RuntimeError("Global metadata manifest SHA differs from pinned private input")
    members = manifest["members"]
    paths = [row["path"] for row in members]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate global metadata member paths")
    for row in members:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe global metadata member path")
        file = root / relative
        if file.is_symlink() or not file.is_file() or file.stat().st_size != row["bytes"] or file_sha(file) != row["sha256"]:
            raise RuntimeError("Global metadata member bytes/SHA mismatch: " + row["path"])


def build_global_plan(series, metadata, labels, official, required_counts=REQUIRED_COUNTS):
    """Pure global selector; tests can provide synthetic count expectations."""
    if not series.SeriesInstanceUID.is_unique or not metadata.series.is_unique:
        raise RuntimeError("Duplicate official/cache series UID")
    if set(series.SeriesInstanceUID) != set(metadata.series):
        raise RuntimeError("Global cache series coverage differs from official metadata")
    if not labels.index.is_unique or set(labels.index) != set(official.index):
        raise RuntimeError("Global label study coverage differs from official metadata")
    pairs = dict(zip(series.SeriesInstanceUID, series.StudyInstanceUID))
    if any(pairs[row.series] != row.study for row in metadata.itertuples()):
        raise RuntimeError("Global cache series-to-study mapping differs from official metadata")
    if set(series.StudyInstanceUID) != set(official.index):
        raise RuntimeError("Official series study coverage differs from training table")
    if not metadata.n.between(3, 64).all() or not metadata.shard.isin(range(7)).all():
        raise RuntimeError("Invalid global cached slice count or shard")
    counts = dict(zip(metadata.series, metadata.n.astype(int)))
    owner = dict(zip(metadata.series, metadata.shard.astype(int)))
    table = build_table(series, counts)
    ids = sorted(official.index)
    if any(not isinstance(uid, str) or re.fullmatch(r"[0-9.]+", uid) is None for uid in ids):
        raise RuntimeError("Invalid official study UID for feature filenames")
    if any(not any(table.get(uid, [])) for uid in ids):
        raise RuntimeError("Study has no globally usable MRI series")
    selected = {uid: [choices[0] if choices else "" for choices in table[uid]] for uid in ids}
    masks = {uid: np.array([bool(value) for value in selected[uid]], dtype=bool) for uid in ids}
    gold = set(official.index[official[LABELS].notna().all(axis=1)])
    if set(labels.index[labels.is_gold.eq(1)]) != gold:
        raise RuntimeError("Gold flags differ from official labels")
    actual = {"studies": len(ids), "weak": len(ids) - len(gold), "gold": len(gold), "series": len(metadata),
              "selected_series": sum(int(mask.sum()) for mask in masks.values())}
    actual["images"] = actual["selected_series"] * 4
    if actual != required_counts:
        raise RuntimeError("Global competition counts differ: " + json.dumps(actual, sort_keys=True))
    lines = []
    for uid in ids:
        for slot, sr in enumerate(selected[uid]):
            if sr:
                lines.append(f"{uid}:{slot}:{sr}:{owner[sr]}:{','.join(map(str, slice_indices(counts[sr],4)))}")
    return {"ids": ids, "selected": selected, "masks": masks, "counts": counts, "owner": owner,
            "labels": labels, "official": official, "gold": gold, "summary": actual,
            "selected_plan_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}


def validate_global(root, expected_manifest_sha):
    root = Path(root)
    manifest = json.loads((root / "global_metadata_manifest.json").read_text())
    verify_global_members(root, manifest, expected_manifest_sha)
    pins = json.loads((root / "cache_input_fingerprints.json").read_text())
    sources = json.loads((root / "cache_source_plan.json").read_text())
    if {row["shard"] for row in sources} != set(range(7)) or len(sources) != 7:
        raise RuntimeError("Global source plan must contain all seven distinct shards")
    frames, fingerprints, pin_by_shard = [], [], {}
    for source in sorted(sources, key=lambda row: row["shard"]):
        shard = source["shard"]
        receipts = root / "cache_receipts" / f"shard_{shard}"
        fingerprint = cache_receipt_fingerprint(receipts)
        fingerprints.append(fingerprint)
        pin_by_shard[shard] = fingerprint
        summary = json.loads((receipts / "cache_summary.json").read_text())
        if summary["preprocess_sha256"] != manifest["preprocess_sha256"]:
            raise RuntimeError("Global cache preprocessing SHA differs")
        frame = pd.read_csv(receipts / "cache384_train_meta.csv")
        frame["shard"] = shard
        frames.append(frame)
    fingerprints.sort(key=lambda row: row["plan_sha256"])
    if fingerprints != pins or len(pins) != 7:
        raise RuntimeError("Global receipt fingerprints differ from all-seven pins")
    if "cache_input_fingerprints" in manifest and manifest["cache_input_fingerprints"] != pins:
        raise RuntimeError("Global manifest pins differ from fingerprint file")
    labels, official, _ = validated_labels(root / "labels_v0.csv", root)
    plan = build_global_plan(pd.read_csv(root / "train_series.csv"), pd.concat(frames, ignore_index=True), labels, official)
    declared = manifest["counts"]
    for actual_key, manifest_key in {"studies": "n_studies", "weak": "n_weak", "gold": "n_gold", "series": "n_cached_series",
                                     "selected_series": "selected_series", "images": "selected_images_k4"}.items():
        if declared[manifest_key] != plan["summary"][actual_key]:
            raise RuntimeError("Global manifest declared counts differ from verified data")
    if declared["slices_per_slot"] != 4 or manifest["source_plan"] != sources:
        raise RuntimeError("Global manifest selection/source plan differs")
    plan.update(root=root, manifest=manifest, manifest_sha256=expected_manifest_sha,
                fingerprints=fingerprints, pin_by_shard=pin_by_shard,
                kernel_by_shard={row["shard"]: row["kernel_id"] for row in sources})
    return plan


def owned_mask(plan, uid, shards):
    return np.array([bool(sr) and plan["owner"][sr] in shards for sr in plan["selected"][uid]], dtype=bool)


def source_hashes():
    import orthofoundation
    root = Path(orthofoundation.__file__).parent
    return {path.name: file_sha(path) for path in sorted(root.glob("*.py"))}


def make_identity(plan, config, assets_manifest_sha):
    if config["slices_per_slot"] != 4 or config["resolution"] != 224 or config["fov"] != .92 or config["extraction_variant"] != "author_no_rope":
        raise RuntimeError("Sharded fallback must preserve frozen scientific extraction settings")
    if config["max_extract_seconds"] != 9300 or config["head_epochs"] != 12 or config["weak_holdout_fold"] != 0:
        raise RuntimeError("Sharded fallback must retain total extraction budget and fixed head schedule/split")
    provenance = {"schema": "sharded_frozen_v1", "global_metadata_manifest_sha256": plan["manifest_sha256"],
                  "global_counts": plan["summary"], "selected_plan_sha256": plan["selected_plan_sha256"],
                  "all_study_ids_sha256": ids_hash(plan["ids"]), "all_cache_fingerprints": plan["fingerprints"],
                  "assets_manifest_sha256": assets_manifest_sha, "checkpoint_sha256": config["checkpoint_sha256"],
                  "orchestrator_sha256": file_sha(__file__), "core_source_sha256": source_hashes(),
                  "feature_config": {key: config[key] for key in ("encoder", "feature_dim", "resolution", "slices_per_slot", "fov", "extraction_variant")},
                  "gold_gradient_policy": "58 Gold studies excluded; no Gold checkpoint selection",
                  "equivalence": "Same sampling and encoder; honestly sharded execution, FP16 batch-shape effects not claimed byte-identical."}
    identity = extraction_identity(provenance)
    provenance["feature_identity"] = identity
    return identity, provenance


def validate_partial_feature(path, uid, plan, shards, identity, provenance):
    expected = owned_mask(plan, uid, shards)
    if not expected.any():
        raise RuntimeError("Partial feature study owns no globally selected slot")
    with np.load(path, allow_pickle=False) as row:
        if row["study"].item() != uid or row["identity"].item() != identity:
            raise RuntimeError("Partial feature identity / UID mismatch")
        if row["global_metadata_manifest_sha256"].item() != provenance["global_metadata_manifest_sha256"] or row["orchestrator_sha256"].item() != provenance["orchestrator_sha256"]:
            raise RuntimeError("Partial feature metadata/source provenance mismatch")
        features, mask, positions = row["features"], row["mask"], row["positions"]
        if features.shape != (6, 4, 1024) or features.dtype != np.float16 or mask.shape != (6,) or mask.dtype != np.bool_ or positions.shape != (6, 4):
            raise RuntimeError("Partial feature shape/dtype differs")
        if not np.array_equal(mask, expected) or not np.array_equal(row["ownership_mask"], expected) or not np.array_equal(row["global_mask"], plan["masks"][uid]):
            raise RuntimeError("Partial ownership/global mask differs from global selected plan")
        if not np.array_equal(row["shards"], np.asarray(sorted(shards))):
            raise RuntimeError("Partial feature shard group differs")
        if not np.isfinite(features).all() or not np.isfinite(positions).all() or not np.all(features[~mask] == 0) or not np.all(positions[~mask] == 0):
            raise RuntimeError("Partial features are nonfinite or unowned slots nonzero")
        if row["selected_series"].shape != (6,):
            raise RuntimeError("Partial selected-series shape differs")
        for slot, sr in enumerate(plan["selected"][uid]):
            if mask[slot]:
                expected_positions = slice_indices(plan["counts"][sr], 4).astype(np.float32) / (plan["counts"][sr] - 1)
                if row["selected_series"][slot] != sr or not np.array_equal(positions[slot], expected_positions):
                    raise RuntimeError("Partial selected series / slice positions differ from global selection")
            elif row["selected_series"][slot] != "":
                raise RuntimeError("Unowned slot has a selected series")
        return features.copy(), mask.copy(), positions.copy()


def augment_partial_files(directory, ids, plan, shards, identity, provenance):
    for uid in ids:
        path = directory / f"{uid}.npz"
        with np.load(path, allow_pickle=False) as row:
            value = {name: row[name].copy() for name in row.files}
        expected = owned_mask(plan, uid, shards)
        if value["identity"].item() != identity or not np.array_equal(value["mask"], expected):
            raise RuntimeError("Core extractor wrote an unexpected partial mask / identity")
        value.update(global_mask=plan["masks"][uid], ownership_mask=expected, shards=np.asarray(sorted(shards)),
                     orchestrator_sha256=provenance["orchestrator_sha256"],
                     global_metadata_manifest_sha256=provenance["global_metadata_manifest_sha256"])
        with Path(str(path) + ".tmp").open("wb") as stream:
            np.savez_compressed(stream, **value)
        Path(str(path) + ".tmp").replace(path)
        validate_partial_feature(path, uid, plan, shards, identity, provenance)


def partial_manifest(output, ids, plan, shards, identity, provenance, complete, pilot):
    entries = [{"study": uid, "file": f"{uid}.npz", "bytes": (output / "features" / f"{uid}.npz").stat().st_size,
                "sha256": file_sha(output / "features" / f"{uid}.npz")} for uid in ids]
    result = {"schema": "sharded_frozen_v1", "feature_identity": identity, "provenance": provenance,
              "shards": sorted(shards), "pilot_only": pilot, "extraction_complete_for_shards": complete,
              "partial_studies": len(ids), "selected_slot_series": sum(int(owned_mask(plan, uid, shards).sum()) for uid in ids),
              "features": entries, "head_training_complete": False}
    result["images_k4"] = result["selected_slot_series"] * 4
    write_json(result, output / "partial_manifest.json")
    return result


def validate_mounted_cache(roots, shards, plan):
    if len(roots) != len(shards) or len(set(shards)) != len(shards) or not set(shards).issubset(range(7)):
        raise RuntimeError("Mounted roots/shard ownership must be distinct and one-to-one")
    mapping, counts, summaries, fingerprints = load_cache_mixed(roots)
    expected = sorted((plan["pin_by_shard"][shard] for shard in shards), key=lambda value: value["plan_sha256"])
    if fingerprints != expected:
        raise RuntimeError("Mounted cache fingerprints differ from their global all-seven pins")
    expected_series = {sr for sr, owner in plan["owner"].items() if owner in shards}
    if set(mapping) != expected_series or any(counts[sr] != plan["counts"][sr] for sr in counts):
        raise RuntimeError("Mounted cache series coverage/counts differ from global ownership")
    return mapping


def budget_projection(rate, startup, images_for_shards, planned_jobs, extraction_limit=9300, allocated_cap=None):
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(startup) or startup < 0 or not 1 <= planned_jobs <= 7:
        raise RuntimeError("Invalid measured throughput/startup/job count")
    if extraction_limit != TOTAL_EXTRACTION_BUDGET_SECONDS:
        raise RuntimeError("Total frozen budget must remain 9000 seconds")
    cap = allocated_cap if allocated_cap is not None else allocated_extraction_cap(images_for_shards)
    if cap <= 0:
        raise RuntimeError("Allocated shard budget must be positive")
    all_shards = REQUIRED_COUNTS["images"] / rate + startup * planned_jobs
    total = PILOT_BUDGET_SECONDS + all_shards
    own = images_for_shards / rate + startup
    return {"images_per_second": rate, "startup_seconds": startup,
            "planned_extraction_jobs": planned_jobs, "projected_extraction_seconds_for_shards": own,
            "allocated_extraction_cap_seconds": cap, "fits_allocated_shard_budget": own <= cap,
            "reserved_pilot_seconds": PILOT_BUDGET_SECONDS, "projected_all_shards_frozen_seconds": all_shards,
            "projected_total_frozen_seconds": total, "projected_total_including_head_seconds": total + HEAD_FIT_BUDGET_SECONDS,
            "projected_total_including_head_job_seconds": total + HEAD_JOB_RESERVE_SECONDS,
            "head_fit_budget_seconds": HEAD_FIT_BUDGET_SECONDS, "head_job_reserve_seconds": HEAD_JOB_RESERVE_SECONDS,
            "total_frozen_budget_seconds": extraction_limit, "fits_total_frozen_budget": total <= extraction_limit,
            "timer_boundary": "Python orchestrator_start through final stage export; pre-Python Kaggle input mounting excluded",
            "gpu_quota_guaranteed": False}


def run_extract(args, plan, config, identity, provenance):
    shards = set(args.shards)
    mapping = validate_mounted_cache(args.cache_roots, args.shards, plan)
    all_ids = [uid for uid in plan["ids"] if owned_mask(plan, uid, shards).any()]
    images = sum(int(owned_mask(plan, uid, shards).sum()) for uid in all_ids) * 4
    allowed_cap = PILOT_BUDGET_SECONDS if args.mode == "pilot" else allocated_extraction_cap(images)
    if args.max_seconds <= 0 or args.max_seconds > allowed_cap:
        raise RuntimeError("Requested shard/pilot runtime exceeds its allocated hard cap")
    local_table = {uid: [[sr] if sr and plan["owner"][sr] in shards else [] for sr in plan["selected"][uid]] for uid in all_ids}
    pilot_ids = [uid for uid in all_ids if uid not in plan["gold"]][:config["pilot_studies"]]
    if len(pilot_ids) != config["pilot_studies"]:
        raise RuntimeError("Not enough local weak studies for pilot smoke")
    output, start = args.output_dir, args.orchestrator_start
    output.mkdir(parents=True, exist_ok=True)
    (output / "features").mkdir(exist_ok=True)
    write_json(provenance, output / "provenance.json")
    write_json(config, output / "config.json")
    prior = None
    if args.resume_partial:
        previous = json.loads((args.resume_partial / "partial_manifest.json").read_text())
        if previous["feature_identity"] != identity or set(previous["shards"]) != shards or previous["provenance"] != provenance:
            raise RuntimeError("Resume partial group/global provenance differs")
        prior = json.loads((args.resume_partial / "pilot_report.json").read_text())
        if prior["feature_identity"] != identity:
            raise RuntimeError("Resume throughput identity differs")
        for entry in previous["features"]:
            uid = entry["study"]
            source = args.resume_partial / "features" / entry["file"]
            if uid not in all_ids or entry["file"] != uid + ".npz" or file_sha(source) != entry["sha256"]:
                raise RuntimeError("Resume partial feature coverage/hash differs")
            validate_partial_feature(source, uid, plan, shards, identity, provenance)
            shutil.copyfile(source, output / "features" / entry["file"])
    if file_sha(args.assets_manifest) != args.assets_manifest_sha256:
        raise RuntimeError("Encoder asset manifest SHA differs from pin")
    model = load_encoder(args.dinov3_repo, args.checkpoint, config["checkpoint_sha256"], config["checkpoint_bytes"])
    startup_seconds = time.monotonic() - start
    loader = MixedVolumeLoader(mapping)
    pilot = extract(model, pilot_ids, local_table, loader, output, identity, config, start, args.max_seconds)
    if not pilot["complete"]:
        raise RuntimeError("Shard pilot extraction stopped before completion")
    augment_partial_files(output / "features", pilot_ids, plan, shards, identity, provenance)
    remaining = args.max_seconds - (time.monotonic() - start)
    if remaining <= 0:
        raise RuntimeError("Pilot has no remaining time for finite head smoke")
    _, smoke = fit(pilot_ids, plan["labels"], output / "features", identity, config, "cuda", output / "smoke_head.pt", epochs=1, max_seconds=min(120, remaining))
    check_elapsed(start, args.max_seconds)
    if smoke["optimizer_updates"] < 1:
        raise RuntimeError("Shard pilot had no finite head optimizer update")
    if pilot["images_encoded"]:
        rate = pilot["images_encoded"] / pilot["seconds"]
    elif prior:
        rate = prior["budget_projection"]["images_per_second"]
    else:
        raise RuntimeError("Cannot estimate shard GPU budget without measured throughput")
    projection = budget_projection(rate, startup_seconds, images, args.planned_extraction_jobs, config["max_extract_seconds"], allocated_extraction_cap(images))
    projected = projection["projected_extraction_seconds_for_shards"]
    report = {"pilot_only": args.mode == "pilot", "feature_identity": identity, "shards": sorted(shards),
              "feature_extraction": pilot, "head_smoke": smoke, "total_images_for_shards": images,
              "projected_extraction_seconds_for_shards": projected, "max_seconds": args.max_seconds,
              "budget_projection": projection,
              "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "head_training_complete": False}
    write_json(report, output / "pilot_report.json")
    partial_manifest(output, pilot_ids, plan, shards, identity, provenance, complete=False, pilot=True)
    check_elapsed(start, args.max_seconds)
    print(json.dumps(report), flush=True)
    if args.mode == "pilot":
        return
    if not projection["fits_total_frozen_budget"] or not projection["fits_allocated_shard_budget"] or projected > args.max_seconds:
        raise RuntimeError("Measured shard extraction ETA exceeds GPU budget")
    result = extract(model, all_ids, local_table, loader, output, identity, config, start, args.max_seconds)
    present = [uid for uid in all_ids if (output / "features" / f"{uid}.npz").is_file()]
    augment_partial_files(output / "features", present, plan, shards, identity, provenance)
    # Hash/export all partial files while still marked incomplete. Only finalize
    # after checking the whole Python stage, including this final export cost.
    manifest = partial_manifest(output, present, plan, shards, identity, provenance, complete=False, pilot=False)
    completed = result["complete"] and present == all_ids and time.monotonic() - start <= args.max_seconds
    manifest["extraction_complete_for_shards"] = completed
    write_json(manifest, output / "partial_manifest.json")
    extraction_report = {"head_training_complete": False, "feature_extraction": result,
                "partial_studies": len(present), "images_k4": manifest["images_k4"],
                "extraction_complete_for_shards": completed, "feature_identity": identity,
                "orchestrator_elapsed_seconds": time.monotonic() - start,
                "timer_boundary": projection["timer_boundary"]}
    write_json(extraction_report, output / "extraction_report.json")
    if time.monotonic() - start > args.max_seconds:
        completed = False
        manifest["extraction_complete_for_shards"] = False
        extraction_report["extraction_complete_for_shards"] = False
        write_json(manifest, output / "partial_manifest.json")
        write_json(extraction_report, output / "extraction_report.json")
    if not completed or manifest["images_k4"] != images:
        raise RuntimeError("Shard extraction incomplete; partial files retained, no head training")


def merge_partials(plan, roots, output, identity, provenance):
    """No gradients; exact disjoint shard/slot coverage before final feature bank."""
    output = Path(output)
    manifests, seen_shards, feature_map = [], set(), collections.defaultdict(list)
    for root in map(Path, roots):
        manifest = json.loads((root / "partial_manifest.json").read_text())
        if manifest["feature_identity"] != identity or manifest["provenance"] != provenance:
            raise RuntimeError("Partial manifest global identity/provenance differs")
        if manifest["pilot_only"] or not manifest["extraction_complete_for_shards"]:
            raise RuntimeError("Pilot/incomplete partial cannot be merged for training")
        shards = set(manifest["shards"])
        if not shards or len(shards) != len(manifest["shards"]) or not shards.issubset(range(7)) or seen_shards & shards:
            raise RuntimeError("Duplicate/overlapping or unexpected shard ownership")
        seen_shards |= shards
        expected_ids = {uid for uid in plan["ids"] if owned_mask(plan, uid, shards).any()}
        entries = manifest["features"]
        ids = [entry["study"] for entry in entries]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise RuntimeError("Partial study coverage missing, duplicate, or unexpected")
        expected_slots = sum(int(owned_mask(plan, uid, shards).sum()) for uid in expected_ids)
        if manifest["partial_studies"] != len(expected_ids) or manifest["selected_slot_series"] != expected_slots or manifest["images_k4"] != expected_slots * 4:
            raise RuntimeError("Partial manifest counts differ from global ownership")
        for entry in entries:
            uid = entry["study"]
            if entry["file"] != uid + ".npz":
                raise RuntimeError("Unsafe or mismatched partial feature filename")
            path = root / "features" / entry["file"]
            if path.stat().st_size != entry["bytes"] or file_sha(path) != entry["sha256"]:
                raise RuntimeError("Partial feature file size/SHA differs")
            validate_partial_feature(path, uid, plan, shards, identity, provenance)
            feature_map[uid].append((path, shards))
        manifests.append(manifest)
    if seen_shards != set(range(7)) or set(feature_map) != set(plan["ids"]):
        raise RuntimeError("Merged bank missing a shard or study")
    # All input contracts were checked before writing a single merged feature.
    target = output / "features"
    target.mkdir(parents=True, exist_ok=True)
    total_slots = 0
    for uid in plan["ids"]:
        features = np.zeros((6, 4, 1024), dtype=np.float16)
        mask = np.zeros(6, dtype=bool)
        positions = np.zeros((6, 4), dtype=np.float32)
        for path, shards in feature_map[uid]:
            value, owned, pos = validate_partial_feature(path, uid, plan, shards, identity, provenance)
            if (mask & owned).any():
                raise RuntimeError("Duplicate owned slot during merge")
            features[owned], positions[owned], mask[owned] = value[owned], pos[owned], True
        if not np.array_equal(mask, plan["masks"][uid]):
            raise RuntimeError("Merged slots do not equal global expected mask")
        total_slots += int(mask.sum())
        path = target / f"{uid}.npz"
        with Path(str(path) + ".tmp").open("wb") as stream:
            np.savez_compressed(stream, features=features, mask=mask, positions=positions,
                                study=uid, identity=identity, selected_series=np.asarray(plan["selected"][uid]),
                                global_metadata_manifest_sha256=provenance["global_metadata_manifest_sha256"],
                                orchestrator_sha256=provenance["orchestrator_sha256"])
        Path(str(path) + ".tmp").replace(path)
    if total_slots != plan["summary"]["selected_series"] or total_slots * 4 != plan["summary"]["images"]:
        raise RuntimeError("Merged series/image counts differ from full global plan")
    result = {"feature_bank_complete": True, "head_training_complete": False, "feature_identity": identity,
              "n_studies": len(feature_map), "selected_slot_series": total_slots, "images_k4": total_slots * 4,
              "partial_jobs": len(manifests), "shards": sorted(seen_shards)}
    write_json(result, output / "merge_report.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "extract", "merge-train"), required=True)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--global-manifest-sha256", required=True)
    parser.add_argument("--assets-manifest-sha256", required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, nargs="+")
    parser.add_argument("--cache-roots", type=Path, nargs="+")
    parser.add_argument("--dinov3-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets-manifest", type=Path)
    parser.add_argument("--resume-partial", type=Path)
    parser.add_argument("--partial-roots", type=Path, nargs="+")
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--planned-extraction-jobs", type=int, default=7)
    args = parser.parse_args()
    args.orchestrator_start = _MODULE_START
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; refuse CPU extraction/training fallback")
    config = json.loads(args.config_json.read_text())
    plan = validate_global(args.global_root, args.global_manifest_sha256)
    identity, provenance = make_identity(plan, config, args.assets_manifest_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(provenance, args.output_dir / "provenance.json")
    write_json(config, args.output_dir / "config.json")
    if args.mode in ("pilot", "extract"):
        if not args.shards or not args.cache_roots or not args.dinov3_repo or not args.checkpoint or not args.assets_manifest:
            parser.error("Extraction requires shard ownership, matching archives and encoder assets")
        run_extract(args, plan, config, identity, provenance)
    else:
        if not args.partial_roots or args.cache_roots or args.shards:
            parser.error("Merge-training accepts only compact partial feature inputs")
        merged = merge_partials(plan, args.partial_roots, args.output_dir, identity, provenance)
        training_start = time.monotonic()
        if training_start - args.orchestrator_start > HEAD_JOB_RESERVE_SECONDS - HEAD_FIT_BUDGET_SECONDS:
            raise RuntimeError("Head job preparation exceeded its 300-second reserve; merged features preserved")
        result = run_training(plan["labels"], plan["official"], args.output_dir / "features", identity, config, args.output_dir)
        write_json({"training_complete": True, "head_training_complete": True, "feature_bank_complete": True,
                    "encoder_frozen": True, "encoder_optimizer_updates": 0, "n_gold_excluded": len(plan["gold"]),
                    "pretraining_validation_merge_seconds": training_start - args.orchestrator_start,
                    "head_training_seconds": time.monotonic() - training_start,
                    "orchestrator_elapsed_seconds": time.monotonic() - args.orchestrator_start,
                    "head_fit_budget_seconds": HEAD_FIT_BUDGET_SECONDS,
                    "planned_head_job_reserve_seconds": HEAD_JOB_RESERVE_SECONDS,
                    "head_timer_boundary": "Core run_training 1200-second fit budget starts after global validation and feature merge",
                    "entire_job_runtime_guaranteed": False,
                    "provenance": provenance, "merged_bank": merged,
                    "checkpoint_receipts": {"head.pt": result["production_head"]["checkpoint_receipt"],
                                            "head_weak_holdout.pt": result["weak_holdout_head"]["checkpoint_receipt"]},
                    **result}, args.output_dir / "training_report.json")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        output = Path(sys.argv[sys.argv.index("--output-dir") + 1]) if "--output-dir" in sys.argv else Path("/kaggle/working/run")
        output.mkdir(parents=True, exist_ok=True)
        write_json({"training_complete": False, "error": str(error), "traceback": traceback.format_exc()}, output / "failure.json")
        raise

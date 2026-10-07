"""Offline Kaggle pilot / frozen extraction / fixed head training entry point."""
import argparse
import hashlib
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cache import MixedVolumeLoader, load_cache_mixed
from .data import LABELS, SliceStudyDataset, build_table, ids_hash, make_views, validated_labels
from .encoder import encode, file_sha, load_encoder
from .train import fit, load_features, run_training


def write_json(value, path):
    path = Path(path)
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def extraction_identity(provenance):
    return hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def feature_exists(path, uid, identity, k):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as value:
        if value["identity"].item() != identity or value["study"].item() != uid:
            raise RuntimeError(f"Existing features have different provenance: {uid}")
        f, m, p = value["features"], value["mask"], value["positions"]
        if f.shape != (6, k, 1024) or m.shape != (6,) or m.dtype != np.bool_ or p.shape != (6, k) or not m.any():
            raise RuntimeError(f"Existing features have invalid shape/mask: {uid}")
        if not np.isfinite(f).all() or not np.isfinite(p).all() or not np.all(f[~m] == 0):
            raise RuntimeError(f"Existing features are nonfinite or have nonzero missing slots: {uid}")
    return True


@torch.inference_mode()
def extract(model, ids, table, volume_loader, output, identity, config, start, max_seconds):
    features_dir = output / "features"
    features_dir.mkdir(exist_ok=True)
    missing = [uid for uid in ids if not feature_exists(features_dir / f"{uid}.npz", uid, identity, config["slices_per_slot"])]
    dataset = SliceStudyDataset(missing, table, volume_loader, config["slices_per_slot"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config["workers"],
                                         pin_memory=True, persistent_workers=False)
    images, newly_written, loop_start = 0, 0, time.monotonic()
    for uids, raw, masks, positions, selected in loader:
        if time.monotonic() - start > max_seconds:
            return {"complete": False, "reason": "extraction_walltime_budget", "newly_written": newly_written,
                    "images_encoded": images, "seconds": time.monotonic() - loop_start}
        uid = uids[0]
        mask, pos = masks[0], positions[0]
        flat = raw[0][mask].flatten(0, 1)
        encoded = []
        for group in flat.split(config["image_batch_size"]):
            views = make_views(group.to("cuda", non_blocking=True), config["resolution"], config["fov"])
            with torch.autocast("cuda", dtype=torch.float16):
                encoded.append(encode(model, views, config["extraction_variant"]).float().cpu())
        value = torch.cat(encoded).numpy().astype(np.float16)
        if not np.isfinite(value).all():
            raise RuntimeError("CLS values overflowed fp16 storage")
        features = np.zeros((6, config["slices_per_slot"], 1024), dtype=np.float16)
        features[mask.numpy()] = value.reshape(int(mask.sum()), config["slices_per_slot"], 1024)
        path = features_dir / f"{uid}.npz"
        with Path(str(path) + ".tmp").open("wb") as stream:
            np.savez_compressed(stream, features=features, mask=mask.numpy(), positions=pos.numpy(), study=uid,
                                identity=identity, selected_series=np.asarray([row[0] for row in selected]))
        Path(str(path) + ".tmp").replace(path)
        images += len(flat)
        newly_written += 1
        elapsed = time.monotonic() - loop_start
        progress = {"stage": "frozen_extraction", "selected_studies": len(ids),
                    "completed_studies": len(ids) - len(missing) + newly_written,
                    "images_encoded_this_call": images, "seconds_this_call": elapsed,
                    "images_per_second": images / max(elapsed, 1e-6),
                    "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "feature_identity": identity, "training_complete": False}
        write_json(progress, output / "progress.json")
        if newly_written % 10 == 0 or newly_written == len(missing):
            print(json.dumps(progress), flush=True)
    return {"complete": True, "newly_written": newly_written, "images_encoded": images,
            "seconds": time.monotonic() - loop_start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--config", type=Path, required=True, help="JSON produced from the checked-in YAML")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cache-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-fingerprints", type=Path, required=True)
    parser.add_argument("--preprocess-sha256", required=True)
    parser.add_argument("--dinov3-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/run"))
    parser.add_argument("--resume-features", type=Path)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; refuse expensive CPU extraction")
    config = json.loads(args.config.read_text())
    labels, official, gold = validated_labels(args.labels, args.data_dir)
    paths, counts, summaries, fingerprints = load_cache_mixed(args.cache_roots)
    if len(fingerprints) != 7 or fingerprints != json.loads(args.cache_fingerprints.read_text()):
        raise RuntimeError("Cache receipts differ from the seven independently verified fingerprints")
    if summaries[0]["preprocess_sha256"] != args.preprocess_sha256:
        raise RuntimeError("Physical cache preprocessing differs from the pinned published source")
    series = pd.read_csv(args.data_dir / "train_series.csv")
    if not series.SeriesInstanceUID.is_unique or set(series.SeriesInstanceUID) != set(paths):
        raise RuntimeError("Cache series coverage differs from all official training series")
    table = build_table(series, counts)
    ids = sorted(labels.index.tolist())
    if any(not re.fullmatch(r"[0-9.]+", uid) or not any(table.get(uid, [])) for uid in ids):
        raise RuntimeError("Invalid study UID or missing usable MRI series")
    provenance = {"experiment": config["experiment"], "checkpoint_sha256": config["checkpoint_sha256"],
                  "assets_manifest_sha256": file_sha(args.assets_manifest), "official_train_sha256": file_sha(args.data_dir / "train.csv"),
                  "official_series_sha256": file_sha(args.data_dir / "train_series.csv"), "labels_sha256": file_sha(args.labels),
                  "cache_fingerprints": fingerprints, "preprocessing_sha256": args.preprocess_sha256,
                  "all_study_ids_sha256": ids_hash(ids), "n_all_studies": len(ids), "n_gold": len(gold),
                  "labels_order": LABELS, "feature_config": {key: config[key] for key in
                      ("encoder", "feature_dim", "resolution", "slices_per_slot", "fov", "extraction_variant")},
                  "encoder_code_sha256": file_sha(Path(__file__).with_name("encoder.py")),
                  "sampling_code_sha256": file_sha(Path(__file__).with_name("data.py")),
                  "cache_loader_code_sha256": file_sha(Path(__file__).with_name("cache.py")),
                  "extraction_source_sha256": file_sha(__file__),
                  "features": "DINOv3-L normalized CLS, explicit author_no_rope or canonical variant; RGB-replicated single MRI slices",
                  "series_policy": "One deepest cached series per plane/fat-suppression slot, UID tie-break",
                  "gradient_policy": "Gold excluded; weak-fold0 head separated from final allweak head",
                  "external_pretraining_patient_overlap_verified": False,
                  "clean_patient_oof_claim": False,
                  "notes": "Report-grouped holdout measures teacher agreement. External pretraining patient overlap not independently established."}
    identity = extraction_identity(provenance)
    provenance["feature_identity"] = identity
    # Head-only ablations can reuse frozen images; audit their source separately.
    provenance["head_code_sha256"] = file_sha(Path(__file__).with_name("heads.py"))
    provenance["training_code_sha256"] = file_sha(Path(__file__).with_name("train.py"))
    write_json(provenance, output / "provenance.json")
    write_json(config, output / "config.json")
    prior_pilot = None
    if args.resume_features:
        prior_report = args.resume_features.parent / "pilot_report.json"
        if prior_report.is_file():
            prior_pilot = json.loads(prior_report.read_text())
            if prior_pilot.get("feature_identity") != identity:
                raise RuntimeError("Resume pilot report has a different extraction identity")
        target = output / "features"
        target.mkdir(exist_ok=True)
        for uid in ids:
            source = args.resume_features / f"{uid}.npz"
            if feature_exists(source, uid, identity, config["slices_per_slot"]):
                shutil.copyfile(source, target / source.name)
    print(json.dumps({"stage": args.stage, "config": config, "provenance": provenance,
                      "gpu": torch.cuda.get_device_name(0)}), flush=True)
    model = load_encoder(args.dinov3_repo, args.checkpoint, config["checkpoint_sha256"], config["checkpoint_bytes"])
    pilot_ids = sorted(labels.index[labels.is_gold.eq(0)])[:config["pilot_studies"]]
    volume_loader = MixedVolumeLoader(paths)
    pilot = extract(model, pilot_ids, table, volume_loader, output, identity, config, start, config["max_extract_seconds"])
    if not pilot["complete"]:
        raise RuntimeError("Pilot feature extraction did not finish")
    # Separate smoke head never becomes production initialization or final results.
    _, smoke = fit(pilot_ids, labels, output / "features", identity, config, "cuda", output / "smoke_head.pt", epochs=1, max_seconds=120)
    if smoke["optimizer_updates"] < 1:
        raise RuntimeError("No finite pilot gradient update")
    all_images = sum(sum(bool(choices) for choices in table[uid]) for uid in ids) * config["slices_per_slot"]
    if pilot["images_encoded"]:
        projected = all_images / (pilot["images_encoded"] / pilot["seconds"])
    elif prior_pilot:
        projected = prior_pilot["projected_full_extraction_seconds"]
    else:
        raise RuntimeError("Pilot features exist without a verified throughput report; cannot estimate GPU budget")
    pilot_report = {"pilot_only": args.stage == "pilot", "feature_identity": identity, "feature_extraction": pilot, "head_smoke": smoke,
                    "projected_full_extraction_seconds": projected, "total_planned_images": all_images,
                    "max_extract_seconds": config["max_extract_seconds"], "training_complete": False,
                    "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
    write_json(pilot_report, output / "pilot_report.json")
    print(json.dumps(pilot_report, indent=2), flush=True)
    if args.stage == "pilot":
        return
    if projected is not None and projected > config["max_extract_seconds"]:
        raise RuntimeError("Measured pilot projects extraction beyond budget; keep features and stop for explicit configuration change")
    full = extract(model, ids, table, volume_loader, output, identity, config, start, config["max_extract_seconds"])
    del model
    torch.cuda.empty_cache()
    if not full["complete"]:
        write_json({"training_complete": False, "feature_extraction": full, "reason": full["reason"]}, output / "training_report.json")
        raise RuntimeError("Frozen extraction stopped at budget; partial features preserved")
    for uid in ids:
        if not feature_exists(output / "features" / f"{uid}.npz", uid, identity, config["slices_per_slot"]):
            raise RuntimeError("Full feature coverage incomplete")
    result = run_training(labels, official, output / "features", identity, config, output)
    report = {"training_complete": True, "pilot_only": False, "feature_identity": identity,
              "feature_studies": len(ids), "encoder_frozen": True, "encoder_optimizer_updates": 0,
              "n_gold_excluded": len(gold), "feature_extraction": full, "total_seconds": time.monotonic() - start,
              "head_code_sha256": provenance["head_code_sha256"],
              "training_code_sha256": provenance["training_code_sha256"],
              "checkpoint_receipts": {
                  "head.pt": result["production_head"]["checkpoint_receipt"],
                  "head_weak_holdout.pt": result["weak_holdout_head"]["checkpoint_receipt"],
                  "smoke_head.pt": smoke["checkpoint_receipt"]},
              "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9, **result}
    write_json(report, output / "training_report.json")
    write_json({"training_complete": True, "completed_studies": len(ids), "total_seconds": report["total_seconds"]}, output / "progress.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Preserve evidence for failed/partial Kaggle jobs, never report success.
        import sys
        output = Path(sys.argv[sys.argv.index("--output-dir") + 1]) if "--output-dir" in sys.argv else Path("/kaggle/working/run")
        output.mkdir(parents=True, exist_ok=True)
        write_json({"training_complete": False, "error": str(error), "traceback": traceback.format_exc()}, output / "failure.json")
        raise

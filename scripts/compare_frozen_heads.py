#!/usr/bin/env python3
"""Controlled attention -> mean+max ablation on a completed frozen feature bank.

No MRI decode or encoder forward runs here. Input/output tables contain study
identifiers and belong in private Kaggle outputs or gitignored artifacts/ only.
The previous attention run must have completed all fixed epochs and all 4407
features. Only experiment and head_kind may change in the comparison config.
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orthofoundation.data import LABELS, ids_hash, validated_labels
from orthofoundation.encoder import file_sha
from orthofoundation.run import extraction_identity, feature_exists, write_json
from orthofoundation.train import load_features, run_training
import orthofoundation.run as extraction_module

FEATURE_KEYS = ("encoder", "feature_dim", "resolution", "slices_per_slot", "fov", "extraction_variant")
SOURCE_KEYS = {"encoder_code_sha256": "encoder.py", "sampling_code_sha256": "data.py",
               "cache_loader_code_sha256": "cache.py", "extraction_source_sha256": "run.py",
               "head_code_sha256": "heads.py", "training_code_sha256": "train.py"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected JSON object: {Path(path).name}")
    return value


def controlled_config(previous, comparison):
    """Refuse silent changes to image geometry, labels, split or optimization."""
    require(set(previous) == set(comparison), "Comparison config keys differ from the source config")
    require(previous.get("head_kind") == "attention", "Source run must be the original attention head")
    require(comparison.get("head_kind") == "meanmax", "Only the meanmax follow-up is supported")
    require(previous.get("experiment") != comparison.get("experiment")
            and isinstance(comparison.get("experiment"), str) and comparison["experiment"],
            "Comparison requires its own experiment name")
    changed = {key for key in previous if previous[key] != comparison[key]}
    require(changed == {"head_kind", "experiment"},
            "Only head_kind and experiment may change; changed keys: " + ", ".join(sorted(changed)))
    require(previous.get("head_epochs") == 12 and previous.get("seed") == 42
            and previous.get("weak_holdout_fold") == 0,
            "Source run must use the fixed 12 epochs, seed 42 and weak holdout fold 0")
    return dict(comparison)


def validate_provenance(provenance, previous_config, labels_path, data_dir, ids, gold):
    """Bind input tables and exact vendored code to the original feature identity."""
    require(provenance.get("experiment") == previous_config["experiment"], "Source experiment differs")
    require(provenance.get("feature_config") == {key: previous_config[key] for key in FEATURE_KEYS},
            "Source feature geometry differs from the saved config")
    require(provenance.get("checkpoint_sha256") == previous_config["checkpoint_sha256"],
            "Source encoder checkpoint SHA differs")
    require(provenance.get("labels_order") == LABELS, "Source label order differs")
    require(provenance.get("n_all_studies") == len(ids)
            and provenance.get("all_study_ids_sha256") == ids_hash(ids)
            and provenance.get("n_gold") == len(gold), "Source study/Gold manifest differs")
    for key, path in (("labels_sha256", labels_path), ("official_train_sha256", Path(data_dir) / "train.csv"),
                      ("official_series_sha256", Path(data_dir) / "train_series.csv")):
        require(provenance.get(key) == file_sha(path), f"Different input manifest: {key}")
    # run.py creates its identity before adding head/training source hashes.
    identity_input = {key: value for key, value in provenance.items()
                      if key not in ("feature_identity", "head_code_sha256", "training_code_sha256")}
    identity = extraction_identity(identity_input)
    require(provenance.get("feature_identity") == identity, "Source provenance/feature identity differs")
    source = Path(extraction_module.__file__).parent
    for key, filename in SOURCE_KEYS.items():
        require(provenance.get(key) == file_sha(source / filename), f"Current core source differs: {filename}")
    return identity


def validate_feature_bank(ids, directory, identity, k):
    """Validate every UID/shape/value with the original readers and hash the bank."""
    directory = Path(directory)
    files = list(directory.glob("*.npz"))
    require({path.stem for path in files} == set(ids) and len(files) == len(ids),
            "Feature file manifest differs from all official study IDs")
    digest, total_bytes = hashlib.sha256(), 0
    for uid in sorted(ids):
        path = directory / f"{uid}.npz"
        require(not path.is_symlink() and feature_exists(path, uid, identity, k), "Missing/linked feature file")
        size, sha = path.stat().st_size, file_sha(path)
        total_bytes += size
        digest.update(f"{path.name}\t{size}\t{sha}\n".encode())
    # load_features also checks normalized position bounds, which resume checks
    # alone do not assert. Keep validation RAM bounded to 64 studies.
    for start in range(0, len(ids), 64):
        load_features(ids[start:start + 64], directory, identity)
    return {"studies": len(ids), "bytes": total_bytes, "sha256": digest.hexdigest(),
            "digest_format": "sorted filename<TAB>bytes<TAB>sha256<LF>",
            "every_uid_geometry_identity_and_values_checked": True}


def validate_head_receipt(directory, name, receipt):
    path = Path(directory) / name
    require(receipt.get("filename") == name and path.is_file(), "Missing original head receipt/checkpoint")
    require(receipt.get("bytes") == path.stat().st_size and receipt.get("sha256") == file_sha(path),
            "Original head checkpoint differs from receipt: " + name)
    return dict(receipt)


def validate_baseline_report(report, provenance, config, labels, gold, directory):
    identity = provenance["feature_identity"]
    require(report.get("training_complete") is True and report.get("pilot_only") is False
            and report.get("encoder_frozen") is True and report.get("encoder_optimizer_updates") == 0,
            "Previous run did not complete frozen-encoder training")
    require(report.get("feature_identity") == identity and report.get("feature_studies") == 4407
            and report.get("n_gold_excluded") == 58, "Previous full feature/Gold coverage differs")
    weak = labels.index[labels.is_gold.eq(0)].tolist()
    holdout = labels.index[labels.is_gold.eq(0) & labels.fold.eq(config["weak_holdout_fold"])].tolist()
    gradient = [uid for uid in weak if uid not in set(holdout)]
    require(bool(holdout) and not set(gradient) & set(holdout), "Invalid weak holdout split")
    for field, ids, filename in (("production_head", weak, "head.pt"),
                                  ("weak_holdout_head", gradient, "head_weak_holdout.pt")):
        trained = report[field]
        require(trained.get("epochs") == config["head_epochs"]
                and trained.get("n_gradient_studies") == len(ids)
                and trained.get("gradient_ids_sha256") == ids_hash(ids)
                and trained.get("gold_used_for_gradients") == 0
                and trained.get("fixed_final_epoch") is True
                and trained.get("gold_checkpoint_selection") is False, "Previous fixed training/split differs: " + field)
        expected_updates = config["head_epochs"] * int(np.ceil(len(ids) / config["head_batch_size"]))
        require(trained.get("optimizer_updates") == expected_updates, "Previous optimizer update count differs")
        for key in ("head_code_sha256", "training_code_sha256"):
            require(trained.get(key) == report.get(key) == provenance.get(key), "Previous code fingerprint differs")
        receipt = validate_head_receipt(directory, filename, trained["checkpoint_receipt"])
        require(report["checkpoint_receipts"].get(filename) == receipt, "Original exported checkpoint receipt differs")
    require(report["weak_holdout_metrics"].get("n") == len(holdout)
            and report["gold_metrics"].get("n") == len(gold), "Previous metric population differs")
    return {"weak_holdout_metrics": report["weak_holdout_metrics"], "gold_metrics": report["gold_metrics"],
            "production_head": report["production_head"], "weak_holdout_head": report["weak_holdout_head"],
            "checkpoint_receipts": {name: report["checkpoint_receipts"][name]
                                    for name in ("head.pt", "head_weak_holdout.pt")}}


def run_comparison(previous_run_dir, labels_path, data_dir, output_dir, comparison_config, device="cuda", assets_manifest=None):
    previous_run_dir, output = Path(previous_run_dir), Path(output_dir)
    require(output.resolve() != previous_run_dir.resolve()
            and previous_run_dir.resolve() not in output.resolve().parents
            and output.resolve() not in previous_run_dir.resolve().parents,
            "Comparison output must be separate from the immutable previous run")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Comparison output must be empty")
    if device == "cuda":
        import torch
        require(torch.cuda.is_available(), "CUDA unavailable; choose --device cpu explicitly for tests")
    start = time.monotonic()
    previous_config = read_json(previous_run_dir / "config.json")
    config = controlled_config(previous_config, comparison_config)
    provenance = read_json(previous_run_dir / "provenance.json")
    previous_report = read_json(previous_run_dir / "training_report.json")
    labels, official, gold = validated_labels(labels_path, data_dir)
    ids = sorted(labels.index.tolist())
    require(len(ids) == 4407 and len(gold) == 58 and int(labels.is_gold.eq(0).sum()) == 4349,
            "Expected all 4407 studies, 4349 weak studies and 58 excluded Gold studies")
    identity = validate_provenance(provenance, previous_config, labels_path, data_dir, ids, gold)
    if assets_manifest:
        require(file_sha(assets_manifest) == provenance["assets_manifest_sha256"], "Different input assets manifest")
    baseline = validate_baseline_report(previous_report, provenance, previous_config, labels, gold, previous_run_dir)
    bank = validate_feature_bank(ids, previous_run_dir / "features", identity, config["slices_per_slot"])
    output.mkdir(parents=True, exist_ok=True)
    source = Path(extraction_module.__file__).parent
    source_hashes = {"orthofoundation/" + path.name: file_sha(path) for path in sorted(source.glob("*.py"))}
    source_hashes["compare_frozen_heads.py"] = file_sha(__file__)
    comparison_provenance = {"experiment": config["experiment"], "comparison": "attention versus mean+max",
                             "feature_identity": identity, "source_experiment": previous_config["experiment"],
                             "source_provenance_sha256": file_sha(previous_run_dir / "provenance.json"),
                             "source_training_report_sha256": file_sha(previous_run_dir / "training_report.json"),
                             "source_config_sha256": file_sha(previous_run_dir / "config.json"),
                             "source_feature_config": provenance["feature_config"], "feature_bank_receipt": bank,
                             "source_encoder_checkpoint": {"sha256": previous_config["checkpoint_sha256"],
                                                           "bytes": previous_config["checkpoint_bytes"]},
                             "source_sha256": source_hashes, "encoder_frozen": True, "encoder_forward_calls": 0,
                             "encoder_optimizer_updates": 0, "changed_config_keys": ["experiment", "head_kind"],
                             "gold_checkpoint_selection": False, "head_budget_seconds": 1200,
                             "clean_patient_oof_claim": False, "external_pretraining_patient_overlap_verified": False}
    write_json(comparison_provenance, output / "provenance.json")
    write_json(config, output / "config.json")
    print(json.dumps({"stage": "head_comparison", "experiment": config["experiment"],
                      "feature_identity": identity, "feature_studies": bank["studies"],
                      "encoder_forward_calls": 0, "validation_seconds": time.monotonic() - start}), flush=True)
    result = run_training(labels, official, previous_run_dir / "features", identity, config, output, device=device)
    report = {"training_complete": True, "pilot_only": False, "experiment": config["experiment"],
              "head_kind": config["head_kind"], "feature_identity": identity, "feature_studies": len(ids),
              "encoder_frozen": True, "encoder_forward_calls": 0, "encoder_optimizer_updates": 0,
              "n_gold_excluded": len(gold), "head_code_sha256": provenance["head_code_sha256"],
              "training_code_sha256": provenance["training_code_sha256"],
              "source_sha256": source_hashes, "feature_bank_receipt": bank,
              "baseline_attention": baseline, "total_seconds": time.monotonic() - start,
              "checkpoint_receipts": {"head.pt": result["production_head"]["checkpoint_receipt"],
                                      "head_weak_holdout.pt": result["weak_holdout_head"]["checkpoint_receipt"]},
              "metric_interpretation": "Weak holdout is teacher agreement; Gold is diagnostic only. No Gold checkpoint selection or clean patient OOF claim.",
              **result}
    # Preserve the baseline's reported metrics alongside this independent fit.
    report["macro_auc_delta_meanmax_minus_attention"] = {
        key: (result[key]["macro_auc"] - baseline[key]["macro_auc"]
              if result[key]["macro_auc"] is not None and baseline[key]["macro_auc"] is not None else None)
        for key in ("weak_holdout_metrics", "gold_metrics")}
    write_json(report, output / "training_report.json")
    write_json({"training_complete": True, "feature_studies": len(ids), "total_seconds": report["total_seconds"]},
               output / "progress.json")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orthofoundation_meanmax.yaml")
    parser.add_argument("--assets-manifest", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    failure_output_allowed = not args.output_dir.exists() or (args.output_dir.is_dir() and not any(args.output_dir.iterdir()))
    try:
        run_comparison(args.previous_run_dir, args.labels, args.data_dir, args.output_dir,
                       config, args.device, args.assets_manifest)
    except Exception as error:
        # A rejected overlapping output path must never modify the source run.
        previous, output = args.previous_run_dir.resolve(), args.output_dir.resolve()
        if failure_output_allowed and previous != output and previous not in output.parents and output not in previous.parents:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json({"training_complete": False, "error": str(error), "traceback": traceback.format_exc()},
                       args.output_dir / "failure.json")
        raise


if __name__ == "__main__":
    main()

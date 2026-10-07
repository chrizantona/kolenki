#!/usr/bin/env python3
"""EXP-OF-002 on a complete native sharded bank; no MRI/encoder execution.

This validates the sharded schema directly. Generic NPZ/value/hash and head
receipt helpers are reused without inventing monolithic provenance fields.
Production metadata validation always retains all 4,407 studies and seven pins.
Root must independently seal the complete immutable attention-source bank SHA
before preparing an executable comparison. This script never derives its own
expected pin. Both scored populations also replay their attention checkpoints.
"""
import os
import time
_MODULE_START = float(os.environ.get("RSNA_SHARDED_COMPARE_BOOTSTRAP_START", time.monotonic()))

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

import sharded_frozen_extract as RUNNER
from compare_frozen_heads import controlled_config, read_json, require, validate_feature_bank, validate_head_receipt
from orthofoundation.data import LABELS, ids_hash, slice_indices
from orthofoundation.encoder import file_sha
from orthofoundation.heads import StudyHead
from orthofoundation.run import write_json
from orthofoundation.train import load_features, run_training, score_auc

REPLAY_TOLERANCES = {"cpu": {"atol": 2e-6, "rtol": 2e-5}, "cuda": {"atol": 1e-6, "rtol": 1e-5}}


def require_source_bank_pin(pin):
    require(isinstance(pin, str) and re.fullmatch(r"[0-9a-f]{64}", pin) is not None,
            "Missing/invalid external full attention source-bank SHA256 pin; root must seal completed immutable output first")


def separate_output(previous, global_root, output):
    for source in (Path(previous).resolve(), Path(global_root).resolve()):
        target = Path(output).resolve()
        require(target != source and source not in target.parents and target not in source.parents,
                "Comparison output must be separate from immutable source/global metadata")
    require(not Path(output).exists() or (Path(output).is_dir() and not any(Path(output).iterdir())),
            "Comparison output must be empty")


def check_elapsed(start, limit):
    require(time.monotonic() - start < limit, "Comparison Python-stage walltime budget exhausted")


def split_ids(plan, config):
    labels = plan["labels"]
    weak = labels.index[labels.is_gold.eq(0)].tolist()
    gold = labels.index[labels.is_gold.eq(1)].tolist()
    holdout = labels.index[labels.is_gold.eq(0) & labels.fold.eq(config["weak_holdout_fold"])].tolist()
    gradient = [uid for uid in weak if uid not in set(holdout)]
    require(bool(gradient) and bool(holdout) and not set(weak) & set(gold), "Invalid weak/Gold split")
    return weak, gold, holdout, gradient


def validate_sharded_bank(plan, directory, identity, provenance):
    """Merged 'mask' is the global mask; partial ownership fields are absent."""
    directory = Path(directory)
    keys = {"features", "mask", "positions", "study", "identity", "selected_series",
            "global_metadata_manifest_sha256", "orchestrator_sha256"}
    for uid in plan["ids"]:
        path = directory / f"{uid}.npz"
        require(path.is_file() and not path.is_symlink(), "Missing/linked merged feature")
        with np.load(path, allow_pickle=False) as row:
            require(set(row.files) == keys, "Merged NPZ fields differ; partial banks are unsupported")
            require(row["study"].item() == uid and row["identity"].item() == identity, "Merged UID/identity differs")
            for key in ("global_metadata_manifest_sha256", "orchestrator_sha256"):
                require(row[key].item() == provenance[key], "Merged NPZ source provenance differs")
            features, mask, positions = row["features"], row["mask"], row["positions"]
            require(features.shape == (6, 4, 1024) and features.dtype == np.float16
                    and mask.shape == (6,) and mask.dtype == np.bool_ and positions.shape == (6, 4)
                    and positions.dtype == np.float32, "Merged feature geometry/dtype differs")
            require(np.array_equal(mask, plan["masks"][uid]) and mask.any(), "Merged global mask differs")
            require(np.isfinite(features).all() and np.isfinite(positions).all()
                    and np.all(features[~mask] == 0) and np.all(positions[~mask] == 0), "Merged feature values/padding differ")
            require(np.array_equal(row["selected_series"], np.asarray(plan["selected"][uid])), "Merged selected series differs")
            for slot in np.flatnonzero(mask):
                sr = plan["selected"][uid][slot]
                expected = slice_indices(plan["counts"][sr], 4).astype(np.float32) / (plan["counts"][sr] - 1)
                require(np.array_equal(positions[slot], expected), "Merged slice positions differ from global selection")
    bank = validate_feature_bank(plan["ids"], directory, identity, 4)
    bank.update(selected_slot_series=plan["summary"]["selected_series"], images_k4=plan["summary"]["images"],
                global_masks_series_positions_and_native_sharded_provenance_checked=True,
                receipt_origin="Current full bank measured for comparison with independently sealed attention-source bank SHA256")
    return bank


def validate_ids_csv(path, expected):
    table = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    require(list(table.columns) == ["StudyInstanceUID"] and table.StudyInstanceUID.tolist() == expected
            and table.StudyInstanceUID.is_unique, "Saved gradient/holdout/Gold ID order or membership differs")


def validate_predictions(path, ids, targets, saved, target_kind):
    table = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    require(list(table.columns) == ["StudyInstanceUID", *LABELS] and table.StudyInstanceUID.tolist() == ids
            and table.StudyInstanceUID.is_unique, "Saved prediction IDs/label order differ")
    predictions = table[LABELS].to_numpy()
    require(np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all(), "Invalid saved probabilities")
    require(list(saved.get("per_class_auc", {})) == LABELS, "Saved per-label metric order/coverage differs")
    actual = score_auc(targets, predictions, target_kind)
    for key in ("n", "target_kind", "classes_with_both_labels"):
        require(actual[key] == saved.get(key), "Saved metric population/interpretation differs")
    for key in ["macro_auc", *LABELS]:
        a = actual["macro_auc"] if key == "macro_auc" else actual["per_class_auc"][key]
        b = saved.get("macro_auc") if key == "macro_auc" else saved.get("per_class_auc", {}).get(key)
        require((a is None and b is None) or (a is not None and b is not None and np.isclose(a, b, rtol=0, atol=1e-12)),
                "Saved metric differs from prediction CSV")


@torch.inference_mode()
def replay_attention_predictions(head, path, ids, feature_dir, identity, config, device):
    """Source exports probabilities, not logits: compare sigmoid FP32 outputs."""
    require(device in REPLAY_TOLERANCES, "Unsupported explicit attention replay device")
    if device == "cuda": require(torch.cuda.is_available(), "CUDA unavailable for attention replay")
    table = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    require(list(table.columns) == ["StudyInstanceUID", *LABELS] and table.StudyInstanceUID.tolist() == ids
            and table.StudyInstanceUID.is_unique, "Attention replay source CSV IDs/labels differ")
    saved = table[LABELS].to_numpy(dtype=np.float64)
    require(np.isfinite(saved).all() and ((saved >= 0) & (saved <= 1)).all(), "Attention source CSV probabilities are invalid")
    head = head.to(device=device, dtype=torch.float32).eval()
    logits_rows, probability_rows = [], []
    started = time.monotonic()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for begin in range(0, len(ids), config["head_batch_size"]):
            batch_ids = ids[begin:begin + config["head_batch_size"]]
            tensors = [value.to(device) for value in load_features(batch_ids, feature_dir, identity)]
            with torch.autocast(device_type=device, enabled=False):
                logits = head(*tensors)
                probabilities = logits.sigmoid()
            require(logits.dtype == torch.float32 and torch.isfinite(logits).all().item()
                    and probabilities.dtype == torch.float32 and torch.isfinite(probabilities).all().item(),
                    "Attention replay logits/probabilities are nonfinite or not FP32")
            logits_rows.append(logits.cpu().numpy()); probability_rows.append(probabilities.cpu().numpy())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    logits = np.concatenate(logits_rows); probabilities = np.concatenate(probability_rows)
    tolerance = REPLAY_TOLERANCES[device]
    absolute = np.abs(probabilities.astype(np.float64) - saved)
    require(np.allclose(probabilities, saved, **tolerance), "Attention checkpoint prediction replay disagrees with original CSV")
    receipt = {"n": len(ids), "ids_sha256": ids_hash(ids), "labels_order": LABELS, "device": device,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU", "torch_version": str(torch.__version__),
        "dtype": "float32", "autocast": False, "cuda_matmul_tf32_during_replay": False,
        "probability_tolerances": dict(tolerance), "tolerance_rule": "abs(replay-source) <= atol + rtol*abs(source)",
        "comparison_space": "Sigmoid probabilities against original exported CSV; source raw logits were not exported",
        "logits_and_probabilities_finite": True, "max_absolute_probability_difference": float(absolute.max()),
        "replay_probability_sha256": hashlib.sha256(probabilities.tobytes(order="C")).hexdigest(),
        "replay_logits_sha256": hashlib.sha256(logits.tobytes(order="C")).hexdigest(),
        "array_digest_format": "FP32 native little-endian row-major values; original CSV row/label order",
        "source_prediction_csv_sha256": file_sha(path), "seconds": time.monotonic() - started}
    return receipt


def validate_sharded_baseline(plan, directory, config, identity, provenance, device="cpu"):
    directory = Path(directory)
    require((directory / "provenance.json").read_bytes() == json.dumps(provenance, indent=2).encode(),
            "Saved sharded provenance bytes differ from reconstructed global identity")
    report = read_json(directory / "training_report.json")
    require(all(report.get(key) is True for key in ("training_complete", "head_training_complete", "feature_bank_complete", "encoder_frozen"))
            and report.get("encoder_optimizer_updates") == 0 and report.get("n_gold_excluded") == len(plan["gold"])
            and report.get("provenance") == provenance, "Previous sharded frozen training/provenance incomplete")
    merged = read_json(directory / "merge_report.json")
    require(report.get("merged_bank") == merged and merged.get("feature_bank_complete") is True
            and merged.get("head_training_complete") is False and merged.get("feature_identity") == identity
            and merged.get("n_studies") == plan["summary"]["studies"]
            and merged.get("selected_slot_series") == plan["summary"]["selected_series"]
            and merged.get("images_k4") == plan["summary"]["images"] and merged.get("shards") == list(range(7))
            and type(merged.get("partial_jobs")) is int and 1 <= merged["partial_jobs"] <= 7,
            "Previous bank is not a complete all-seven global merge")
    require(report.get("head_fit_budget_seconds") == 1200 and report.get("planned_head_job_reserve_seconds") == 1500,
            "Previous head budgets differ")
    weak, gold, holdout, gradient = split_ids(plan, config)
    for filename, ids in (("gradient_train_ids.csv", weak), ("excluded_gold_ids.csv", gold),
                          ("weak_holdout_ids.csv", holdout), ("weak_holdout_gradient_ids.csv", gradient)):
        validate_ids_csv(directory / filename, ids)
    replay_receipts = {}
    for key, ids, filename in (("production_head", weak, "head.pt"), ("weak_holdout_head", gradient, "head_weak_holdout.pt")):
        trained = report[key]
        steps = int(np.ceil(len(ids) / config["head_batch_size"]))
        require(trained.get("epochs") == 12 and trained.get("optimizer_updates") == 12 * steps
                and trained.get("n_gradient_studies") == len(ids) and trained.get("gradient_ids_sha256") == ids_hash(ids)
                and trained.get("gold_used_for_gradients") == 0 and trained.get("fixed_final_epoch") is True
                and trained.get("gold_checkpoint_selection") is False, "Previous fixed epochs/updates/Gold gradient policy differs")
        history = trained.get("history", [])
        require(len(history) == 12 and all(row["epoch"] == index + 1 and row["updates"] == (index + 1) * steps
                and np.isfinite(row["loss"]) for index, row in enumerate(history)), "Previous epoch/update history differs")
        for field, module in (("head_code_sha256", "heads.py"), ("training_code_sha256", "train.py")):
            require(trained.get(field) == provenance["core_source_sha256"][module], "Previous head/training source differs")
        receipt = validate_head_receipt(directory, filename, trained["checkpoint_receipt"])
        require(report["checkpoint_receipts"].get(filename) == receipt, "Previous exported head receipt differs")
        saved = torch.load(directory / filename, map_location="cpu", weights_only=True)
        require(saved.get("labels") == LABELS and saved.get("feature_identity") == identity and saved.get("config") == config
                and saved.get("training_report") == {key: value for key, value in trained.items() if key != "checkpoint_receipt"},
                "Previous checkpoint label/config/identity/training metadata differs")
        head = StudyHead(dim=config["head_dim"], kind="attention", dropout=config["head_dropout"])
        head.load_state_dict(saved["model"], strict=True)
        require(all(torch.isfinite(value).all().item() for value in saved["model"].values()), "Previous head contains nonfinite state")
        scored_ids, csv_name = (gold, "gold_predictions.csv") if key == "production_head" else (holdout, "weak_holdout_predictions.csv")
        replay_receipts[key] = replay_attention_predictions(head, directory / csv_name, scored_ids, directory / "features", identity, config, device)
        replay_receipts[key]["source_checkpoint_receipt"] = receipt
    validate_predictions(directory / "weak_holdout_predictions.csv", holdout,
        (plan["labels"].loc[holdout, LABELS].to_numpy() >= .5).astype(int), report["weak_holdout_metrics"],
        "thresholded_public_teacher_consensus; not expert ground truth")
    validate_predictions(directory / "gold_predictions.csv", gold, plan["official"].loc[gold, LABELS].to_numpy(dtype=int),
        report["gold_metrics"], "official_expert_labels; diagnostic only")
    return {"attention_prediction_replay": replay_receipts,
            **{key: report[key] for key in ("weak_holdout_metrics", "gold_metrics", "weak_holdout_head", "production_head", "checkpoint_receipts")}}


def run_comparison(previous, global_root, global_manifest_sha, assets_manifest_sha, expected_identity,
                   output, comparison_config, device="cuda", start=None, max_seconds=1500, expected_bank_sha256=None):
    previous, global_root, output = Path(previous), Path(global_root), Path(output)
    separate_output(previous, global_root, output)
    require_source_bank_pin(expected_bank_sha256)
    require(max_seconds == 1500, "Sharded comparison retains a 1500s job and 1200s head-fit budget")
    if device == "cuda": require(torch.cuda.is_available(), "CUDA unavailable; choose CPU explicitly")
    start = time.monotonic() if start is None else start
    previous_config = read_json(previous / "config.json")
    config = controlled_config(previous_config, comparison_config)
    plan = RUNNER.validate_global(global_root, global_manifest_sha)
    identity, provenance = RUNNER.make_identity(plan, previous_config, assets_manifest_sha)
    require(identity == expected_identity, "Reconstructed native sharded identity differs from pinned source bank")
    bank = validate_sharded_bank(plan, previous / "features", identity, provenance)
    require(bank["sha256"] == expected_bank_sha256, "Full attention source-bank SHA256 differs from independently sealed pin")
    bank["expected_sha256"] = expected_bank_sha256
    bank["pin_origin"] = "Explicit trusted root pin sealed from completed immutable attention output BEFORE comparison; never inferred from current bank"
    baseline = validate_sharded_baseline(plan, previous, previous_config, identity, provenance, device)
    validation_seconds = time.monotonic() - start
    require(validation_seconds <= 300, "Comparison validation exceeded its 300-second reserve; no fitting")
    check_elapsed(start, max_seconds)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {"orthofoundation/" + name: value for name, value in provenance["core_source_sha256"].items()}
    for name in ("sharded_frozen_extract.py", "compare_frozen_heads.py", "compare_sharded_heads.py"):
        source_hashes[name] = file_sha(Path(__file__).with_name(name))
    comparison_provenance = {"schema": "sharded_head_comparison_v1", "experiment": config["experiment"], "feature_identity": identity,
        "source_sharded_provenance": provenance, "source_provenance_sha256": file_sha(previous / "provenance.json"),
        "source_training_report_sha256": file_sha(previous / "training_report.json"), "source_merge_report_sha256": file_sha(previous / "merge_report.json"),
        "source_config_sha256": file_sha(previous / "config.json"), "feature_bank_receipt": bank, "source_sha256": source_hashes,
        "attention_prediction_replay": baseline["attention_prediction_replay"],
        "encoder_forward_calls": 0, "encoder_optimizer_updates": 0, "changed_config_keys": ["experiment", "head_kind"],
        "head_fit_budget_seconds": 1200, "job_budget_seconds": 1500, "validation_reserve_seconds": 300,
        "seed_caveat": "Same seed does not guarantee identical minibatch order: different head initialization consumes RNG differently.",
        "clean_patient_oof_claim": False, "external_pretraining_patient_overlap_verified": False}
    write_json(comparison_provenance, output / "provenance.json"); write_json(config, output / "config.json")
    fit_start = time.monotonic()
    result = run_training(plan["labels"], plan["official"], previous / "features", identity, config, output, device=device)
    check_elapsed(start, max_seconds)
    delta = lambda a, b: a - b if a is not None and b is not None else None
    report = {"training_complete": True, "head_training_complete": True, "experiment": config["experiment"], "head_kind": "meanmax",
        "feature_identity": identity, "feature_studies": len(plan["ids"]), "n_gold_excluded": len(plan["gold"]),
        "encoder_frozen": True, "encoder_forward_calls": 0, "encoder_optimizer_updates": 0, "baseline_attention": baseline,
        "feature_bank_receipt": bank, "source_sha256": source_hashes, "provenance": comparison_provenance,
        "validation_seconds": validation_seconds, "head_training_seconds": time.monotonic() - fit_start, "total_seconds": time.monotonic() - start,
        "checkpoint_receipts": {"head.pt": result["production_head"]["checkpoint_receipt"], "head_weak_holdout.pt": result["weak_holdout_head"]["checkpoint_receipt"]},
        "metric_interpretation": "Weak holdout measures teacher agreement; Gold is diagnostic only, excluded from gradients and checkpoint selection.",
        "macro_auc_delta_meanmax_minus_attention": {key: delta(result[key]["macro_auc"], baseline[key]["macro_auc"]) for key in ("weak_holdout_metrics", "gold_metrics")},
        "per_label_auc_delta_meanmax_minus_attention": {key: {label: delta(result[key]["per_class_auc"][label], baseline[key]["per_class_auc"][label]) for label in LABELS}
            for key in ("weak_holdout_metrics", "gold_metrics")}, **result}
    write_json(report, output / "training_report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run-dir", type=Path, required=True)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--global-manifest-sha256", required=True)
    parser.add_argument("--assets-manifest-sha256", required=True, help="Pinned string only; no encoder assets mounted")
    parser.add_argument("--expected-feature-identity", required=True)
    parser.add_argument("--source-bank-sha256", required=True, help="External full-bank pin sealed by root before comparison; never current-bank auto-hash")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orthofoundation_meanmax.yaml")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-seconds", type=int, default=1500)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    failure_output_allowed = False
    try:
        separate_output(args.previous_run_dir, args.global_root, args.output_dir)
        failure_output_allowed = True
        report = run_comparison(args.previous_run_dir, args.global_root, args.global_manifest_sha256, args.assets_manifest_sha256,
            args.expected_feature_identity, args.output_dir, config, args.device, _MODULE_START, args.max_seconds, args.source_bank_sha256)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        if failure_output_allowed:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_json({"training_complete": False, "error": str(error), "traceback": traceback.format_exc()}, args.output_dir / "failure.json")
        raise


if __name__ == "__main__":
    main()

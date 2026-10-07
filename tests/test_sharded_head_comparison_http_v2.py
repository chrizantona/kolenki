"""The native-sharded adapter must reject incompatible banks before fitting."""
import ast
import base64
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import nbformat
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_sharded_heads_http_v2 as COMPARE
import build_sharded_head_comparison_notebook_http_v2 as BUILDER
from test_sharded_fallback import fixture_plan, partial_roots
from orthofoundation.run import write_json
from orthofoundation.train import run_training


class ShardedHeadComparisonHTTPV2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, base):
        plan, _, _ = fixture_plan()
        original = yaml.safe_load((ROOT / "configs/orthofoundation_frozen_http_v2.yaml").read_text())
        original.update(head_dim=16, head_dropout=0.)
        comparison = dict(original, head_kind="meanmax", experiment="synthetic_meanmax_comparison")
        identity, provenance = COMPARE.RUNNER.make_identity(plan, original, "a" * 64)
        previous = base / "attention"
        roots = partial_roots(base, plan, identity, provenance)
        merged = COMPARE.RUNNER.merge_partials(plan, roots, previous, identity, provenance)
        write_json(original, previous / "config.json"); write_json(provenance, previous / "provenance.json")
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_training(plan["labels"], plan["official"], previous / "features", identity, original, previous, "cpu")
        report = {"training_complete": True, "head_training_complete": True, "feature_bank_complete": True,
                  "encoder_frozen": True, "encoder_optimizer_updates": 0, "n_gold_excluded": 1,
                  "head_fit_budget_seconds": 1200, "planned_head_job_reserve_seconds": 1500,
                  "provenance": provenance, "merged_bank": merged,
                  "checkpoint_receipts": {"head.pt": result["production_head"]["checkpoint_receipt"],
                                          "head_weak_holdout.pt": result["weak_holdout_head"]["checkpoint_receipt"]}, **result}
        write_json(report, previous / "training_report.json")
        global_root = base / "global"; global_root.mkdir()
        return plan, original, comparison, identity, provenance, previous, global_root, report

    def test_complete_native_bank_trains_meanmax_with_same_split_and_exact_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan, _, config, identity, provenance, previous, global_root, baseline = self.fixture(base)
            sealed_pin = COMPARE.validate_sharded_bank(plan, previous / "features", identity, provenance)["sha256"]
            source_bytes = {path.relative_to(previous): path.read_bytes() for path in previous.rglob("*") if path.is_file()}
            with patch.object(COMPARE.RUNNER, "validate_global", return_value=plan), \
                    patch.object(COMPARE.RUNNER, "extract", side_effect=AssertionError("Encoder extraction forbidden")), \
                    patch.object(COMPARE.RUNNER, "load_encoder", side_effect=AssertionError("Encoder loading forbidden")), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = COMPARE.run_comparison(previous, global_root, "synthetic_global_sha", "a" * 64, identity,
                    base / "meanmax", config, device="cpu", expected_bank_sha256=sealed_pin)
            self.assertEqual(report["encoder_forward_calls"], 0)
            self.assertEqual(report["source_sha256"]["orthofoundation_http_v2/sharded_frozen_extract.py"],
                             hashlib.sha256(Path(COMPARE.RUNNER.__file__).read_bytes()).hexdigest())
            self.assertEqual(report["source_sha256"]["compare_sharded_heads.py"],
                             hashlib.sha256(Path(COMPARE.__file__).read_bytes()).hexdigest())
            self.assertNotIn("sharded_frozen_extract.py", report["source_sha256"])
            self.assertTrue(report["head_training_complete"])
            self.assertEqual(report["production_head"]["optimizer_updates"], 12)
            self.assertEqual(report["weak_holdout_head"]["optimizer_updates"], 12)
            self.assertEqual(report["production_head"]["gold_used_for_gradients"], 0)
            self.assertFalse(report["production_head"]["gold_checkpoint_selection"])
            self.assertEqual(report["baseline_attention"]["production_head"], baseline["production_head"])
            self.assertEqual(report["feature_bank_receipt"]["images_k4"], 16)
            self.assertEqual(report["feature_bank_receipt"]["expected_sha256"], sealed_pin)
            for receipt in report["baseline_attention"]["attention_prediction_replay"].values():
                self.assertEqual(receipt["device"], "cpu")
                self.assertTrue(receipt["logits_and_probabilities_finite"])
                self.assertEqual(receipt["probability_tolerances"], COMPARE.REPLAY_TOLERANCES["cpu"])
                self.assertLess(receipt["max_absolute_probability_difference"], 2e-6)
            self.assertEqual(set(report["per_label_auc_delta_meanmax_minus_attention"]["gold_metrics"]), set(COMPARE.LABELS))
            self.assertIn("different head initialization", report["provenance"]["seed_caveat"])
            gradient = pd.read_csv(base / "meanmax/gradient_train_ids.csv", dtype=str).StudyInstanceUID.tolist()
            self.assertNotIn("1.1.3", gradient)
            for path, original in source_bytes.items():self.assertEqual((previous / path).read_bytes(), original)

    def test_incomplete_merge_wrong_source_updates_or_gold_ids_reject_before_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan, original, config, identity, provenance, previous, global_root, report = self.fixture(base)
            sealed_pin = COMPARE.validate_sharded_bank(plan, previous / "features", identity, provenance)["sha256"]
            changes = [lambda row:row.update(head_training_complete=False),
                       lambda row:row["merged_bank"].update(shards=[0]),
                       lambda row:row["production_head"].update(optimizer_updates=11),
                       lambda row:row["weak_holdout_head"].update(training_code_sha256="wrong")]
            for mutate in changes:
                with self.subTest(mutate=mutate):
                    changed = copy.deepcopy(report); mutate(changed); write_json(changed, previous / "training_report.json")
                    with patch.object(COMPARE.RUNNER, "validate_global", return_value=plan), patch.object(COMPARE, "run_training") as fit:
                        with self.assertRaises(RuntimeError):
                            COMPARE.run_comparison(previous, global_root, "synthetic_global_sha", "a" * 64, identity,
                                base / "rejected", config, "cpu", expected_bank_sha256=sealed_pin)
                        fit.assert_not_called()
                    self.assertFalse((base / "rejected").exists())
            write_json(report, previous / "training_report.json")
            pd.Series(["1.1.1", "1.1.3"], name="StudyInstanceUID").to_csv(previous / "gradient_train_ids.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "ID order or membership"):
                COMPARE.validate_sharded_baseline(plan, previous, original, identity, provenance)

    def test_wrong_identity_global_positions_mask_and_provenance_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan, original, config, identity, provenance, previous, global_root, _ = self.fixture(base)
            sealed_pin = COMPARE.validate_sharded_bank(plan, previous / "features", identity, provenance)["sha256"]
            with patch.object(COMPARE.RUNNER, "validate_global", return_value=plan), patch.object(COMPARE, "run_training") as fit:
                with self.assertRaisesRegex(RuntimeError, "pinned source bank"):
                    COMPARE.run_comparison(previous, global_root, "synthetic_global_sha", "a" * 64, "wrong",
                        base / "rejected", config, "cpu", expected_bank_sha256=sealed_pin)
                fit.assert_not_called()
            path = previous / "features/1.1.1.npz"
            with np.load(path, allow_pickle=False) as row: content = {key:row[key].copy() for key in row.files}
            for mutation, message in ((lambda row:row["positions"].__setitem__((0, 0), .8), "slice positions"),
                                      (lambda row:row["mask"].__setitem__(0, False), "global mask"),
                                      (lambda row:row.__setitem__("orchestrator_sha256", np.asarray("wrong")), "source provenance")):
                with self.subTest(message=message):
                    changed = copy.deepcopy(content); mutation(changed); np.savez_compressed(path, **changed)
                    with self.assertRaisesRegex(RuntimeError, message):
                        COMPARE.validate_sharded_bank(plan, previous / "features", identity, provenance)
            write_json(dict(provenance, all_cache_fingerprints=[]), previous / "provenance.json")
            with self.assertRaisesRegex(RuntimeError, "provenance bytes"):
                COMPARE.validate_sharded_baseline(plan, previous, original, identity, provenance)

    def test_builder_only_mounts_future_complete_bank_and_small_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan, _, _ = fixture_plan()
            global_root = base / "global"; global_root.mkdir()
            (global_root / "global_metadata_manifest.json").write_text("{}")
            docker = base / "docker.json"; docker.write_text(json.dumps({"docker_image": "pinned_image"}))
            for device in ("cpu", "cuda"):
                directory = base / device
                argv = ["builder", "--global-root", str(global_root), "--assets-manifest-sha256", "a" * 64,
                        "--docker-metadata", str(docker), "--device", device, "--output-dir", str(directory), "--prepare-placeholder"]
                with patch.object(sys, "argv", argv), patch.object(BUILDER.RUNNER, "validate_global", return_value=plan), \
                        contextlib.redirect_stdout(io.StringIO()): BUILDER.main()
                metadata = json.loads((directory / "kernel-metadata.json").read_text())
                self.assertTrue(metadata["is_private"]); self.assertFalse(metadata["enable_internet"])
                self.assertEqual(metadata["enable_gpu"], device == "cuda")
                self.assertEqual(metadata["competition_sources"], [])
                self.assertEqual(metadata["dataset_sources"], ["alanchoo/rsna-knee-ortho-global-metadata-v1"])
                self.assertEqual(metadata["kernel_sources"], ["alanchoo/rsna-knee-ortho-http-merge-heads-v2"])
                notebook = nbformat.read(directory / "compare.ipynb", as_version=4); nbformat.validate(notebook)
                cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
                self.assertIn("bootstrap_start.json", cells[0].source)
                literals = {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(cells[1].source).body if isinstance(node,ast.Assign)}
                settings = literals["SETTINGS"]; payload = json.loads(zlib.decompress(base64.b64decode(literals["COMPRESSED_PAYLOAD"])))
                self.assertEqual(len(payload), 10)
                for name, encoded in payload.items():
                    source = (ROOT / "scripts/compare_sharded_heads_http_v2.py" if name == "compare_sharded_heads.py"
                              else ROOT / ("src" if name.startswith("orthofoundation/") else "scripts") / name)
                    self.assertEqual(base64.b64decode(encoded), source.read_bytes())
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), settings["source_sha256"][name])
                self.assertFalse(any(name.endswith(".csv") for name in payload))
                self.assertEqual(settings["max_seconds"], 1500)
                self.assertEqual(settings["head_fit_budget_seconds"], 1200)
                self.assertFalse(settings["source_bank_ready_verified_during_preparation"])
                self.assertFalse(settings["comparison_launch_allowed"])
                self.assertIsNone(settings["source_bank_sha256"])
                self.assertIn("timeout=remaining", cells[1].source)
                events, failures = [], []
                with self.assertRaisesRegex(RuntimeError, "root must seal"):
                    exec(cells[1].source, {"bootstrap_checkpoint": events.append, "bootstrap_save_failure": failures.append})
                self.assertEqual(events, ["imports", "sealed_source_bank_pin"])
                self.assertEqual(len(failures), 1)

    def test_scored_and_unscored_finite_feature_mutations_fail_external_full_bank_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan, _, config, identity, provenance, previous, global_root, _ = self.fixture(base)
            # Simulates root sealing the immutable completed attention output,
            # before either corruption and independently of comparison execution.
            sealed_pin = COMPARE.validate_sharded_bank(plan, previous / "features", identity, provenance)["sha256"]
            for uid in ("1.1.3", "1.1.2"):  # Gold scored; weak training row never scored by these two checkpoints.
                with self.subTest(uid=uid):
                    path = previous / "features" / (uid + ".npz")
                    original = path.read_bytes()
                    with np.load(path, allow_pickle=False) as row: changed = {key:row[key].copy() for key in row.files}
                    slot = int(np.flatnonzero(changed["mask"])[0]); changed["features"][slot, 0, 0] += 8
                    np.savez_compressed(path, **changed)
                    with patch.object(COMPARE.RUNNER, "validate_global", return_value=plan), patch.object(COMPARE, "run_training") as fit:
                        with self.assertRaisesRegex(RuntimeError, "differs from independently sealed pin"):
                            COMPARE.run_comparison(previous, global_root, "synthetic_global_sha", "a" * 64, identity,
                                base / "rejected", config, "cpu", expected_bank_sha256=sealed_pin)
                        fit.assert_not_called()
                    self.assertFalse((base / "rejected").exists())
                    path.write_bytes(original)

    def test_wrong_missing_or_invalid_bank_pin_rejects_before_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan, _, config, identity, _, previous, global_root, _ = self.fixture(base)
            for pin in (None, "invalid", "0" * 64):
                with self.subTest(pin=pin), patch.object(COMPARE.RUNNER, "validate_global", return_value=plan), patch.object(COMPARE, "run_training") as fit:
                    with self.assertRaisesRegex(RuntimeError, "source-bank SHA256"):
                        COMPARE.run_comparison(previous, global_root, "synthetic_global_sha", "a" * 64, identity,
                            base / "rejected", config, "cpu", expected_bank_sha256=pin)
                    fit.assert_not_called()
            self.assertFalse((base / "rejected").exists())

    def test_checkpoint_replay_rejects_changed_scored_features_csv_or_self_consistent_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan, config, _, identity, provenance, previous, _, report = self.fixture(base)
            feature = previous / "features/1.1.3.npz"; original_feature = feature.read_bytes()
            with np.load(feature, allow_pickle=False) as row: changed = {key:row[key].copy() for key in row.files}
            slot = int(np.flatnonzero(changed["mask"])[0]); changed["features"][slot, 0, 0] += 8
            np.savez_compressed(feature, **changed)
            with self.assertRaisesRegex(RuntimeError, "prediction replay disagrees"):
                COMPARE.validate_sharded_baseline(plan, previous, config, identity, provenance)
            feature.write_bytes(original_feature)
            csv = previous / "gold_predictions.csv"; original_csv = csv.read_bytes()
            table = pd.read_csv(csv); table[COMPARE.LABELS] = (table[COMPARE.LABELS] + .03).clip(0, 1); table.to_csv(csv, index=False)
            with self.assertRaisesRegex(RuntimeError, "prediction replay disagrees"):
                COMPARE.validate_sharded_baseline(plan, previous, config, identity, provenance)
            csv.write_bytes(original_csv)
            checkpoint = previous / "head.pt"
            saved = torch.load(checkpoint, weights_only=True); saved["model"]["class_bias"] += .5; torch.save(saved, checkpoint)
            # Updated bytes/SHA receipts and unchanged metadata remain internally
            # consistent; replay must still detect predictions disagree with CSV.
            receipt = {"filename": "head.pt", "bytes": checkpoint.stat().st_size, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            report["production_head"]["checkpoint_receipt"] = receipt; report["checkpoint_receipts"]["head.pt"] = receipt
            write_json(report, previous / "training_report.json")
            with self.assertRaisesRegex(RuntimeError, "prediction replay disagrees"):
                COMPARE.validate_sharded_baseline(plan, previous, config, identity, provenance)


if __name__ == "__main__":
    unittest.main()

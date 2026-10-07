"""CPU fixtures exercise global selection, ownership, coverage and head separation."""
import copy
import ast
import base64
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
import zlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("sharded_fallback_test_runner", ROOT / "scripts/sharded_frozen_extract.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
import build_sharded_notebooks as BUILDER
from orthofoundation.train import load_features, run_training


def fixture_plan():
    studies = ["1.1.1", "1.1.2", "1.1.3"]
    rows = [
        (studies[0], "1.2.10", 1, "Sagittal", 0, 30),
        (studies[0], "1.2.11", 1, "Sagittal", 1, 35),  # Global best; local shard0 must not substitute10.
        (studies[0], "1.2.12", 1, "Coronal", 0, 20),
        (studies[1], "1.2.13", 1, "Sagittal", 2, 24),
        (studies[2], "1.2.14", 1, "Sagittal", 6, 16),
    ]
    series = pd.DataFrame([row[:4] for row in rows], columns=["StudyInstanceUID", "SeriesInstanceUID", "Fat_Suppression", "Anatomical_Plane"])
    metadata = pd.DataFrame([(row[0], row[1], row[4], row[5]) for row in rows], columns=["study", "series", "shard", "n"])
    labels = pd.DataFrame({"is_gold": [0, 0, 1], "fold": [0, 1, -1], **{label: [.8, .2, .9] for label in RUNNER.LABELS}}, index=studies)
    official = pd.DataFrame({label: [np.nan, np.nan, 1] for label in RUNNER.LABELS}, index=studies)
    required = {"studies": 3, "weak": 2, "gold": 1, "series": 5, "selected_series": 4, "images": 16}
    plan = RUNNER.build_global_plan(series, metadata, labels, official, required)
    plan.update(manifest_sha256="synthetic_global_sha", fingerprints=[{"plan_sha256": str(shard)} for shard in range(7)],
                pin_by_shard={shard: {"plan_sha256": str(shard)} for shard in range(7)})
    return plan, series, metadata


def partial_roots(base, plan, identity, provenance):
    roots = []
    for shard in range(7):
        root = base / f"partial_{shard}"
        (root / "features").mkdir(parents=True)
        ids = [uid for uid in plan["ids"] if RUNNER.owned_mask(plan, uid, {shard}).any()]
        for uid in ids:
            mask = RUNNER.owned_mask(plan, uid, {shard})
            features = np.zeros((6, 4, 1024), np.float16)
            features[mask] = (shard + 1) / 10
            positions = np.zeros((6, 4), np.float32)
            selected = [""] * 6
            for slot in np.flatnonzero(mask):
                sr = plan["selected"][uid][slot]
                selected[slot] = sr
                positions[slot] = RUNNER.slice_indices(plan["counts"][sr], 4) / (plan["counts"][sr] - 1)
            np.savez_compressed(root / "features" / f"{uid}.npz", features=features, mask=mask, positions=positions,
                                study=uid, identity=identity, selected_series=np.asarray(selected),
                                ownership_mask=mask, global_mask=plan["masks"][uid], shards=np.asarray([shard]),
                                orchestrator_sha256=provenance["orchestrator_sha256"],
                                global_metadata_manifest_sha256=provenance["global_metadata_manifest_sha256"])
        RUNNER.partial_manifest(root, ids, plan, {shard}, identity, provenance, complete=True, pilot=False)
        roots.append(root)
    return roots


class ShardedFallbackContracts(unittest.TestCase):
    def setUp(self):
        self.plan, self.series, self.metadata = fixture_plan()
        self.config = yaml.safe_load((ROOT / "configs/orthofoundation_frozen.yaml").read_text())
        self.identity, self.provenance = RUNNER.make_identity(self.plan, self.config, "synthetic_asset_sha")

    def test_global_best_series_is_selected_before_local_shard_filter(self):
        self.assertEqual(self.plan["selected"]["1.1.1"][0], "1.2.11")
        self.assertFalse(RUNNER.owned_mask(self.plan, "1.1.1", {0})[0])
        self.assertTrue(RUNNER.owned_mask(self.plan, "1.1.1", {0})[1])
        self.assertTrue(RUNNER.owned_mask(self.plan, "1.1.1", {1})[0])

    def test_wrong_series_study_mapping_and_gold_flags_are_rejected(self):
        metadata = self.metadata.copy()
        metadata.loc[0, "study"] = "1.1.2"
        with self.assertRaisesRegex(RuntimeError, "series-to-study"):
            RUNNER.build_global_plan(self.series, metadata, self.plan["labels"], self.plan["official"], self.plan["summary"])
        labels = self.plan["labels"].copy()
        labels.loc["1.1.3", "is_gold"] = 0
        with self.assertRaisesRegex(RuntimeError, "Gold"):
            RUNNER.build_global_plan(self.series, self.metadata, labels, self.plan["official"], self.plan["summary"])

    def test_global_member_tamper_and_mounted_pin_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata.csv").write_text("synthetic")
            manifest = {"members": [{"path": "metadata.csv", "bytes": 9, "sha256": RUNNER.file_sha(root / "metadata.csv")}]}
            (root / "global_metadata_manifest.json").write_text(json.dumps(manifest))
            pin = RUNNER.file_sha(root / "global_metadata_manifest.json")
            RUNNER.verify_global_members(root, manifest, pin)
            (root / "metadata.csv").write_text("tampered!")
            with self.assertRaisesRegex(RuntimeError, "bytes/SHA"):
                RUNNER.verify_global_members(root, manifest, pin)
        with patch.object(RUNNER, "load_cache_mixed", return_value=({}, {}, [], [{"plan_sha256": "wrong"}])):
            with self.assertRaisesRegex(RuntimeError, "fingerprints"):
                RUNNER.validate_mounted_cache([Path("unused")], [0], self.plan)

    def test_disjoint_merge_preserves_boundary_features_and_absent_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = partial_roots(base, self.plan, self.identity, self.provenance)
            result = RUNNER.merge_partials(self.plan, roots, base / "merged", self.identity, self.provenance)
            self.assertTrue(result["feature_bank_complete"])
            self.assertFalse(result["head_training_complete"])
            self.assertEqual(result["images_k4"], 16)
            with np.load(base / "merged/features/1.1.1.npz") as feature:
                np.testing.assert_array_equal(feature["features"][0], np.full((4, 1024), .2, dtype=np.float16))
                np.testing.assert_array_equal(feature["features"][1], np.full((4, 1024), .1, dtype=np.float16))
                self.assertTrue(np.all(feature["features"][~feature["mask"]] == 0))
                np.testing.assert_array_equal(feature["mask"], self.plan["masks"]["1.1.1"])

    def test_overlap_and_missing_study_fail_before_any_merge_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = partial_roots(base, self.plan, self.identity, self.provenance)
            output = base / "merged"
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                RUNNER.merge_partials(self.plan, [*roots, roots[0]], output, self.identity, self.provenance)
            self.assertFalse((output / "features").exists())
            path = roots[1] / "partial_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["features"] = []
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "study coverage"):
                RUNNER.merge_partials(self.plan, roots, output, self.identity, self.provenance)
            self.assertFalse((output / "features").exists())

    def test_wrong_identity_or_validly_hashed_wrong_slice_positions_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = partial_roots(base, self.plan, self.identity, self.provenance)
            with self.assertRaisesRegex(RuntimeError, "identity"):
                RUNNER.merge_partials(self.plan, roots, base / "bad_identity", "other_identity", self.provenance)
            path = roots[1] / "features/1.1.1.npz"
            with np.load(path) as row:
                value = {key: row[key].copy() for key in row.files}
            value["positions"][0, 0] += .1
            np.savez_compressed(path, **value)
            manifest_path = roots[1] / "partial_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["features"][0].update(bytes=path.stat().st_size, sha256=RUNNER.file_sha(path))
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "slice positions"):
                RUNNER.merge_partials(self.plan, roots, base / "bad_positions", self.identity, self.provenance)

    def test_global_gpu_budget_prevents_seven_individually_small_but_expensive_jobs(self):
        slow = RUNNER.budget_projection(5, 30, 1000, 7)
        self.assertLess(slow["projected_extraction_seconds_for_shards"], 600)
        self.assertFalse(slow["fits_total_frozen_budget"])
        fast = RUNNER.budget_projection(25, 30, 12500, 7)
        self.assertTrue(fast["fits_total_frozen_budget"])
        self.assertAlmostEqual(fast["projected_total_including_head_seconds"] - fast["projected_total_frozen_seconds"], 1200)
        self.assertAlmostEqual(fast["projected_total_including_head_job_seconds"] - fast["projected_total_frozen_seconds"], 1500)

    def test_total_caps_include_pilot_and_reject_negative_or_insufficient_budgets(self):
        counts = [12240, 12528, 12580, 12332, 12364, 12496, 10796]
        caps = [RUNNER.allocated_extraction_cap(count) for count in counts]
        self.assertLessEqual(sum(caps) + RUNNER.PILOT_BUDGET_SECONDS, 9000)
        self.assertFalse(RUNNER.budget_projection(25, 30, 12500, 7, allocated_cap=1)["fits_allocated_shard_budget"])
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            RUNNER.budget_projection(-1, 0, 12500, 7)
        with patch.object(RUNNER.time, "monotonic", return_value=11):
            with self.assertRaisesRegex(RuntimeError, "walltime budget"):
                RUNNER.check_elapsed(0, 10)

    def test_only_complete_merged_bank_trains_fresh_fixed_heads_with_gold_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = partial_roots(base, self.plan, self.identity, self.provenance)
            output = base / "merged"
            RUNNER.merge_partials(self.plan, roots, output, self.identity, self.provenance)
            config = dict(self.config, head_dim=32, head_dropout=0.)
            result = run_training(self.plan["labels"], self.plan["official"], output / "features", self.identity, config, output, "cpu")
            self.assertEqual(result["weak_holdout_head"]["optimizer_updates"], 12)
            self.assertEqual(result["production_head"]["optimizer_updates"], 12)
            self.assertEqual(result["weak_holdout_head"]["n_gradient_studies"], 1)
            self.assertEqual(result["production_head"]["n_gradient_studies"], 2)
            self.assertEqual(result["production_head"]["gold_used_for_gradients"], 0)
            self.assertFalse(result["production_head"]["gold_checkpoint_selection"])
            self.assertNotIn("1.1.3", pd.read_csv(output / "gradient_train_ids.csv").StudyInstanceUID.tolist())

    def test_notebook_vendors_exact_core_plus_new_runner_and_starts_diagnostics_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            settings = {"mode": "pilot", "config": self.config}
            BUILDER.create_notebook(directory, settings, {"competition_sources": [], "is_private": True})
            notebook = nbformat.read(directory / "train.ipynb", as_version=4)
            nbformat.validate(notebook)
            cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
            self.assertIn("bootstrap_start.json", cells[0].source)
            literals = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(cells[1].source).body if isinstance(node, ast.Assign)}
            payload = json.loads(zlib.decompress(base64.b64decode(literals["COMPRESSED_PAYLOAD"])))
            self.assertEqual(len(payload), 8)
            self.assertIn("sharded_frozen_extract.py", payload)
            for name, encoded in payload.items():
                source = ROOT / ("scripts" if name == "sharded_frozen_extract.py" else "src") / name
                self.assertEqual(base64.b64decode(encoded), source.read_bytes())
            self.assertFalse(any(name.endswith(".csv") for name in payload))
            self.assertEqual(literals["SETTINGS"]["config"], self.config)

    def test_new_kaggle_titles_resolve_to_exact_pilot_shard_and_merge_references(self):
        slugs = ["rsna-knee-ortho-sharded-pilot", *[f"rsna-knee-ortho-shard-{shard}" for shard in range(7)],
                 "rsna-knee-ortho-merge-heads"]
        titles = [BUILDER.title_for_slug(slug) for slug in slugs]
        normalized = [re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") for title in titles]
        self.assertEqual(normalized, slugs)
        self.assertEqual(len(set(normalized)), 9)
        with self.assertRaisesRegex(RuntimeError, "normalize"):
            BUILDER.title_for_slug("Not-a-normalized-slug")


if __name__ == "__main__":
    unittest.main()

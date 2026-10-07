"""Controlled head ablation: unchanged feature bank, input manifest and epochs."""
import ast
import base64
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from compare_frozen_heads import (FEATURE_KEYS, SOURCE_KEYS, controlled_config, run_comparison,
                                  validate_feature_bank, validate_head_receipt, validate_provenance)
from orthofoundation.data import LABELS, ids_hash
from orthofoundation.encoder import file_sha
from orthofoundation.run import extraction_identity
from orthofoundation.train import fit


class HeadComparisonContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.previous = yaml.safe_load((ROOT / "configs/orthofoundation_frozen.yaml").read_text())
        self.comparison = yaml.safe_load((ROOT / "configs/orthofoundation_meanmax.yaml").read_text())

    def feature(self, directory, uid="1.1", identity="fixture", k=4):
        feature = np.zeros((6, k, 1024), np.float16)
        feature[0] = np.random.default_rng(3).normal(size=(k, 1024)).astype(np.float16)
        np.savez(Path(directory) / (uid + ".npz"), features=feature,
                 mask=np.array([True, False, False, False, False, False]),
                 positions=np.zeros((6, k), np.float32), study=uid, identity=identity)

    def provenance_fixture(self, directory):
        root = Path(directory)
        labels_path = root / "labels.csv"
        labels_path.write_text("fixture labels\n")
        (root / "train.csv").write_text("official fixture\n")
        (root / "train_series.csv").write_text("official series fixture\n")
        ids, gold = ["1.1", "1.2"], ["1.2"]
        provenance = {"experiment": self.previous["experiment"],
                      "feature_config": {key: self.previous[key] for key in FEATURE_KEYS},
                      "checkpoint_sha256": self.previous["checkpoint_sha256"], "labels_order": LABELS,
                      "n_all_studies": len(ids), "all_study_ids_sha256": ids_hash(ids), "n_gold": len(gold),
                      "labels_sha256": file_sha(labels_path), "official_train_sha256": file_sha(root / "train.csv"),
                      "official_series_sha256": file_sha(root / "train_series.csv"),
                      "assets_manifest_sha256": "a" * 64}
        source = ROOT / "src/orthofoundation"
        for key, filename in SOURCE_KEYS.items():
            if key not in ("head_code_sha256", "training_code_sha256"):
                provenance[key] = file_sha(source / filename)
        provenance["feature_identity"] = extraction_identity(provenance)
        for key in ("head_code_sha256", "training_code_sha256"):
            provenance[key] = file_sha(source / SOURCE_KEYS[key])
        return provenance, labels_path, root, ids, gold

    def test_checked_in_config_changes_exactly_head_and_experiment(self):
        result = controlled_config(self.previous, self.comparison)
        self.assertEqual(result["head_kind"], "meanmax")
        self.assertEqual({key for key in self.previous if self.previous[key] != result[key]},
                         {"head_kind", "experiment"})
        self.assertEqual(result["head_epochs"], 12)
        self.assertEqual(result["seed"], 42)

    def test_geometry_split_optimizer_and_forward_changes_are_refused(self):
        # These are the changes that would make the head comparison misleading.
        changes = {"resolution": 336, "slices_per_slot": 8, "fov": .8, "extraction_variant": "canonical_dinov3_rope",
                   "checkpoint_sha256": "b" * 64, "head_epochs": 20, "head_dim": 128, "head_lr": .01,
                   "weak_holdout_fold": 1, "seed": 2}
        for key, value in changes.items():
            with self.subTest(key=key):
                changed = dict(self.comparison, **{key: value})
                with self.assertRaisesRegex(RuntimeError, "Only head_kind"):
                    controlled_config(self.previous, changed)
        with self.assertRaisesRegex(RuntimeError, "keys"):
            controlled_config(self.previous, dict(self.comparison, new_sampling_policy=True))

    def test_different_label_table_and_provenance_manifest_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            provenance, labels_path, root, ids, gold = self.provenance_fixture(temporary)
            expected = validate_provenance(provenance, self.previous, labels_path, root, ids, gold)
            self.assertEqual(expected, provenance["feature_identity"])
            changed = copy.deepcopy(provenance)
            changed["assets_manifest_sha256"] = "b" * 64
            with self.assertRaisesRegex(RuntimeError, "identity"):
                validate_provenance(changed, self.previous, labels_path, root, ids, gold)
            labels_path.write_text("different labels\n")
            with self.assertRaisesRegex(RuntimeError, "Different input manifest: labels_sha256"):
                validate_provenance(provenance, self.previous, labels_path, root, ids, gold)

    def test_same_bank_is_checked_and_different_identity_or_geometry_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.feature(temporary)
            result = validate_feature_bank(["1.1"], temporary, "fixture", 4)
            self.assertEqual(result["studies"], 1)
            self.assertTrue(result["every_uid_geometry_identity_and_values_checked"])
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                validate_feature_bank(["1.1"], temporary, "different", 4)
            with self.assertRaisesRegex(RuntimeError, "shape"):
                validate_feature_bank(["1.1"], temporary, "fixture", 8)
            self.feature(temporary, "1.2")
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                validate_feature_bank(["1.1"], temporary, "fixture", 4)

    def test_out_of_bounds_positions_are_rejected_by_original_feature_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.feature(temporary)
            path = Path(temporary) / "1.1.npz"
            with np.load(path, allow_pickle=False) as row:
                content = {key: row[key].copy() for key in row.files}
            content["positions"][0, 0] = 1.1
            np.savez(path, **content)
            with self.assertRaisesRegex(RuntimeError, "positions"):
                validate_feature_bank(["1.1"], temporary, "fixture", 4)

    def test_original_weight_receipt_rejects_replaced_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.pt"
            path.write_bytes(b"original checkpoint")
            receipt = {"filename": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}
            self.assertEqual(validate_head_receipt(temporary, path.name, receipt), receipt)
            path.write_bytes(b"replaced checkpoint")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                validate_head_receipt(temporary, path.name, receipt)

    def test_meanmax_trains_fixed_epochs_and_excludes_gold(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.feature(temporary)
            labels = pd.DataFrame({"is_gold": [0], **{label: [.8] for label in LABELS}}, index=["1.1"])
            config = dict(self.comparison, head_dim=16, head_batch_size=1, head_dropout=0)
            _, report = fit(["1.1"], labels, temporary, "fixture", config, "cpu", Path(temporary) / "head.pt")
            self.assertEqual(report["epochs"], 12)
            self.assertEqual(report["optimizer_updates"], 12)
            self.assertTrue(report["fixed_final_epoch"])
            self.assertFalse(report["gold_checkpoint_selection"])
            self.assertEqual(report["gold_used_for_gradients"], 0)
            labels.loc["1.1", "is_gold"] = 1
            with self.assertRaisesRegex(RuntimeError, "Gold"):
                fit(["1.1"], labels, temporary, "fixture", config, "cpu", Path(temporary) / "forbidden.pt")
            self.assertFalse((Path(temporary) / "forbidden.pt").exists())

    def test_source_output_overlap_is_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "original.txt"
            marker.write_text("immutable")
            for output in (root, root / "child"):
                with self.assertRaisesRegex(RuntimeError, "separate"):
                    run_comparison(root, root / "labels.csv", root, output, self.comparison, "cpu")
            self.assertEqual(list(root.iterdir()), [marker])

    def test_private_notebook_embeds_exact_current_source_and_only_needed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_path = root / "docker.json"
            docker = "gcr.io/kaggle-private-byod/python@sha256:" + "a" * 64
            metadata_path.write_text(json.dumps({"docker_image": docker}))
            subprocess.run([sys.executable, str(ROOT / "scripts/build_head_comparison_notebook.py"),
                            "--docker-metadata", str(metadata_path), "--output-dir", str(root / "notebook")],
                           check=True, capture_output=True, text=True)
            metadata = json.loads((root / "notebook/kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_internet"])
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(metadata["docker_image"], docker)
            self.assertEqual(metadata["kernel_sources"], ["alanchoo/rsna-knee-orthofoundation-full"])
            self.assertEqual(metadata["dataset_sources"], ["alanchoo/orthofoundation-l-assets-v1"])
            notebook = nbformat.read(root / "notebook/compare.ipynb", as_version=4)
            tree = ast.parse(notebook.cells[1].source)
            compressed = ast.literal_eval(tree.body[0].value)
            settings = ast.literal_eval(tree.body[1].value)
            payload = json.loads(zlib.decompress(base64.b64decode(compressed)))
            expected = {"orthofoundation/" + path.name: path for path in (ROOT / "src/orthofoundation").glob("*.py")}
            expected["compare_frozen_heads.py"] = ROOT / "scripts/compare_frozen_heads.py"
            self.assertEqual(set(payload), set(expected))
            self.assertEqual(len(payload), 8)
            for name, path in expected.items():
                value = base64.b64decode(payload[name])
                self.assertEqual(value, path.read_bytes())
                self.assertEqual(hashlib.sha256(value).hexdigest(), settings["source_sha256"][name])


if __name__ == "__main__":
    unittest.main()

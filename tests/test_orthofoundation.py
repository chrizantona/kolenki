"""Meaningful contracts: missing series, extraction provenance and Gold exclusion."""
import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orthofoundation.data import LABELS, build_table, make_views, slice_indices, validated_labels
from orthofoundation.heads import StudyHead
from orthofoundation.run import extraction_identity, feature_exists
from orthofoundation.train import fit, load_features, score_auc


class PipelineContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_absent_slots_never_change_either_head_prediction(self):
        features = torch.randn(2, 6, 4, 1024)
        mask = torch.tensor([[True, False, False, True, False, False], [False, True, False, False, False, False]])
        positions = torch.rand(2, 6, 4)
        changed = features.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 1e6
        positions_changed = positions.clone()
        positions_changed[~mask] = 1000
        for kind in ("attention", "meanmax"):
            head = StudyHead(kind=kind).eval()
            with torch.no_grad():
                expected = head(features, mask, positions)
                actual = head(changed, mask, positions_changed)
            torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
            self.assertTrue(torch.isfinite(actual).all())

    def test_empty_study_is_rejected_instead_of_attending_to_padding(self):
        with self.assertRaisesRegex(ValueError, "Empty"):
            StudyHead()(torch.zeros(1, 6, 4, 1024), torch.zeros(1, 6, dtype=torch.bool), torch.zeros(1, 6, 4))

    def test_crop_is_single_slice_rgb_with_imagenet_normalization(self):
        slices = torch.full((2, 384, 384), 255, dtype=torch.uint8)
        result = make_views(slices)
        self.assertEqual(result.shape, (2, 3, 224, 224))
        expected = torch.tensor([(1-.485)/.229, (1-.456)/.224, (1-.406)/.225])
        torch.testing.assert_close(result[0, :, 20, 20], expected)

    def test_series_ties_are_deterministic_and_missing_slots_stay_empty(self):
        series = pd.DataFrame({"StudyInstanceUID": ["1", "1"], "SeriesInstanceUID": ["2", "1"],
                               "Fat_Suppression": [1, 1], "Anatomical_Plane": ["Sagittal", "Sagittal"]})
        self.assertEqual(build_table(series, {"1": 8, "2": 8})["1"], [["1", "2"], [], [], [], [], []])

    def test_forward_variant_changes_feature_identity(self):
        a = extraction_identity({"extraction_variant": "author_no_rope"})
        b = extraction_identity({"extraction_variant": "canonical_dinov3_rope"})
        self.assertNotEqual(a, b)

    def test_feature_resume_rejects_changed_provenance_and_nonzero_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "1.npz"
            features = np.zeros((6, 4, 1024), np.float16)
            mask = np.array([True, False, False, False, False, False])
            def save(identity):
                np.savez(path, features=features, mask=mask, positions=np.zeros((6, 4), np.float32), study="1", identity=identity)
            save("original")
            self.assertTrue(feature_exists(path, "1", "original", 4))
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                feature_exists(path, "1", "changed", 4)
            features[1, 0, 0] = 1
            save("original")
            with self.assertRaisesRegex(RuntimeError, "missing slots"):
                feature_exists(path, "1", "original", 4)

    def test_gold_flags_must_match_official_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            official = pd.DataFrame({"StudyInstanceUID": ["1.1", "1.2"], **{label: [np.nan, 1] for label in LABELS}})
            official.to_csv(root / "train.csv", index=False)
            labels = pd.DataFrame({"StudyInstanceUID": ["1.1", "1.2"], "is_gold": [0, 0], "fold": [0, 0],
                                   **{label: [.2, .3] for label in LABELS}})
            labels.to_csv(root / "labels.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "Gold"):
                validated_labels(root / "labels.csv", root)

    def test_auc_does_not_crash_on_single_class_gold_target(self):
        targets = np.zeros((3, 12), dtype=int)
        targets[:, 0] = [0, 1, 1]
        result = score_auc(targets, np.ones((3, 12)) * .5, "diagnostic")
        self.assertEqual(result["classes_with_both_labels"], 1)
        self.assertEqual(result["macro_auc"], .5)
        self.assertIsNone(result["per_class_auc"][LABELS[1]])

    def test_fresh_head_actually_updates_and_rejects_gold_gradient(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.array([True, False, False, False, False, False])
            features = np.zeros((6, 4, 1024), np.float16)
            features[0] = np.random.default_rng(4).normal(size=(4, 1024)).astype(np.float16)
            np.savez(root / "1.npz", features=features, mask=mask, positions=np.zeros((6, 4), np.float32), study="1", identity="test")
            labels = pd.DataFrame({"is_gold": [0], **{label: [.8] for label in LABELS}}, index=["1"])
            config = {"seed": 2, "head_dim": 32, "head_kind": "attention", "head_dropout": 0,
                      "head_lr": .01, "head_weight_decay": 0, "head_epochs": 1, "head_batch_size": 1}
            head, report = fit(["1"], labels, root, "test", config, "cpu", root / "head.pt")
            self.assertEqual(report["optimizer_updates"], 1)
            self.assertTrue((head.class_bias != 0).all())
            self.assertTrue((root / "head.pt").is_file())
            self.assertEqual(report["checkpoint_receipt"]["bytes"], (root / "head.pt").stat().st_size)
            self.assertEqual(report["checkpoint_receipt"]["sha256"], hashlib.sha256((root / "head.pt").read_bytes()).hexdigest())
            labels.loc["1", "is_gold"] = 1
            with self.assertRaisesRegex(RuntimeError, "Gold"):
                fit(["1"], labels, root, "test", config, "cpu", root / "forbidden.pt")
            self.assertFalse((root / "forbidden.pt").exists())


if __name__ == "__main__":
    unittest.main()

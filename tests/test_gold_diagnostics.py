"""Synthetic Gold58 cases only; no real labels, predictions or model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gold_diagnostics_test', ROOT / 'scripts/compare_gold_diagnostics.py')
GOLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLD)


class GoldDiagnosticContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic_gold58_')
        self.root = Path(self.temp.name)
        self.ids = [f'1.1.{n:04d}' for n in range(58)]
        self.targets = np.tile(np.arange(58)[:, None] % 2, (1, 12))
        logits = 4 * self.targets - 2
        self.official = self.root / 'official.csv'
        self.baseline = self.root / 'raw_logits.csv'
        self.candidate = self.root / 'probabilities.csv'
        self.write(self.official, self.targets)
        self.write(self.baseline, logits)
        self.write(self.candidate, np.where(self.targets, .7, .3))
        self.pin = GOLD.file_sha(self.baseline)
        self.ids_pin = hashlib.sha256('\n'.join(sorted(self.ids)).encode()).hexdigest()

    def tearDown(self): self.temp.cleanup()

    def write(self, path, values, ids=None):
        table = pd.DataFrame(values, columns=GOLD.LABELS)
        table.insert(0, GOLD.ID, ids or self.ids)
        table.to_csv(path, index=False)

    def load(self):
        return GOLD.load_inputs(self.official, self.baseline, {'attention': self.candidate},
                                baseline_sha=self.pin, gold_ids_sha=self.ids_pin)

    def test_alignment_is_by_exact_shared_ids_and_official_weak_rows_are_excluded(self):
        table = pd.read_csv(self.candidate, dtype={GOLD.ID: str}).sample(frac=1, random_state=7)
        table.to_csv(self.candidate, index=False)
        official = pd.read_csv(self.official, dtype={GOLD.ID: str})
        weak = pd.DataFrame([{GOLD.ID: '1.1.9999', **{label: np.nan for label in GOLD.LABELS}}])
        pd.concat([official, weak], ignore_index=True).to_csv(self.official, index=False)
        targets, scores, probabilities, receipts = self.load()
        np.testing.assert_array_equal(targets, self.targets)
        np.testing.assert_allclose(scores['attention'], np.where(self.targets, .7, .3))
        self.assertEqual(receipts['baseline_raw_logits_sha256'], self.pin)

    def test_duplicate_missing_or_wrong_gold_ids_and_external_baseline_pin_fail(self):
        original = self.candidate.read_bytes()
        for mutation in ('duplicate', 'wrong', 'missing'):
            table = pd.read_csv(self.candidate, dtype={GOLD.ID: str})
            if mutation == 'duplicate': table.loc[0, GOLD.ID] = table.loc[1, GOLD.ID]
            elif mutation == 'wrong': table.loc[0, GOLD.ID] = '1.1.9999'
            else: table = table.iloc[:-1]
            table.to_csv(self.candidate, index=False)
            with self.assertRaises(ValueError): self.load()
            self.candidate.write_bytes(original)
        self.baseline.write_bytes(self.baseline.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'external pin'): self.load()

    def test_label_order_nonfinite_nonbinary_and_nonprobability_inputs_fail(self):
        files = [self.candidate, self.official]
        for path in files:
            original = path.read_bytes(); table = pd.read_csv(path, dtype={GOLD.ID: str})
            table[[GOLD.ID, *reversed(GOLD.LABELS)]].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'label order'): self.load()
            path.write_bytes(original)
        for path, value in [(self.candidate, np.inf), (self.candidate, -1), (self.official, .5)]:
            original = path.read_bytes(); table = pd.read_csv(path, dtype={GOLD.ID: str})
            table[GOLD.LABELS[0]] = table[GOLD.LABELS[0]].astype(float)
            table.loc[0, GOLD.LABELS[0]] = value; table.to_csv(path, index=False)
            with self.assertRaises(ValueError): self.load()
            path.write_bytes(original)
        table = pd.read_csv(self.official, dtype={GOLD.ID: str})
        table[GOLD.LABELS[0]] = table[GOLD.LABELS[0]].astype(float)
        table.loc[0, GOLD.LABELS[0]] = np.nan; table.to_csv(self.official, index=False)
        with self.assertRaisesRegex(ValueError, 'Partially missing'): self.load()

    def test_nonfinite_raw_logits_reject_even_with_matching_explicit_file_pin(self):
        table = pd.read_csv(self.baseline, dtype={GOLD.ID: str})
        table[GOLD.LABELS[0]] = table[GOLD.LABELS[0]].astype(float)
        table.loc[0, GOLD.LABELS[0]] = np.inf
        table.to_csv(self.baseline, index=False)
        self.pin = GOLD.file_sha(self.baseline)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'): self.load()

    def assert_snapshot_survives_replacement_before_parse(self, path, replacement, parse_number):
        original_hashes = {source: GOLD.file_sha(source)
                           for source in (self.official, self.baseline, self.candidate)}
        original_read_csv = pd.read_csv
        calls = 0

        def replace_before_parse(source, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == parse_number:
                self.write(path, replacement)
            return original_read_csv(source, *args, **kwargs)

        with patch.object(GOLD.pd, 'read_csv', side_effect=replace_before_parse):
            targets, scores, probabilities, receipts = GOLD.load_inputs(
                self.official, self.baseline, {'attention': self.candidate},
                baseline_sha=self.pin, gold_ids_sha=self.ids_pin,
                candidate_pins={'attention': original_hashes[self.candidate]})
        self.assertEqual(calls, 3)
        self.assertNotEqual(GOLD.file_sha(path), original_hashes[path])
        np.testing.assert_array_equal(targets, self.targets)
        np.testing.assert_array_equal(scores['convnext_reader'], 4 * self.targets - 2)
        np.testing.assert_array_equal(scores['attention'], np.where(self.targets, .7, .3))
        np.testing.assert_array_equal(probabilities['attention'], np.where(self.targets, .7, .3))
        self.assertEqual(receipts['official_labels_sha256'], original_hashes[self.official])
        self.assertEqual(receipts['baseline_raw_logits_sha256'], original_hashes[self.baseline])
        self.assertEqual(receipts['ortho_probability_csv_sha256']['attention'], original_hashes[self.candidate])

    def test_pinned_candidate_replacement_between_hash_and_parse_uses_same_snapshot(self):
        self.assert_snapshot_survives_replacement_before_parse(
            self.candidate, np.where(self.targets, .2, .8), 3)

    def test_pinned_baseline_replacement_between_hash_and_parse_uses_same_snapshot(self):
        self.assert_snapshot_survives_replacement_before_parse(
            self.baseline, 2 - 4 * self.targets, 2)

    def test_official_replacement_between_hash_and_parse_binds_labels_to_receipt(self):
        self.assert_snapshot_survives_replacement_before_parse(
            self.official, 1 - self.targets, 1)

    def test_optional_verified_ortho_pin_and_once_only_sigmoid_errors_blend(self):
        with self.assertRaisesRegex(ValueError, 'supplied verified pin'):
            GOLD.load_inputs(self.official, self.baseline, {'attention': self.candidate},
                baseline_sha=self.pin, gold_ids_sha=self.ids_pin, candidate_pins={'attention': '0' * 64})
        targets, scores, probabilities, receipts = self.load()
        np.testing.assert_allclose(probabilities['convnext_reader'], 1 / (1 + np.exp(-scores['convnext_reader'])))
        report = GOLD.summarize(targets, scores, probabilities, receipts, repeats=50, predeclared_blend='attention')
        self.assertEqual(report['baseline_sigmoid_applications_for_probability_operations'], 1)
        self.assertEqual(report['models']['convnext_reader']['per_label']['ACL']['errors_at_0_5'], 0)
        self.assertEqual(report['models']['fixed_50_50_blend']['macro_auc_all_12'], 1.)
        self.assertFalse(report['predeclared_blend']['weight_search'])
        serialized = json.dumps(report, allow_nan=False)
        for uid in self.ids: self.assertNotIn(uid, serialized)
        self.assertNotIn('StudyInstanceUID', serialized)
        self.assertNotIn(str(self.root), serialized)

    def test_bootstrap_is_paired_exact_tie_aware_and_undefined_draws_disclosed(self):
        targets, scores, probabilities, receipts = self.load()
        targets[:, 0] = 0; targets[0, 0] = 1
        scores['attention'][:, 0] = (np.arange(58) % 3) / 2
        values, defined = GOLD.bootstrap_auc(targets, scores, repeats=60)
        draws = np.random.default_rng(GOLD.SEED).integers(0, 58, size=(60, 58))
        for repeat, ids in enumerate(draws):
            if defined[repeat, 0]:
                self.assertAlmostEqual(values['attention'][repeat, 0],
                    roc_auc_score(targets[ids, 0], scores['attention'][ids, 0]), places=12)
            else: self.assertTrue(np.isnan(values['attention'][repeat, 0]))
        report = GOLD.summarize(targets, scores, probabilities, receipts, repeats=60)
        self.assertGreater(report['bootstrap']['undefined_repeats_per_label']['ACL'], 0)
        self.assertEqual(report['bootstrap']['undefined_macro_repeats'], int((~defined.all(axis=1)).sum()))
        self.assertEqual(report, GOLD.summarize(targets, scores, probabilities, receipts, repeats=60))

    def test_extreme_logits_keep_raw_auc_order_and_plot_contains_aggregate_only(self):
        logits = np.tile(np.linspace(900, 1100, 58)[:, None], (1, 12))
        targets = np.tile((np.arange(58) >= 29)[:, None], (1, 12))
        baseline_prob = GOLD.sigmoid_once(logits)
        self.assertTrue(np.isfinite(baseline_prob).all())
        np.testing.assert_array_equal(baseline_prob, np.ones_like(logits))
        report = GOLD.summarize(targets, {'convnext_reader': logits, 'attention': targets * .4 + .3},
            {'convnext_reader': baseline_prob, 'attention': targets * .4 + .3}, {}, repeats=30)
        self.assertEqual(report['models']['convnext_reader']['macro_auc_all_12'], 1.)
        self.assertEqual(report['models']['convnext_reader']['per_label']['ACL']['errors_at_0_5'], 29)
        png = self.root / 'aggregate.png'; GOLD.plot_report(report, png)
        self.assertTrue(png.is_file()); self.assertGreater(png.stat().st_size, 10000)
        self.assertEqual(png.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')


if __name__ == '__main__': unittest.main()

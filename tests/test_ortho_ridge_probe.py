"""Meaningful synthetic CPU contracts; no real probe/head/encoder fits."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import probe_orthofoundation_ridge as R


def bank_fixture(directory):
    digest = hashlib.sha256(); ids = ['1.2.1', '1.2.2']
    for index, uid in enumerate(ids):
        f = np.zeros((6, 4, 1024), np.float16); f[0] = index + 1
        m = np.array([True, False, False, False, False, False])
        p = np.zeros((6, 4), np.float32); p[0] = [0, .3, .7, 1]
        stream = BytesIO(); np.savez_compressed(stream, features=f, mask=m, positions=p,
            study=np.asarray(uid), identity=np.asarray(R.C.TRAINING_IDENTITY),
            selected_series=np.asarray(['1.3.1', '', '', '', '', '']),
            global_metadata_manifest_sha256=np.asarray(R.GLOBAL_SHA),
            orchestrator_sha256=np.asarray(R.C.SOURCE8['sharded_frozen_extract.py']))
        data = stream.getvalue(); (directory / (uid + '.npz')).write_bytes(data)
        digest.update(f'{uid}.npz\t{len(data)}\t{hashlib.sha256(data).hexdigest()}\n'.encode())
    return ids, digest.hexdigest()


def official_fixture():
    ids = [f'1.2.{index}' for index in range(4407)]
    labels = pd.DataFrame(.3, index=ids, columns=R.LABELS)
    labels['is_gold'] = np.r_[np.zeros(4349, int), np.ones(58, int)]
    labels['fold'] = np.r_[np.repeat([0, 1, 2, 3, 4], [870, 870, 870, 870, 869]), np.full(58, -1)]
    official = pd.DataFrame(np.nan, index=ids, columns=R.LABELS)
    official.iloc[-58:] = np.repeat((np.arange(58) % 2)[:, None], 12, axis=1)
    return labels, official


class RidgeProbeContracts(unittest.TestCase):
    def test_direct_output_symlink_refused_before_inputs_and_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'ordinary'; target.mkdir()
            linked = Path(temporary) / 'linked'; linked.symlink_to(target, target_is_directory=True)
            with patch.object(R, 'snapshot') as read, patch.object(R, 'fit_probe') as fit:
                with self.assertRaisesRegex(RuntimeError, 'Linked output directory refused'):
                    R.run(SimpleNamespace(output_dir=str(linked)))
                read.assert_not_called(); fit.assert_not_called()
            self.assertEqual(list(target.iterdir()), [])

    def test_observed_only_gradient_statistics_and_eval_cannot_change_them(self):
        pooled = np.zeros((4, 6, 2), np.float32); pooled[0, 0] = [1, 3]; pooled[1, 0] = [3, 7]
        pooled[2, 0] = [1000, 5000]; pooled[3, 0] = [-1000, -2000]
        masks = np.zeros((4, 6), bool); masks[:, 0] = True
        scaler = R.SlotCenterScale().fit(pooled[:2], masks[:2], ['g0', 'g1'], ['holdout', 'Gold'])
        np.testing.assert_array_equal(scaler.mean[0], [2, 5]); np.testing.assert_array_equal(scaler.std[0], [1, 2])
        before = scaler.mean.copy(); scaler.transform(pooled[2:], masks[2:]); np.testing.assert_array_equal(before, scaler.mean)
        np.testing.assert_array_equal(scaler.counts, [2, 0, 0, 0, 0, 0])

    def test_gold_and_holdout_scaler_leakage_refuses_before_ridge_fit(self):
        pooled = np.zeros((1, 6, 2), np.float32); masks = np.ones((1, 6), bool)
        for forbidden in ['Gold', 'holdout']:
            with patch.object(R, 'Ridge') as fit:
                with self.assertRaisesRegex(RuntimeError, 'cannot fit'):
                    R.SlotCenterScale().fit(pooled, masks, [forbidden], [forbidden])
                fit.assert_not_called()

    def test_missing_slots_zero_and_mask_indicators_after_nonzero_center(self):
        pooled = np.zeros((2, 6, 2), np.float32); pooled[0, 0] = [5, 9]
        masks = np.zeros((2, 6), bool); masks[0, 0] = True
        scaler = R.SlotCenterScale().fit(pooled, masks, ['g0', 'g1'], [])
        result = scaler.transform(pooled, masks)
        np.testing.assert_array_equal(result[1, :12], np.zeros(12))
        np.testing.assert_array_equal(result[:, -6:], masks)
        self.assertEqual(float(scaler.std[0, 0]), .001)

    def test_wrong_missing_bank_pin_and_self_consistent_feature_tamper_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); ids, pin = bank_fixture(directory)
            pooled, mask, receipt = R.read_bank(ids, directory, pin)
            self.assertEqual(receipt['sha256'], pin); self.assertEqual(pooled.shape, (2, 6, 1024))
            for wrong in [None, '0' * 64]:
                with self.assertRaises(RuntimeError): R.read_bank(ids, directory, wrong)
            path = directory / (ids[0] + '.npz')
            with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as archive:
                row = {key: archive[key].copy() for key in archive.files}
            row['features'][0, 0, 0] += 1
            np.savez_compressed(path, **row)
            with self.assertRaisesRegex(RuntimeError, 'external ROOT pin'): R.read_bank(ids, directory, pin)

    def test_label_order_and_duplicate_columns_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'labels.csv'
            for order in [list(reversed(R.LABELS)), R.LABELS + [R.LABELS[0]]]:
                data = ('StudyInstanceUID,' + ','.join(order) + '\n1.2.1,' + ','.join(['0'] * len(order)) + '\n').encode()
                path.write_bytes(data)
                with self.assertRaises(RuntimeError): R.csv_snapshot(path, hashlib.sha256(data).hexdigest(), 'first')

    def test_exact_splits_and_wrong_gold_flag_refusal(self):
        labels, official = official_fixture()
        weak, gold, holdout, gradient = R.split_ids(labels, official)
        self.assertEqual(tuple(map(len, [weak, gold, holdout, gradient])), (4349, 58, 870, 3479))
        self.assertFalse(set(gradient) & set(holdout)); self.assertFalse(set(weak) & set(gold))
        labels.iloc[-1, labels.columns.get_loc('is_gold')] = 0
        with self.assertRaises(RuntimeError): R.split_ids(labels, official)

    def test_targets_clip_and_sigmoid_exactly_once_finite_refusals(self):
        logits = R.target_logits(np.array([[0., .5, 1.]]))
        self.assertTrue(np.isfinite(logits).all()); self.assertEqual(logits[0, 1], 0)
        np.testing.assert_allclose(R.probabilities_from_logits(logits), [[.0001, .5, .9999]])
        self.assertAlmostEqual(R.probabilities_from_logits(np.array([[2.]]))[0, 0], .8807970779778823)
        for value in [np.nan, np.inf]:
            with self.assertRaises(RuntimeError): R.target_logits(np.array([[value]]))
            with self.assertRaises(RuntimeError): R.probabilities_from_logits(np.array([[value]]))

    def test_recipe_change_refused_and_current_frozen_science_unchanged(self):
        recipe, _ = R.load_recipe(ROOT / 'configs/orthofoundation_slot_centered_ridge.yaml')
        R.C.verify_frozen_sources(ROOT)
        self.assertEqual(R.snapshot(ROOT / 'src/orthofoundation_inference/contracts.py')[1], R.CONTRACTS_SHA)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'; changed = dict(recipe); changed['ridge_alpha'] = 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, 'predeclared'): R.load_recipe(path)

    def test_time_and_memory_bound_fail_closed(self):
        with patch.object(R.time, 'monotonic', return_value=301), patch.object(R, 'peak_memory_bytes', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'Python budget'): R.budget(0, R.RECIPE)
        with patch.object(R.time, 'monotonic', return_value=1), patch.object(R, 'peak_memory_bytes', return_value=2147483649):
            with self.assertRaisesRegex(RuntimeError, 'memory budget'): R.budget(0, R.RECIPE)


if __name__ == '__main__': unittest.main()

"""Synthetic contracts only; tiny fake heads/stub encoders, no DINO loading."""
import hashlib
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orthofoundation_inference import contracts as C
from orthofoundation_inference import pipeline as P
from orthofoundation.data import LABELS, MEAN, STD, make_views, slice_indices
from orthofoundation.heads import StudyHead
from orthofoundation.train import predict
from convnext_reader import preprocess


class OrthoInferenceContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC_ortho_inference_')
        self.base = Path(self.temp.name)
        self.config, self.sources, _ = C.verify_frozen_sources()

    def tearDown(self): self.temp.cleanup()

    def rows(self):
        return pd.DataFrame([
            ('1.1', '1.2.10', 'Sagittal', 1), ('1.1', '1.2.09', 'Sagittal', 1),
            ('1.1', '1.2.11', 'Sagittal', 1), ('1.1', '1.2.12', 'Coronal', 0),
            ('1.1', '1.2.13', 'Axial', 1)],
            columns=['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'Fat_Suppression'])

    def fake_head_and_seal(self, mutation=None):
        head = StudyHead(dim=256, kind='attention', dropout=.2)
        trained = {'epochs': 12, 'n_gradient_studies': 4349, 'optimizer_updates': 816,
            'gold_used_for_gradients': 0, 'fixed_final_epoch': True, 'gold_checkpoint_selection': False,
            'gradient_ids_sha256': hashlib.sha256(b'SYNTHETIC_manifest').hexdigest(),
            'head_code_sha256': C.SOURCE8['orthofoundation/heads.py'],
            'training_code_sha256': C.SOURCE8['orthofoundation/train.py'],
            'history': [{'epoch': j + 1, 'updates': (j + 1) * 68, 'loss': .5} for j in range(12)]}
        saved = {'model': head.state_dict(), 'labels': LABELS, 'config': dict(self.config),
                 'feature_identity': C.TRAINING_IDENTITY, 'training_report': trained}
        if mutation: mutation(saved)
        path = self.base / 'head.pt'; torch.save(saved, path); head_sha = C.snapshot(path)[1]
        bank_sha = hashlib.sha256(b'SYNTHETIC_full_bank').hexdigest()
        seal = {'schema': 'ROOT_http_v2_attention_full_bank_seal_v1', 'ROOT_actual_full_merge_PASS': True,
            'feature_identity': C.TRAINING_IDENTITY, 'validation_helper_sha256': C.ROOT_SEAL_HELPER_SHA,
            'source_kernel': 'alanchoo/rsna-knee-ortho-http-merge-heads-v2',
            'kernel_version_origin_verified_separately_by_ROOT': True, 'root_confirmed_kernel_version': 1,
            'downloaded_source8_sha256': C.SOURCE8, 'approved_config_yaml_sha256': C.CONFIG_SHA,
            'assets_manifest_sha256': C.ASSETS_SHA,
            'counts': {'studies': 4407, 'weak': 4349, 'Gold': 58, 'CV_gradient': 3479, 'weak_holdout': 870,
                       'production_gradient': 4349, 'selected_slots': 21334, 'images_k4': 85336},
            'source_bank_sha256': bank_sha,
            'feature_bank_receipt': {'studies': 4407, 'images_k4': 85336, 'selected_slot_series': 21334, 'sha256': bank_sha},
            'both_heads': {'epochs': 12, 'CV_optimizer_updates': 660, 'production_optimizer_updates': 816,
                           'Gold_gradient_rows': 0, 'Gold_checkpoint_selection': False},
            'checkpoint_receipts': {'head.pt': {'filename': 'head.pt', 'sha256': head_sha, 'bytes': path.stat().st_size}}}
        seal_path = self.base / 'SYNTHETIC_ROOT_seal.json'; seal_path.write_text(json.dumps(seal))
        return path, head_sha, seal_path, C.snapshot(seal_path)[1]

    def test_placeholder_refuses_without_reading_any_model_or_dicom_inputs(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_orthofoundation_inference.py'), '--prepare-placeholder'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Placeholder only', result.stderr)
        self.assertNotIn('pipeline.py', result.stderr)
        with patch.object(C, 'snapshot', side_effect=AssertionError('No file reads without external pins')):
            with self.assertRaisesRegex(RuntimeError, 'external ROOT'):
                C.load_production_head('head.pt', None, 'seal.json', None, self.config)

    def test_only_strict_production_head_bound_to_external_root_seal_is_admitted(self):
        paths = self.fake_head_and_seal()
        head, admission = C.load_production_head(*paths, self.config)
        self.assertFalse(head.training); self.assertTrue(all(not p.requires_grad for p in head.parameters()))
        self.assertEqual(admission['fixed_epochs'], 12); self.assertEqual(admission['Gold_gradient_studies'], 0)
        with self.assertRaisesRegex(RuntimeError, 'ROOT checkpoint receipt'):
            C.load_production_head(paths[0], '0' * 64, paths[2], paths[3], self.config)
        with self.assertRaisesRegex(RuntimeError, 'snapshot SHA'):
            C.load_production_head(paths[0], paths[1], paths[2], '0' * 64, self.config)
        for name in ('smoke_head.pt', 'head_weak_holdout.pt'):
            with self.assertRaisesRegex(RuntimeError, 'production head.pt'):
                C.load_production_head(self.base / name, paths[1], paths[2], paths[3], self.config)
        # A valid ROOT seal cannot authorize checkpoint bytes replaced afterward.
        paths[0].write_bytes(b'SYNTHETIC_modified_checkpoint')
        with self.assertRaisesRegex(RuntimeError, 'snapshot SHA'):
            C.load_production_head(*paths, self.config)

    def test_external_pin_does_not_admit_wrong_labels_cv_counts_gold_or_nonfinite_state(self):
        mutations = [lambda ck: ck.update(labels=list(reversed(LABELS))),
            lambda ck: ck['training_report'].update(n_gradient_studies=3479),
            lambda ck: ck['training_report'].update(gold_used_for_gradients=1),
            lambda ck: ck['training_report'].update(gold_checkpoint_selection=True),
            lambda ck: ck['training_report']['history'][0].update(updates=1),
            lambda ck: ck['model']['class_bias'].fill_(float('nan'))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                paths = self.fake_head_and_seal(mutation)
                with self.assertRaises(RuntimeError): C.load_production_head(*paths, self.config)

    def test_decoded_capped_depth_then_uid_ties_and_short_slot_mask(self):
        counts = {'1.2.10': 64, '1.2.09': 64, '1.2.11': 40, '1.2.12': 3, '1.2.13': 2}
        seen = []
        def decoder(path, plane_name):
            seen.append(plane_name); n = counts[path.name]
            return np.full((n, 384, 384), 23, np.uint8), {'n': n, 'n_raw': 1000, 'n_bad': 1}
        cache = P.CanonicalVolumeCache(self.rows(), self.base / 'NO_REAL_DICOM', self.base / 'cache', decoder)
        cache.decode(); uid, raw, mask, positions, selected = P.study_sample('1.1', self.rows(), cache)
        self.assertEqual(selected, ['1.2.09', '', '', '', '1.2.12', ''])
        np.testing.assert_array_equal(mask.numpy(), [True, False, False, False, True, False])
        np.testing.assert_array_equal(positions[0].numpy(), slice_indices(64, 4).astype(np.float32) / 63)
        np.testing.assert_array_equal(positions[4].numpy(), [0, .5, .5, 1])
        self.assertEqual(raw.dtype, torch.uint8); self.assertTrue((raw[~mask] == 0).all())
        self.assertEqual(sorted(seen), ['Axial', 'Coronal', 'Sagittal', 'Sagittal', 'Sagittal'])
        self.assertTrue(all(len(cache(sr)) <= 64 for sr in cache.paths))

    def test_six_slots_use_metadata_plane_and_fat_suppression_in_frozen_order(self):
        rows = pd.DataFrame([('1.1', f'1.3.{slot}', plane, fs)
            for slot, (fs, plane) in enumerate((fs, plane) for fs in (1, 0)
                                              for plane in ('Sagittal', 'Coronal', 'Axial'))],
            columns=['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'Fat_Suppression'])
        def decoder(path, plane_name):
            # Actual orientation can differ; slot ownership remains official metadata.
            return np.full((3, 384, 384), int(path.name.rsplit('.', 1)[1]), np.uint8), {'n_raw': 3, 'plane': 'Axial'}
        cache = P.CanonicalVolumeCache(rows, self.base, self.base / 'cache', decoder)
        cache.decode(); _, raw, mask, _, selected = P.study_sample('1.1', rows, cache)
        self.assertEqual(selected, [f'1.3.{slot}' for slot in range(6)])
        self.assertTrue(mask.all().item())
        self.assertEqual([raw[slot, 0, 0, 0].item() for slot in range(6)], list(range(6)))

    def test_all_bad_or_short_series_fail_without_empty_mask_fallback(self):
        def decoder(path, plane_name):
            return np.zeros((1, 384, 384), np.uint8), {'n_raw': 0, 'n_bad': 9}
        cache = P.CanonicalVolumeCache(self.rows(), self.base, self.base / 'cache', decoder)
        self.assertEqual(set(cache.decode().values()), {0})
        with self.assertRaisesRegex(RuntimeError, 'no usable canonical MRI slot'):
            P.study_sample('1.1', self.rows(), cache)

    def test_canonical_loader_uses_metadata_plane_fallback_and_caps_before_percentiles(self):
        directory = self.base / 'SYNTHETIC_DICOM'; directory.mkdir()
        for n in range(80): (directory / f'{n:03d}.synthetic').write_bytes(b'not_real_DICOM')
        def read(path):
            n = int(Path(path).stem)
            image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) + n
            return [(image, None, None, [.4, .4], float(79 - n))]
        with patch.object(preprocess, '_read', side_effect=read):
            volume, info = preprocess.load_series(directory, plane_name='Coronal')
        self.assertEqual(info['plane'], 'Coronal'); self.assertEqual(info['n_raw'], 80)
        self.assertEqual(volume.shape, (64, 384, 384)); self.assertEqual(volume.dtype, np.uint8)
        self.assertTrue(np.isfinite(volume).all())

    def test_literal353_crop_rgb_normalization_and_fp16_buffer_before_fp32_head(self):
        raw = torch.zeros((6, 4, 384, 384), dtype=torch.uint8)
        raw[0] = torch.arange(384, dtype=torch.int64)[None, None, :].remainder(256).to(torch.uint8).expand(4, 384, 384)
        mask = torch.tensor([True, False, False, False, False, False]); positions = torch.zeros(6, 4)
        captured = []
        def encode_stub(model, views, variant):
            captured.append((views.clone(), variant)); return torch.full((len(views), 1024), 1.234567, dtype=torch.float32)
        tensors = P.feature_tensors(None, raw, mask, positions, device='cpu', encode_fn=encode_stub)
        expected_gray = F.interpolate(raw[0, :, None, 15:368, 15:368].float() / 255,
                                      (224, 224), mode='bilinear', align_corners=False, antialias=True)
        expected = (expected_gray.expand(-1, 3, -1, -1) - torch.tensor(MEAN)[None, :, None, None]) / torch.tensor(STD)[None, :, None, None]
        torch.testing.assert_close(captured[0][0], expected, rtol=0, atol=0)
        self.assertEqual(captured[0][1], 'author_no_rope')
        self.assertEqual(tensors[0].dtype, torch.float32); self.assertEqual(tensors[1].dtype, torch.bool)
        self.assertEqual(tensors[2].dtype, torch.float32)
        self.assertEqual(tensors[0][0, 0, 0, 0].item(), float(np.float16(1.234567)))
        self.assertNotEqual(tensors[0][0, 0, 0, 0].item(), torch.tensor(1.234567).item())
        self.assertTrue((tensors[0][0, ~mask] == 0).all()); self.assertTrue((tensors[2][0, ~mask] == 0).all())

    def test_cls_overflow_and_nonfinite_positions_refuse_before_head(self):
        raw = torch.zeros((6, 4, 384, 384), dtype=torch.uint8)
        mask = torch.tensor([True, False, False, False, False, False]); pos = torch.zeros(6, 4)
        with self.assertRaisesRegex(RuntimeError, 'FP16 feature buffer'):
            P.feature_tensors(None, raw, mask, pos, 'cpu', lambda model, views, variant: torch.full((len(views), 1024), 1e9))
        pos[0, 0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'geometry/mask/positions'):
            P.feature_tensors(None, raw, mask, pos, 'cpu', lambda *args: self.fail('No encoder call'))

    def test_manifest_preserves_study_order_and_rejects_duplicate_or_missing_coverage(self):
        pd.DataFrame({'StudyInstanceUID': ['1.2', '1.1']}).to_csv(self.base / 'test.csv', index=False)
        rows = pd.concat([self.rows(), self.rows().assign(StudyInstanceUID='1.2', SeriesInstanceUID=lambda d: d.SeriesInstanceUID + '.1')])
        rows.to_csv(self.base / 'test_series.csv', index=False)
        ids, _, receipt = P.read_metadata(self.base, 'test')
        self.assertEqual(ids, ['1.2', '1.1']); self.assertEqual(receipt['n_test_studies'], 2)
        self.rows().to_csv(self.base / 'test_series.csv', index=False)
        with self.assertRaisesRegex(RuntimeError, 'metadata coverage differs'): P.read_metadata(self.base, 'test')
        rows.to_csv(self.base / 'test_series.csv', index=False)
        pd.DataFrame({'StudyInstanceUID': ['1.1', '1.1']}).to_csv(self.base / 'test.csv', index=False)
        with self.assertRaisesRegex(RuntimeError, 'unique nonmissing'): P.read_metadata(self.base, 'test')
        pd.DataFrame({'StudyInstanceUID': ['..']}).to_csv(self.base / 'test.csv', index=False)
        with self.assertRaisesRegex(RuntimeError, 'Invalid study identifier'): P.read_metadata(self.base, 'test')

    def test_fp32_eval_head_sigmoid_once_and_missing_slots_do_not_change_prediction(self):
        head, _ = C.load_production_head(*self.fake_head_and_seal(), self.config)
        features = torch.randn(1, 6, 4, 1024); mask = torch.tensor([[True, False, False, False, False, False]])
        positions = torch.zeros(1, 6, 4)
        with torch.inference_mode(): expected = head(features, mask, positions).sigmoid().numpy()
        np.testing.assert_array_equal(predict(head, (features, mask, positions), 'cpu', 64), expected)
        features[:, 1:] = 10000; positions[:, 1:] = .9
        np.testing.assert_array_equal(predict(head, (features, mask, positions), 'cpu', 64), expected)
        with patch.object(P.torch.cuda, 'is_available', return_value=False), patch.object(P, 'load_encoder', side_effect=AssertionError('No Mac encoder load')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA required'):
                P.run_inference(self.base, 'test', self.base / 'outputs', head, {}, self.config, self.sources, self.base)

    def _stub_inference_export(self, phase):
        now = [0.0]; output = self.base / phase; output.mkdir()
        own_csv = output / '_own.csv'; own_csv.write_bytes(b'SYNTHETIC_existing_reader')
        original_csv, original_snapshot, original_write, original_link = pd.DataFrame.to_csv, P.snapshot, Path.write_text, P.os.link
        class TinyHead:
            def to(self, **kwargs): return self
            def eval(self): return self
            def requires_grad_(self, value): return self
        def decoder(path, plane_name):
            return np.zeros((3, 384, 384), np.uint8), {'n_raw': 3, 'n_bad': 0}
        original_cache = P.CanonicalVolumeCache
        def export_csv(frame, destination, *args, **kwargs):
            result = original_csv(frame, destination, *args, **kwargs)
            if phase == 'CSV_export': now[0] = 2.0
            return result
        def snapshot(path, *args, **kwargs):
            result = original_snapshot(path, *args, **kwargs)
            if phase == 'CSV_hash' and Path(path).name == '_ortho.csv': now[0] = 2.0
            return result
        def write_text(path, *args, **kwargs):
            result = original_write(path, *args, **kwargs)
            if phase == 'receipt_write' and path.name == '_ortho_provenance.json': now[0] = 2.0
            return result
        def publish(source, target):
            if phase == 'publication_failure' and Path(target).name == '_ortho_provenance.json':
                raise OSError('SYNTHETIC receipt publication failure')
            result = original_link(source, target)
            if phase == 'publication_overrun': now[0] = 2.0
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(P.time, 'monotonic', side_effect=lambda: now[0]))
            stack.enter_context(patch.object(P.torch.cuda, 'is_available', return_value=True))
            stack.enter_context(patch.object(P.torch.cuda, 'reset_peak_memory_stats'))
            stack.enter_context(patch.object(P.torch.cuda, 'max_memory_allocated', return_value=1024))
            stack.enter_context(patch.object(P, 'read_metadata', return_value=(['1.1'], self.rows(), {'n_test_studies': 1})))
            stack.enter_context(patch.object(P, 'prepare_encoder_assets', return_value=(self.base, self.base / 'NO_WEIGHTS')))
            stack.enter_context(patch.object(P, 'load_encoder', return_value=None))
            stack.enter_context(patch.object(P, 'CanonicalVolumeCache', side_effect=lambda rows, root, scratch: original_cache(rows, root, scratch, decoder)))
            stack.enter_context(patch.object(P, 'feature_tensors', return_value=None))
            stack.enter_context(patch.object(P, 'predict', return_value=np.full((1, 12), .5)))
            stack.enter_context(patch.object(pd.DataFrame, 'to_csv', export_csv))
            stack.enter_context(patch.object(P, 'snapshot', snapshot))
            stack.enter_context(patch.object(Path, 'write_text', write_text))
            stack.enter_context(patch.object(P.os, 'link', publish))
            if phase == 'within_budget':
                result = P.run_inference(self.base / 'DATA', 'test', output, TinyHead(), {}, self.config, self.sources, self.base / 'ASSETS', 1)
                self.assertTrue(result['inference_complete'])
                self.assertEqual(set(path.name for path in output.iterdir()), {'_own.csv', '_ortho.csv', '_ortho_provenance.json'})
                self.assertEqual(C.snapshot(output / '_ortho.csv')[1], result['prediction_csv_sha256'])
            else:
                error, message = (OSError, 'publication failure') if phase == 'publication_failure' else (RuntimeError, 'budget exhausted')
                with self.assertRaisesRegex(error, message):
                    P.run_inference(self.base / 'DATA', 'test', output, TinyHead(), {}, self.config, self.sources, self.base / 'ASSETS', 1)
                self.assertEqual([path.name for path in output.iterdir()], ['_own.csv'])
        self.assertEqual(own_csv.read_bytes(), b'SYNTHETIC_existing_reader')

    def test_csv_export_overrun_removes_owned_stage_and_never_completes(self):
        self._stub_inference_export('CSV_export')

    def test_csv_hash_overrun_removes_owned_stage_and_never_completes(self):
        self._stub_inference_export('CSV_hash')

    def test_receipt_write_overrun_removes_owned_stage_and_never_completes(self):
        self._stub_inference_export('receipt_write')

    def test_within_budget_exports_matching_pair_and_preserves_existing_own_csv(self):
        self._stub_inference_export('within_budget')

    def test_publication_overrun_rolls_back_already_created_owned_csv(self):
        self._stub_inference_export('publication_overrun')

    def test_second_publication_failure_rolls_back_owned_pair(self):
        self._stub_inference_export('publication_failure')


if __name__ == '__main__': unittest.main()

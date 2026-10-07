"""EXP-OF-003 fixed CPU Ridge probe. Preparation is not actual-run authority.

Same sealed frozen bank, no MRI/encoder/GPU/Gold model selection. All supervised
transforms fit on their own weak gradient IDs; holdout and Gold never fit them.
"""
import time
_PYTHON_START = time.monotonic()
import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orthofoundation_inference import contracts as C

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
RECIPE = {'experiment': 'orthofoundation_slot_centered_ridge_v1',
    'pooling': 'mean_across_K4_per_observed_slot',
    'standardization': 'gradient_observed_only_per_slot_per_dimension',
    'missing_slots': 'zero_with_six_mask_indicators', 'std_floor': .001,
    'ridge_alpha': 1000., 'ridge_solver': 'cholesky', 'fit_intercept': True,
    'target': 'logit_soft_probability', 'target_clip_epsilon': .0001,
    'prediction': 'sigmoid_once', 'weak_holdout_fold': 0, 'seed': 42,
    'blas_threads': 2, 'maximum_seconds': 300, 'maximum_memory_bytes': 2147483648}
CONTRACTS_SHA = '712d3ce2b9a32d74501afe5bc4c5dbc3724a8e6f9f5a1577248bdca6f56113c2'
GLOBAL_SHA = 'b2a87794d768ca3f6ee3968345f840f1028cff004708a2b3e0cfdfcfb2c15e60'
ROOT_RECIPE_SHA = '722c24729272562f0e0a639e740b8f4eaadf6f59641a0ef393bd6a4953b004e5'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def snapshot(path, pin=None, limit=64 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit,
            'Required bounded ordinary input missing or linked')
    data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest()
    require(len(data) <= limit, 'Input grew beyond bounded size')
    if pin is not None:
        C.require_pin(pin, 'Ridge input'); require(sha == pin, 'Externally pinned input snapshot differs')
    return data, sha


def peak_memory_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == 'darwin' else rss * 1024)


def budget(start, recipe):
    require(time.monotonic() - start < recipe['maximum_seconds'], 'Ridge Python budget exhausted')
    require(peak_memory_bytes() < recipe['maximum_memory_bytes'], 'Ridge memory budget exhausted')


def load_recipe(path):
    data, sha = snapshot(path, limit=64 * 1024)
    recipe = yaml.safe_load(data)
    require(recipe == RECIPE, 'Only the predeclared fixed EXP-OF-003 recipe is allowed')
    return recipe, sha


def csv_snapshot(path, pin, index):
    data, sha = snapshot(path, pin)
    header = next(csv.reader(StringIO(data.decode('utf-8'))))
    require(len(header) == len(set(header)), 'Duplicate CSV columns')
    frame = pd.read_csv(BytesIO(data), dtype={header[0]: str})
    frame = frame.set_index(frame.columns[0] if index == 'first' else index)
    require(frame.index.is_unique and frame.index.notna().all()
            and [column for column in frame.columns if column in LABELS] == LABELS,
            'Study uniqueness or exact label order differs')
    return frame, sha


def split_ids(labels, official, fold=0):
    require(len(labels) == len(official) == 4407 and set(labels.index) == set(official.index),
            'Expected exact all4407 official study IDs')
    require(labels.is_gold.isin([0, 1]).all() and labels.loc[labels.is_gold.eq(0), 'fold'].isin(range(5)).all(),
            'Gold flags or weak fold values invalid')
    gold = official.index[official[LABELS].notna().all(axis=1)].tolist()
    weak = labels.index[labels.is_gold.eq(0)].tolist()
    holdout = labels.index[labels.is_gold.eq(0) & labels.fold.eq(fold)].tolist()
    training = [uid for uid in weak if uid not in set(holdout)]
    require(set(gold) == set(labels.index[labels.is_gold.eq(1)])
            and (len(weak), len(gold), len(holdout), len(training)) == (4349, 58, 870, 3479)
            and not set(training) & set(holdout) and not set(weak) & set(gold),
            'Weak/holdout/Gold coverage or exclusion differs')
    y = labels[LABELS].to_numpy(dtype=np.float64)
    require(np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all()
            and official.loc[gold, LABELS].isin([0, 1]).all().all(), 'Finite weak probabilities/binary Gold required')
    return weak, gold, holdout, training


def verify_root_seal(path, seal_pin, bank_pin):
    C.require_pin(seal_pin, 'completed training seal'); C.require_pin(bank_pin, 'complete source bank')
    data, sha = snapshot(path, seal_pin, 1024 * 1024); seal = json.loads(data)
    require(seal.get('schema') == 'ROOT_http_v2_attention_full_bank_seal_v1'
            and seal.get('ROOT_actual_full_merge_PASS') is True
            and seal.get('feature_identity') == C.TRAINING_IDENTITY
            and seal.get('validation_helper_sha256') == C.ROOT_SEAL_HELPER_SHA
            and seal.get('source_kernel') == 'alanchoo/rsna-knee-ortho-http-merge-heads-v2'
            and seal.get('kernel_version_origin_verified_separately_by_ROOT') is True
            and type(seal.get('root_confirmed_kernel_version')) is int
            and seal['root_confirmed_kernel_version'] > 0,
            'Independent complete ROOT attention-bank seal required')
    require(seal.get('source_bank_sha256') == seal.get('feature_bank_receipt', {}).get('sha256') == bank_pin
            and seal.get('global_metadata_manifest_sha256') == GLOBAL_SHA
            and seal.get('downloaded_source8_sha256') == C.SOURCE8
            and seal.get('approved_config_yaml_sha256') == C.CONFIG_SHA
            and seal.get('assets_manifest_sha256') == C.ASSETS_SHA,
            'Externally pinned ROOT bank/global/science/encoder provenance differs')
    require(seal.get('counts') == {'studies': 4407, 'weak': 4349, 'Gold': 58, 'CV_gradient': 3479,
        'weak_holdout': 870, 'production_gradient': 4349, 'selected_slots': 21334, 'images_k4': 85336}
        and seal.get('both_heads') == {'epochs': 12, 'CV_optimizer_updates': 660, 'production_optimizer_updates': 816,
        'Gold_gradient_rows': 0, 'Gold_checkpoint_selection': False}, 'ROOT complete bank/Gold policy differs')
    return seal, sha


def read_bank(ids, directory, bank_pin, identity=C.TRAINING_IDENTITY, start=None, recipe=RECIPE):
    """Mean-pool each immutable verified FP16 snapshot in FP32, keep memory bounded."""
    C.require_pin(bank_pin, 'source bank')
    directory = Path(directory); ids = list(ids)
    require(len(ids) == len(set(ids)) and all(re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', uid) for uid in ids),
            'Unique namespace-safe study IDs required')
    require({path.stem for path in directory.glob('*.npz')} == set(ids), 'Exact feature-file coverage differs')
    pooled = np.zeros((len(ids), 6, 1024), dtype=np.float32)
    masks = np.zeros((len(ids), 6), dtype=bool); digest = hashlib.sha256(); total = slots = 0
    for index, uid in enumerate(ids):
        if start is not None: budget(start, recipe)
        data, sha = snapshot(directory / (uid + '.npz'), limit=1024 * 1024)
        # Hash order is independently fixed; digest is checked before any fit.
        with np.load(BytesIO(data), allow_pickle=False) as row:
            f, m, p = row['features'], row['mask'], row['positions']
            require(row['study'].item() == uid and row['identity'].item() == identity
                and row['global_metadata_manifest_sha256'].item() == GLOBAL_SHA
                and row['orchestrator_sha256'].item() == C.SOURCE8['sharded_frozen_extract.py'], 'Feature science/UID identity differs')
            require(f.dtype == np.float16 and f.shape == (6, 4, 1024) and m.dtype == np.bool_ and m.shape == (6,)
                and m.any() and p.dtype == np.float32 and p.shape == (6, 4)
                and np.isfinite(f).all() and np.isfinite(p).all() and (f[~m] == 0).all()
                and (p[~m] == 0).all() and ((p >= 0) & (p <= 1)).all()
                and [bool(sr) for sr in row['selected_series']] == m.tolist(), 'Invalid frozen feature geometry/values/masks')
            pooled[index] = f.astype(np.float32).mean(axis=1); masks[index] = m; slots += int(m.sum())
        total += len(data)
        digest.update(f'{uid}.npz\t{len(data)}\t{sha}\n'.encode())
    require(ids == sorted(ids), 'Bank must use canonical sorted study order')
    require(digest.hexdigest() == bank_pin, 'Full bank differs from independent external ROOT pin; no fit')
    return pooled, masks, {'studies': len(ids), 'bytes': total, 'sha256': digest.hexdigest(),
        'selected_slots': slots, 'images_K4': 4 * slots,
        'digest_format': 'sorted filename<TAB>bytes<TAB>sha256<LF>'}


class SlotCenterScale:
    def __init__(self, std_floor=.001):
        self.std_floor = std_floor

    def fit(self, pooled, masks, ids, forbidden_ids):
        require(not set(ids) & set(forbidden_ids), 'Holdout/Gold cannot fit slot statistics')
        require(pooled.ndim == 3 and pooled.shape[1] == 6 and masks.shape == pooled.shape[:2]
            and masks.dtype == np.bool_ and len(ids) == len(pooled) and len(ids) == len(set(ids))
            and np.isfinite(pooled).all() and (pooled[~masks] == 0).all(), 'Invalid scaler gradient input')
        self.mean = np.zeros(pooled.shape[1:], dtype=np.float64)
        self.std = np.full(pooled.shape[1:], self.std_floor, dtype=np.float64)
        self.counts = masks.sum(axis=0).astype(int)
        for slot in range(6):
            observed = pooled[masks[:, slot], slot]
            if len(observed):
                self.mean[slot] = observed.mean(axis=0, dtype=np.float64)
                self.std[slot] = np.maximum(observed.std(axis=0, dtype=np.float64, ddof=0), self.std_floor)
        self.gradient_ids_sha256 = hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
        return self

    def transform(self, pooled, masks):
        require(hasattr(self, 'mean') and np.isfinite(pooled).all() and masks.dtype == np.bool_
            and pooled.shape[1:] == self.mean.shape and masks.shape == pooled.shape[:2]
            and (pooled[~masks] == 0).all(), 'Invalid transform input')
        normalized = (pooled.astype(np.float64) - self.mean) / self.std
        normalized[~masks] = 0
        result = np.concatenate([normalized.reshape(len(pooled), -1), masks.astype(np.float64)], axis=1)
        require(np.isfinite(result).all(), 'Nonfinite standardized features')
        return result


def target_logits(probabilities, eps=.0001):
    require(probabilities.ndim == 2 and np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all(), 'Finite target probabilities required')
    p = np.clip(probabilities.astype(np.float64), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def probabilities_from_logits(logits):
    require(np.isfinite(logits).all(), 'Nonfinite Ridge prediction logits')
    value = expit(logits)
    require(np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all(), 'Nonfinite probability output')
    return value


def score(target, prediction, kind):
    require(target.shape == prediction.shape and np.isfinite(prediction).all(), 'Metric alignment/finite values differ')
    values = [float(roc_auc_score(target[:, index], prediction[:, index]))
        if np.unique(target[:, index]).size == 2 else None for index in range(len(LABELS))]
    return {'n': len(target), 'macro_auc': float(np.mean([v for v in values if v is not None])),
        'per_label_auc': dict(zip(LABELS, values)), 'target_kind': kind,
        'undefined_classes': sum(v is None for v in values)}


def fit_probe(pooled, masks, all_ids, gradient_ids, forbidden_ids, targets, recipe):
    require(not set(gradient_ids) & set(forbidden_ids), 'Gold/holdout gradients forbidden')
    by_id = {uid: index for index, uid in enumerate(all_ids)}
    selected = [by_id[uid] for uid in gradient_ids]
    scaler = SlotCenterScale(recipe['std_floor']).fit(pooled[selected], masks[selected], gradient_ids, forbidden_ids)
    design = scaler.transform(pooled[selected], masks[selected])
    model = Ridge(alpha=recipe['ridge_alpha'], fit_intercept=True, solver='cholesky')
    model.fit(design, target_logits(targets, recipe['target_clip_epsilon']))
    require(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(), 'Nonfinite closed-form Ridge weights')
    return scaler, model


def run(args, start=_PYTHON_START):
    lexical_output = Path(args.output_dir)
    require(not lexical_output.is_symlink(), 'Linked output directory refused')
    C.require_pin(args.root_seal_sha256, 'completed training seal'); C.require_pin(args.source_bank_sha256, 'complete source bank')
    require(args.execute_reviewed_cpu_probe is True, 'Prepared probe requires an explicit reviewed CPU execution request')
    _, contracts_sha = snapshot(ROOT / 'src/orthofoundation_inference/contracts.py', CONTRACTS_SHA)
    _, source_pins, base_config_sha = C.verify_frozen_sources(ROOT)
    recipe, recipe_sha = load_recipe(args.config); budget(start, recipe)
    declared_bytes, declared_sha = snapshot(ROOT / 'experiments/orthofoundation_ridge_probe/ROOT_RECIPE.json', ROOT_RECIPE_SHA)
    declared = json.loads(declared_bytes)
    require(declared.get('schema') == 'predeclared_Ortho_slot_centered_Ridge_probe_v1'
        and declared.get('source_bank_sha256') == args.source_bank_sha256
        and declared.get('ROOT_source_seal_sha256') == args.root_seal_sha256
        and declared.get('Ridge') == {'alpha': 1000, 'fit_intercept': True, 'solver': 'cholesky'}
        and declared.get('Gold_gradient_or_model_selection') == 0 and declared.get('hyperparameter_search') is False
        and declared.get('training_executed') is False, 'ROOT pre-fit recipe/pins differ')
    seal, seal_sha = verify_root_seal(args.root_seal, args.root_seal_sha256, args.source_bank_sha256)
    global_dir = Path(args.global_metadata)
    manifest_bytes, _ = snapshot(global_dir / 'global_metadata_manifest.json', GLOBAL_SHA, 1024 * 1024)
    members = {row['path']: row for row in json.loads(manifest_bytes)['members']}
    labels, labels_sha = csv_snapshot(global_dir / 'labels_v0.csv', members['labels_v0.csv']['sha256'], 'first')
    official, official_sha = csv_snapshot(global_dir / 'train.csv', members['train.csv']['sha256'], 'StudyInstanceUID')
    weak, gold, holdout, gradient = split_ids(labels, official)
    ids = sorted(labels.index); bank = Path(args.feature_bank).resolve()
    output = lexical_output.resolve()
    require(output != bank and output not in bank.parents and bank not in output.parents
            and (not output.exists() or not any(output.iterdir())), 'Separate empty private output directory required')
    pooled, masks, bank_receipt = read_bank(ids, bank, args.source_bank_sha256, start=start, recipe=recipe)
    require(bank_receipt['studies'] == 4407 and bank_receipt['selected_slots'] == 21334
            and bank_receipt['images_K4'] == 85336 and bank_receipt['bytes'] == seal['feature_bank_receipt']['bytes'],
            'Complete sealed bank coverage differs')
    output.mkdir(parents=True, exist_ok=True)
    by_id = {uid: index for index, uid in enumerate(ids)}; reports = {}; threads_observed = []
    published = []
    try:
      with threadpool_limits(limits=2):
        threads_observed = [{'internal_api': row['internal_api'], 'num_threads': row['num_threads']}
            for row in threadpool_info()]
        require(all(row['num_threads'] <= 2 for row in threads_observed), 'BLAS thread bound differs')
        with tempfile.TemporaryDirectory(prefix='.ridge_staged_', dir=output) as stage_name:
            stage = Path(stage_name)
            for name, training_ids, forbidden, evaluated in (
                ('weak_holdout', gradient, gold + holdout, holdout),
                ('production', weak, gold, gold)):
                budget(start, recipe); phase = time.monotonic()
                scaler, model = fit_probe(pooled, masks, ids, training_ids, forbidden,
                    labels.loc[training_ids, LABELS].to_numpy(dtype=np.float64), recipe)
                budget(start, recipe)
                test_indices = [by_id[uid] for uid in evaluated]
                logits = model.predict(scaler.transform(pooled[test_indices], masks[test_indices]))
                predictions = probabilities_from_logits(logits); budget(start, recipe)
                target = ((labels.loc[evaluated, LABELS].to_numpy() >= .5).astype(int)
                    if name == 'weak_holdout' else official.loc[evaluated, LABELS].to_numpy(dtype=int))
                metrics = score(target, predictions, 'thresholded weak report consensus' if name == 'weak_holdout'
                    else 'official58 expert Gold; diagnostic only')
                pd.DataFrame(predictions, index=pd.Index(evaluated, name='StudyInstanceUID'), columns=LABELS).to_csv(
                    stage / (name + '_predictions.csv'))
                np.savez_compressed(stage / (name + '_ridge.npz'), mean=scaler.mean, std=scaler.std,
                    observed_slot_counts=scaler.counts, coef=model.coef_, intercept=model.intercept_,
                    labels=np.asarray(LABELS), gradient_ids_sha256=scaler.gradient_ids_sha256)
                reports[name] = {'n_gradient_studies': len(training_ids), 'gradient_ids_sha256': scaler.gradient_ids_sha256,
                    'observed_gradient_slot_counts': scaler.counts.tolist(), 'Gold_gradient_rows': 0,
                    'scaler_fit_on_gradient_only': True, 'Gold_model_selection': False, 'parameter_search': False,
                    'closed_form_fits': 1, 'metrics': metrics, 'seconds': time.monotonic() - phase}
                del scaler, model
                budget(start, recipe)
            exported = {path.name: {'bytes': path.stat().st_size, 'sha256': snapshot(path)[1]}
                for path in stage.iterdir()}
            report = {'schema': 'Ortho_slot_centered_Ridge_probe_v1', 'completed': True, 'experiment': recipe['experiment'],
                'recipe': recipe, 'recipe_yaml_sha256': recipe_sha, 'probe_source_sha256': snapshot(__file__)[1],
                'ROOT_pre_fit_recipe_sha256': declared_sha,
                'ROOT_completed_attention_seal_sha256': seal_sha, 'source_bank_receipt': bank_receipt,
                'source_training_feature_identity': C.TRAINING_IDENTITY,
                'source8_sha256': source_pins, 'parent_attention_config_sha256': base_config_sha,
                'inference_admission_contracts_sha256': contracts_sha,
                'labels_snapshot_sha256': labels_sha, 'official_train_snapshot_sha256': official_sha,
                'label_order': LABELS, 'heads': reports, 'exported_private_files': exported,
                'CPU_only': True, 'encoder_loads': 0, 'GPU_calls': 0, 'parameter_search': False,
                'Gold_used_for_transform_or_gradient_or_model_selection': 0,
                'BLAS_thread_receipts': threads_observed, 'peak_memory_bytes': peak_memory_bytes(),
                'Python_seconds_before_report_export': time.monotonic() - start,
                'timer_boundary': 'First Python statement including imports, validation, fits and exports; external parent300s required.',
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
                'limits': 'Diagnostic fixed linear probe; no new clinical truth, test prediction, blend/submission or leaderboard gain claim.'}
            (stage / 'probe_report.json').write_text(json.dumps(report, indent=2) + '\n'); budget(start, recipe)
            for path in sorted(stage.iterdir(), key=lambda path: (path.name == 'probe_report.json', path.name)):
                budget(start, recipe); target = output / path.name
                info = path.stat(); published.append((target, info.st_dev, info.st_ino))
                os.link(path, target)  # Exclusive publication; report last.
                budget(start, recipe)
            budget(start, recipe)
      budget(start, recipe)  # Includes temporary cleanup and thread restoration.
    except Exception:
        for path, device, inode in published:
            if path.exists():
                info = path.stat()
                if (info.st_dev, info.st_ino) == (device, inode): path.unlink()
        raise
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--feature-bank', required=True); ap.add_argument('--source-bank-sha256', required=True)
    ap.add_argument('--root-seal', required=True); ap.add_argument('--root-seal-sha256', required=True)
    ap.add_argument('--global-metadata', required=True); ap.add_argument('--output-dir', required=True)
    ap.add_argument('--config', default=str(ROOT / 'configs/orthofoundation_slot_centered_ridge.yaml'))
    ap.add_argument('--execute-reviewed-cpu-probe', action='store_true')
    args = ap.parse_args()
    try:
        report = run(args)
        print(json.dumps({'completed': True, 'weak_holdout_macro_auc': report['heads']['weak_holdout']['metrics']['macro_auc'],
            'Gold58_diagnostic_macro_auc': report['heads']['production']['metrics']['macro_auc'],
            'peak_memory_bytes': report['peak_memory_bytes']}), flush=True)
    except Exception as error:
        print(json.dumps({'completed': False, 'failure_type': type(error).__name__, 'clinical_identifiers_suppressed': True}), flush=True)
        raise SystemExit(1)


if __name__ == '__main__': main()

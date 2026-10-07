#!/usr/bin/env python3
"""Aggregate Gold58 diagnostics; no fitting, encoder, IDs or raw prediction export.

The fixed reference is one two-epoch ConvNeXt EMA reader, not the full .944
ensemble. Ortho checkpoints/source banks must be verified separately by root.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
          'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
ID = 'StudyInstanceUID'
BASELINE_SHA256 = 'c9ef127918f1e7758853327ed1d0d2edb4ae3be38eb05935f60e4390585a7cf5'
GOLD_IDS_SHA256 = '3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea'
BASELINE_CHECKPOINT_SHA256 = '2557782563a0b416cf85c7bf98a0caaf685dc4f991409b184b8b118730dfa25a'
DOCUMENTED_BASELINE_AUC = .9111602412161779
SEED = 42
THRESHOLD = .5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_snapshot(path):
    """One immutable byte snapshot binds the receipt hash to parsed values."""
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def sigmoid_once(logits):
    """Stable for finite extreme logits; no sigmoid is applied to Ortho inputs."""
    result = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    result[positive] = 1 / (1 + np.exp(-logits[positive]))
    exp = np.exp(logits[~positive])
    result[~positive] = exp / (1 + exp)
    return result


def read_table(data, official=False):
    table = pd.read_csv(BytesIO(data), dtype={ID: str})
    require(ID in table and table[ID].notna().all() and table[ID].is_unique, 'Missing or duplicate study IDs')
    if official:
        require([name for name in table.columns if name in LABELS] == LABELS,
                'Official label order must match the twelve competition labels')
        labels = table[LABELS]
        require((labels.isna().sum(axis=1).isin([0, 12])).all(), 'Partially missing official labels are unsupported')
        table = table.loc[labels.notna().all(axis=1), [ID, *LABELS]]
    else:
        require(list(table.columns) == [ID, *LABELS], 'Prediction columns or label order differ')
    require(len(table) == 58, 'Require exactly 58 unique Gold studies')
    values = table[LABELS].to_numpy(dtype=np.float64)
    require(np.isfinite(values).all(), 'Nonfinite labels or predictions')
    if official:
        require(np.isin(values, [0, 1]).all(), 'Official Gold labels must be binary')
    return table.set_index(ID)


def load_inputs(official_path, baseline_path, candidates, baseline_sha=BASELINE_SHA256,
                gold_ids_sha=GOLD_IDS_SHA256, candidate_pins=None):
    baseline_data, baseline_actual_sha = csv_snapshot(baseline_path)
    require(baseline_actual_sha == baseline_sha, 'Baseline raw-logit CSV SHA differs from external pin')
    official_data, official_sha = csv_snapshot(official_path)
    official = read_table(official_data, official=True)
    ids = sorted(official.index)
    require(hashlib.sha256('\n'.join(ids).encode()).hexdigest() == gold_ids_sha, 'Official Gold study set differs from sealed Gold58')
    baseline = read_table(baseline_data)
    require(set(baseline.index) == set(ids), 'Baseline study IDs differ from official Gold58')
    require(bool(candidates), 'At least one independently verified Ortho probability CSV is required')
    pins = candidate_pins or {}
    require(set(pins).issubset(candidates), 'Prediction SHA pin supplied for unknown model')
    logits = baseline.loc[ids, LABELS].to_numpy(dtype=np.float64)
    scores = {'convnext_reader': logits}
    probabilities = {'convnext_reader': sigmoid_once(logits)}
    receipts = {'official_labels_sha256': official_sha, 'baseline_raw_logits_sha256': baseline_actual_sha,
                'Gold_ids_sha256': gold_ids_sha, 'ortho_probability_csv_sha256': {}}
    for name, path in candidates.items():
        require(re.fullmatch(r'[a-z][a-z0-9_]{0,31}', name) is not None
                and name not in {'convnext_reader', 'fixed_50_50_blend'}, 'Invalid or reserved model name')
        data, actual_sha = csv_snapshot(path)
        if name in pins:
            require(actual_sha == pins[name], 'Ortho probability CSV differs from supplied verified pin')
        table = read_table(data)
        require(set(table.index) == set(ids), 'Ortho study IDs differ from official Gold58')
        values = table.loc[ids, LABELS].to_numpy(dtype=np.float64)
        require(((values >= 0) & (values <= 1)).all(), 'Ortho inputs must be probabilities in [0,1]')
        scores[name] = values
        probabilities[name] = values
        receipts['ortho_probability_csv_sha256'][name] = actual_sha
    return official.loc[ids, LABELS].to_numpy(dtype=int), scores, probabilities, receipts


def bootstrap_auc(targets, scores, repeats):
    """Paired study bootstrap via exact weighted positive/negative pair counts."""
    require(type(repeats) is int and 1 <= repeats <= 100000, 'Bootstrap repeats must be 1..100000')
    n = len(targets)
    draws = np.random.default_rng(SEED).integers(0, n, size=(repeats, n))
    counts = np.zeros((repeats, n), dtype=np.float64)
    np.add.at(counts, (np.arange(repeats)[:, None], draws), 1)
    values = {name: np.full((repeats, 12), np.nan) for name in scores}
    defined = np.zeros((repeats, 12), dtype=bool)
    for label in range(12):
        positive, negative = targets[:, label] == 1, targets[:, label] == 0
        wp, wn = counts[:, positive], counts[:, negative]
        denominator = wp.sum(axis=1) * wn.sum(axis=1)
        valid = denominator > 0
        defined[:, label] = valid
        for name, prediction in scores.items():
            a, b = prediction[positive, label], prediction[negative, label]
            comparisons = (a[:, None] > b).astype(float) + .5 * (a[:, None] == b)
            numerator = ((wp @ comparisons) * wn).sum(axis=1)
            values[name][valid, label] = numerator[valid] / denominator[valid]
    return values, defined


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {'low': float(np.quantile(finite, .025)) if len(finite) else None,
            'high': float(np.quantile(finite, .975)) if len(finite) else None,
            'defined_repeats': int(len(finite)), 'dropped_undefined_repeats': int(len(values) - len(finite))}


def rank_correlation(a, b):
    a, b = rankdata(a), rankdata(b)
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1)) if np.ptp(a) and np.ptp(b) else None


def overlap(a, b):
    union = int((a | b).sum())
    return {'both_wrong': int((a & b).sum()), 'baseline_only_wrong': int((a & ~b).sum()),
            'candidate_only_wrong': int((~a & b).sum()), 'both_correct': int((~a & ~b).sum()),
            'error_jaccard': float((a & b).sum() / union) if union else None}


def summarize(targets, scores, probabilities, receipts, repeats=2000, predeclared_blend=None):
    require(targets.shape == (58, 12) and np.isfinite(targets).all() and np.isin(targets, [0, 1]).all(),
            'Diagnostics require the binary Gold58 by twelve-label matrix')
    require(set(scores) == set(probabilities) and 'convnext_reader' in scores, 'Score/probability model keys differ')
    for name in scores:
        require(scores[name].shape == (58, 12) and probabilities[name].shape == (58, 12)
                and np.isfinite(scores[name]).all() and np.isfinite(probabilities[name]).all()
                and ((probabilities[name] >= 0) & (probabilities[name] <= 1)).all(), 'Invalid diagnostic score/probability matrix')
    scores, probabilities = dict(scores), dict(probabilities)
    if predeclared_blend is not None:
        require(predeclared_blend in probabilities and predeclared_blend != 'convnext_reader', 'Unknown predeclared blend source')
        blend = .5 * probabilities['convnext_reader'] + .5 * probabilities[predeclared_blend]
        scores['fixed_50_50_blend'] = probabilities['fixed_50_50_blend'] = blend
    boot, defined = bootstrap_auc(targets, scores, repeats)
    macro_defined = defined.all(axis=1)
    models = {}
    baseline_errors = (probabilities['convnext_reader'] >= THRESHOLD) != targets
    for name, prediction in scores.items():
        per_label = {}
        errors = (probabilities[name] >= THRESHOLD) != targets
        for j, label in enumerate(LABELS):
            positive = int(targets[:, j].sum())
            auc = float(roc_auc_score(targets[:, j], prediction[:, j])) if 0 < positive < 58 else None
            reference = scores['convnext_reader'][:, j]
            base_auc = float(roc_auc_score(targets[:, j], reference)) if auc is not None else None
            per_label[label] = {'positive': positive, 'negative': 58 - positive, 'auc': auc,
                'auc_95pct_bootstrap': interval(boot[name][:, j]),
                'auc_delta_vs_reader': auc - base_auc if auc is not None else None,
                'delta_95pct_paired_bootstrap': interval(boot[name][:, j] - boot['convnext_reader'][:, j]),
                'spearman_rank_vs_reader': rank_correlation(reference, prediction[:, j]),
                'errors_at_0_5': int(errors[:, j].sum()), 'error_overlap_vs_reader': overlap(baseline_errors[:, j], errors[:, j])}
        observed = [row['auc'] for row in per_label.values()]
        macro = float(np.mean(observed)) if all(value is not None for value in observed) else None
        macro_values = np.full(repeats, np.nan)
        macro_delta = np.full(repeats, np.nan)
        macro_values[macro_defined] = boot[name][macro_defined].mean(axis=1)
        macro_delta[macro_defined] = (boot[name][macro_defined] - boot['convnext_reader'][macro_defined]).mean(axis=1)
        models[name] = {'macro_auc_all_12': macro, 'macro_auc_95pct_bootstrap': interval(macro_values),
            'macro_delta_vs_reader': None if macro is None else float(np.mean([row['auc_delta_vs_reader'] for row in per_label.values()])),
            'macro_delta_95pct_paired_bootstrap': interval(macro_delta), 'per_label': per_label,
            'error_overlap_all_696_decisions': overlap(baseline_errors, errors),
            'error_overlap_studies_with_any_error': overlap(baseline_errors.any(axis=1), errors.any(axis=1))}
    return {'schema': 'aggregate_Gold58_diagnostics_v1', 'n_studies': 58, 'labels_order': LABELS,
        'reference': {'scope': 'Single fixed-final two-epoch ConvNeXt EMA reader; full ensemble public .944 is a separate observation',
                      'documented_checkpoint_sha256': BASELINE_CHECKPOINT_SHA256, 'clean_OOF_claim': False},
        'source_receipts': receipts, 'baseline_sigmoid_applications_for_probability_operations': 1,
        'baseline_AUC_and_rank_use': 'Raw logits preserve ordering, including extreme sigmoid saturation',
        'error_threshold_predeclared': THRESHOLD,
        'bootstrap': {'method': 'Paired study resampling with replacement; same draws for every model and label',
            'seed': SEED, 'repeats': repeats, 'interval': 'Percentile 95%; diagnostic sampling variability of Gold58',
            'undefined_rule': 'Per-label repeats with only one class are dropped; macro repeat requires all twelve classes defined',
            'undefined_repeats_per_label': {label: int((~defined[:, j]).sum()) for j, label in enumerate(LABELS)},
            'undefined_macro_repeats': int((~macro_defined).sum())},
        'predeclared_blend': {'source': predeclared_blend, 'weights': [.5, .5] if predeclared_blend else None,
            'recipe_supplied_before_evaluation': predeclared_blend is not None, 'weight_search': False},
        'models': models, 'limits': 'Gold diagnostic only; no checkpoint selection, weight fitting, clean generalization or leaderboard-gain claim. Root must separately verify Ortho completed training/checkpoints/full bank.'}


def plot_report(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models = report['models']; names = list(models); y = np.arange(12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True, layout='constrained')
    colors = ['#6b7280', '#0f766e', '#2563eb', '#b45309', '#7c3aed']
    for index, name in enumerate(names):
        offset = (index - (len(names) - 1) / 2) * min(.13, .6 / len(names))
        rows = [models[name]['per_label'][label] for label in LABELS]
        axes[0].scatter([row['auc'] for row in rows], y + offset, color=colors[index % len(colors)], label=name, s=28)
        if name != 'convnext_reader':
            for j, row in enumerate(rows):
                ci = row['delta_95pct_paired_bootstrap']; value = row['auc_delta_vs_reader']
                if value is not None:
                    axes[1].scatter(value, y[j] + offset, color=colors[index % len(colors)], s=28)
                    if ci['low'] is not None: axes[1].plot([ci['low'], ci['high']], [y[j] + offset] * 2, color=colors[index % len(colors)], linewidth=1.5)
    axes[0].set(yticks=y, yticklabels=LABELS, xlabel='Gold AUC', xlim=(0, 1))
    axes[0].invert_yaxis(); axes[0].legend(fontsize=8, loc='lower left')
    axes[1].axvline(0, color='#9ca3af', linewidth=1)
    axes[1].set_xlabel('Δ AUC vs single ConvNeXt reader · paired 95% bootstrap')
    for axis in axes: axis.grid(axis='x', alpha=.15)
    fig.suptitle('Gold58 diagnostics · fixed models and recipe · no leaderboard or clean OOF claim', fontsize=12)
    fig.savefig(path, dpi=180); plt.close(fig)


def named_values(values):
    result = {}
    for value in values:
        name, separator, path = value.partition('=')
        require(separator and name and path and name not in result, 'Use unique NAME=VALUE arguments')
        result[name] = path
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-labels', type=Path, required=True)
    parser.add_argument('--baseline-logits', type=Path, required=True)
    parser.add_argument('--ortho-probabilities', action='append', required=True, metavar='NAME=CSV')
    parser.add_argument('--ortho-sha256', action='append', default=[], metavar='NAME=SHA', help='Optional independently verified CSV pins')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-repeats', type=int, default=2000)
    parser.add_argument('--predeclared-blend', metavar='NAME', help='One fixed 50:50 probability blend, declared before inspecting Gold results')
    args = parser.parse_args()
    labels, scores, probabilities, receipts = load_inputs(args.official_labels, args.baseline_logits,
        named_values(args.ortho_probabilities), candidate_pins=named_values(args.ortho_sha256))
    report = summarize(labels, scores, probabilities, receipts, args.bootstrap_repeats, args.predeclared_blend)
    baseline_auc = report['models']['convnext_reader']['macro_auc_all_12']
    require(baseline_auc is not None and np.isclose(baseline_auc, DOCUMENTED_BASELINE_AUC, atol=1e-12, rtol=0),
            'Pinned reader AUC differs from documented official-label reference')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_report(report, args.output_dir / 'Gold_AUC_diagnostics.png')
    (args.output_dir / 'Gold_diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'aggregate_only': True, 'n_Gold': 58, 'models': list(report['models']),
                      'output_files': ['Gold_diagnostics.json', 'Gold_AUC_diagnostics.png']}))


if __name__ == '__main__': main()

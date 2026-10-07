"""Canonical test DICOM -> frozen CLS -> admitted production attention head."""
from contextlib import nullcontext
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from convnext_reader.preprocess import load_series
from orthofoundation.data import LABELS, PLANES, SliceStudyDataset, build_table, ids_hash, make_views
from orthofoundation.encoder import encode, load_encoder
from orthofoundation.train import predict
from .contracts import ASSETS_SHA, CONFIG_SHA, ENCODER_BYTES, ENCODER_SHA, ROOT, require, snapshot


def read_metadata(data_dir, split):
    require(re.fullmatch(r'[a-z][a-z0-9_]*', split) is not None, 'Invalid explicit data split')
    root = Path(data_dir)
    study_bytes, study_sha = snapshot(root / f'{split}.csv')
    series_bytes, series_sha = snapshot(root / f'{split}_series.csv')
    studies = pd.read_csv(BytesIO(study_bytes), dtype={'StudyInstanceUID': str})
    series = pd.read_csv(BytesIO(series_bytes), dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    require('StudyInstanceUID' in studies and studies.StudyInstanceUID.is_unique and studies.StudyInstanceUID.notna().all()
            and len(studies) > 0, 'Study manifest must contain unique nonmissing studies')
    ids = studies.StudyInstanceUID.tolist()
    require(all(re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', uid) is not None for uid in ids), 'Invalid study identifier for DICOM paths')
    require(set(['StudyInstanceUID', 'SeriesInstanceUID', 'Anatomical_Plane', 'Fat_Suppression']).issubset(series.columns),
            'Series metadata lacks canonical selection columns')
    series = series[series.StudyInstanceUID.isin(set(ids))].copy()
    require(series.SeriesInstanceUID.is_unique and series.SeriesInstanceUID.notna().all()
            and set(series.StudyInstanceUID) == set(ids), 'Requested test study/series metadata coverage differs')
    require(all(re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', uid) is not None for uid in series.SeriesInstanceUID), 'Invalid series identifier for DICOM paths')
    require(series.Anatomical_Plane.isin(PLANES).all() and series.Fat_Suppression.isin([0, 1]).all(),
            'Unsupported anatomical plane/fat-suppression metadata')
    return ids, series, {'test_study_manifest_sha256': study_sha, 'test_series_manifest_sha256': series_sha,
                        'test_study_ids_sha256': ids_hash(ids), 'n_test_studies': len(ids), 'n_eligible_test_series': len(series)}


class CanonicalVolumeCache:
    """Memoize every candidate of one study in owned temporary storage."""
    def __init__(self, rows, series_root, scratch, decoder=load_series):
        self.rows, self.series_root, self.scratch, self.decoder = rows, Path(series_root), Path(scratch), decoder
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.paths, self.counts = {}, {}
        self.n_bad_slices, self.unusable_series, self.decoded_series = 0, 0, 0
        self.volume_digest = hashlib.sha256()

    def decode(self, check_budget=lambda: None):
        for row in self.rows.sort_values('SeriesInstanceUID').itertuples():
            check_budget()
            path = self.series_root / row.StudyInstanceUID / row.SeriesInstanceUID
            try:
                volume, info = self.decoder(path, plane_name=row.Anatomical_Plane)
            except Exception:
                self.counts[row.SeriesInstanceUID] = 0; self.unusable_series += 1
                continue
            require(isinstance(volume, np.ndarray) and volume.dtype == np.uint8 and volume.ndim == 3
                    and volume.shape[1:] == (384, 384) and 1 <= len(volume) <= 64,
                    'Canonical decoder returned wrong uint8 geometry/capped depth')
            check_budget()
            self.n_bad_slices += int(info.get('n_bad', 0))
            if info.get('n_raw', len(volume)) == 0:
                self.counts[row.SeriesInstanceUID] = 0; self.unusable_series += 1
                continue
            self.decoded_series += 1
            self.counts[row.SeriesInstanceUID] = len(volume)
            volume_sha = hashlib.sha256(volume.tobytes(order='C')).hexdigest()
            self.volume_digest.update(f'{row.SeriesInstanceUID}\t{len(volume)}\t{volume_sha}\n'.encode())
            if len(volume) < 3:
                self.unusable_series += 1
                continue
            destination = self.scratch / (hashlib.sha256(row.SeriesInstanceUID.encode()).hexdigest() + '.npy')
            np.save(destination, volume, allow_pickle=False); self.paths[row.SeriesInstanceUID] = destination
        return self.counts

    def __call__(self, series):
        require(series in self.paths, 'Selected canonical volume was not memoized')
        return np.load(self.paths[series], mmap_mode='r', allow_pickle=False)


def study_sample(study, rows, cache):
    table = build_table(rows, cache.counts)
    require(any(table.get(study, [])), 'Requested study has no usable canonical MRI slot')
    return SliceStudyDataset([study], table, cache, 4)[0]


@torch.inference_mode()
def feature_tensors(encoder, raw, mask, positions, device='cuda', encode_fn=encode, check_budget=lambda: None):
    """Exactly training grouping/FP16 storage rounding, then FP32 head input."""
    require(raw.shape == (6, 4, 384, 384) and raw.dtype == torch.uint8
            and raw.device.type == mask.device.type == positions.device.type == 'cpu'
            and mask.shape == (6,) and mask.dtype == torch.bool and mask.any().item()
            and positions.shape == (6, 4) and positions.dtype == torch.float32
            and torch.isfinite(positions).all().item() and ((positions >= 0) & (positions <= 1)).all().item()
            and (positions[~mask] == 0).all().item(), 'Invalid sampled test geometry/mask/positions')
    require(device in ('cuda', 'cpu'), 'Unsupported explicit feature device')
    flat = raw[mask].flatten(0, 1); encoded = []
    for group in flat.split(16):
        check_budget()
        views = make_views(group.to(device), resolution=224, fov=.92)
        context = torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else nullcontext()
        with context:
            result = encode_fn(encoder, views, 'author_no_rope')
        require(result.shape == (len(group), 1024) and torch.isfinite(result).all().item(), 'Invalid finite frozen CLS output')
        encoded.append(result.float().cpu())
    with np.errstate(over='ignore', invalid='ignore'):
        values = torch.cat(encoded).numpy().astype(np.float16)
    require(np.isfinite(values).all(), 'CLS overflowed the training FP16 feature buffer')
    features = np.zeros((6, 4, 1024), dtype=np.float16)
    features[mask.numpy()] = values.reshape(int(mask.sum()), 4, 1024)
    return (torch.from_numpy(features.astype(np.float32))[None], mask[None].clone(), positions[None].clone())


def prepare_encoder_assets(assets, scratch):
    """Verified small source archive -> owned offline source; weights stay mapped."""
    assets, scratch = Path(assets), Path(scratch)
    manifest_data, _ = snapshot(assets / 'asset_manifest.json', ASSETS_SHA)
    manifest = json.loads(manifest_data)
    checkpoint = manifest['checkpoint']
    require(checkpoint['filename'] == 'OrthoFoundation-L.pth' and checkpoint['bytes'] == ENCODER_BYTES
            and checkpoint['sha256'] == ENCODER_SHA, 'Encoder release pin differs')
    source = manifest['dinov3_source']
    require(source['archive_root'] == 'dinov3' and source['commit'] == '6876159a11b4df116f30f667f8c9888617df0751',
            'Pinned DINOv3 source commit/root differs')
    archive, _ = snapshot(assets / source['filename'], source['sha256'])
    destination = scratch / 'encoder_source'; destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(archive), mode='r:gz') as tar:
        members = tar.getmembers(); files = []
        for member in members:
            name = Path(member.name)
            require(not name.is_absolute() and '..' not in name.parts and name.parts and name.parts[0] == 'dinov3'
                    and not member.issym() and not member.islnk() and (member.isfile() or member.isdir()), 'Unsafe pinned DINOv3 archive member')
            if member.isfile(): files.append(member.name)
        require(len(files) == len(set(files)), 'Duplicate source archive member')
        tar.extractall(destination, filter='data')
    require(not any(name == 'dinov3' or name.startswith('dinov3.') for name in sys.modules),
            'Fresh isolated inference process required before pinned DINOv3 import')
    weight_path = assets / checkpoint['filename']
    require(weight_path.is_file() and not weight_path.is_symlink() and weight_path.stat().st_size == ENCODER_BYTES,
            'Pinned encoder weight file missing/linked or wrong size')
    for license in manifest['licenses']:
        snapshot(assets / license['file'])
    return destination / 'dinov3', weight_path


def run_inference(data_dir, split, output_dir, head, admission, config, frozen_sources, assets, max_seconds=3600):
    require(torch.cuda.is_available(), 'CUDA required for actual encoder inference; no CPU encoder fallback')
    require(type(max_seconds) is int and max_seconds > 0, 'Positive inference budget required')
    started = time.monotonic()
    def check_budget(): require(time.monotonic() - started < max_seconds, 'Inference budget exhausted; no completed prediction claim')
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    require(not any((output / name).exists() for name in ('_ortho.csv', '_ortho_provenance.json')),
            'Independent Ortho output already exists; choose a fresh output directory')
    for source in (Path(data_dir).resolve(), Path(assets).resolve()):
        require(output.resolve() != source and source not in output.resolve().parents, 'Output must be outside immutable inference inputs')
    ids, series, metadata = read_metadata(data_dir, split)
    selection_digest, volumes_digest = hashlib.sha256(), hashlib.sha256()
    predictions, encoded_images, decoded_series, unusable_series, bad_slices = [], 0, 0, 0, 0
    with tempfile.TemporaryDirectory(prefix='owned_ortho_test_inference_') as temporary:
        scratch = Path(temporary)
        dinov3, checkpoint = prepare_encoder_assets(assets, scratch)
        check_budget()
        torch.cuda.reset_peak_memory_stats()
        encoder = load_encoder(dinov3, checkpoint, ENCODER_SHA, ENCODER_BYTES)
        head = head.to(device='cuda', dtype=torch.float32).eval().requires_grad_(False)
        by_study = {uid: rows for uid, rows in series.groupby('StudyInstanceUID', sort=False)}
        for study in ids:
            check_budget()
            with tempfile.TemporaryDirectory(prefix='canonical_study_', dir=scratch) as owned_study:
                cache = CanonicalVolumeCache(by_study[study], Path(data_dir) / f'{split}_series', owned_study)
                cache.decode(check_budget)
                _, raw, mask, positions, selected = study_sample(study, by_study[study], cache)
                tensors = feature_tensors(encoder, raw, mask, positions, check_budget=check_budget)
                with torch.autocast('cuda', enabled=False):
                    probabilities = predict(head, tensors, 'cuda', config['head_batch_size'])
                require(probabilities.shape == (1, 12) and np.isfinite(probabilities).all()
                        and ((probabilities >= 0) & (probabilities <= 1)).all(), 'Invalid production test probabilities')
                predictions.append(probabilities[0]); encoded_images += int(mask.sum()) * 4
                decoded_series += cache.decoded_series; unusable_series += cache.unusable_series; bad_slices += cache.n_bad_slices
                volumes_digest.update(f'{study}\t{cache.volume_digest.hexdigest()}\n'.encode())
                for slot, sr in enumerate(selected):
                    selection_digest.update(f'{study}:{slot}:{sr}:{positions[slot].tolist()}\n'.encode())
            check_budget()
        del encoder, head
    source_hashes = dict(frozen_sources)
    for path in sorted(Path(__file__).parent.glob('*.py')):
        _, source_hashes['orthofoundation_inference/' + path.name] = snapshot(path)
    _, source_hashes['run_orthofoundation_inference.py'] = snapshot(ROOT / 'scripts/run_orthofoundation_inference.py')
    provenance = {'schema': 'orthofoundation_test_inference_v1', **admission, **metadata,
        'training_config_yaml_sha256': CONFIG_SHA, 'encoder_checkpoint_sha256': ENCODER_SHA,
        'encoder_checkpoint_bytes': ENCODER_BYTES, 'encoder_assets_manifest_sha256': ASSETS_SHA,
        'source_sha256': source_hashes, 'test_selected_plan_sha256': selection_digest.hexdigest(),
        'canonical_test_volume_manifest_sha256': volumes_digest.hexdigest(),
        'feature_recipe': {'resolution': 224, 'literal_crop': 353, 'fov': .92, 'K': 4, 'n_slots': 6,
            'max_decoded_depth': 64, 'gray_RGB_replication': True, 'extraction_variant': 'author_no_rope',
            'encoder_autocast_dtype': 'float16', 'CLS_storage_rounding': 'float16', 'head_input_dtype': 'float32'},
        'training_global4407_cache_coverage_claimed_for_test': False,
        'test_scope': 'Exactly the explicit supplied split manifest; separate test identity, no labels or model selection',
        'automatic_blend_or_submission': False}
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    frame = pd.DataFrame(np.stack(predictions), index=pd.Index(ids, name='StudyInstanceUID'), columns=LABELS)
    require(frame.index.is_unique and frame.index.tolist() == ids and list(frame.columns) == LABELS, 'Final test study/label coverage differs')
    published = []
    try:
        check_budget()
        # Stage both small exports on the output filesystem. Exclusive hard links
        # publish without overwriting another result; failure rolls back our links.
        with tempfile.TemporaryDirectory(prefix='owned_ortho_export_', dir=output) as export:
            staged_csv, staged_report = Path(export) / '_ortho.csv', Path(export) / '_ortho_provenance.json'
            frame.to_csv(staged_csv)
            check_budget()
            _, csv_sha = snapshot(staged_csv)
            check_budget()
            report = {'inference_complete': True, 'test_feature_identity': identity, 'provenance': provenance,
                'n_test_studies': len(ids), 'decoded_series': decoded_series, 'unusable_series': unusable_series,
                'skipped_bad_slices': bad_slices, 'encoded_images': encoded_images, 'production_head_only': True,
                'owned_canonical_volume_storage_removed': True, 'seconds': time.monotonic() - started,
                'timer_boundary': 'run_inference after external production-head admission; includes asset verification/load, DICOM decode, features/head and staged CSV/hash; excludes imports/admission and Kaggle mount',
                'duration_recorded_at': 'Before staged receipt write/publication; all later export/publication/cleanup operations also checked against the same deadline',
                'budget_guard': 'Cooperative checks through final export/publication; future parent must enforce its owned-process hard timeout',
                'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(), 'prediction_csv_sha256': csv_sha,
                'labels_order': LABELS, 'sigmoid_applications': 1, 'automatic_blend_or_submission': False}
            staged_report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            check_budget()
            for source, target in [(staged_csv, output / '_ortho.csv'), (staged_report, output / '_ortho_provenance.json')]:
                check_budget()
                stat = source.stat(); published.append((target, stat.st_dev, stat.st_ino))
                os.link(source, target)  # Atomic exclusive creation: existing outputs are refused.
                check_budget()
        check_budget()
        return report
    except BaseException:
        for path, device, inode in reversed(published):
            try:
                stat = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.st_dev == device and stat.st_ino == inode:
                path.unlink()
        raise

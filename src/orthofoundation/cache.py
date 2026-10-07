"""OPTIONAL DRAFT: plain NPY, fresh-DICOM ZIP and exact-original-NPY ZIP storage.

ZIP handles open lazily and remain local to each DataLoader process. A series is
materialised individually (at most 64x384x384 uint8), never the whole shard.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


ZIP_RECEIPT_FILES = ('source_cache_summary.json', 'reference_cache_summary.json',
                     'cache_zip_manifest.json', 'direct_decode_receipt.json',
                     'direct_decode_metadata.json')


def _check(condition, message):
    if not condition:
        raise RuntimeError(message)


def _exact_repack_fingerprint(root, sm, meta, fp):
    """Validate existing original NPY -> ZIP proof; no fresh-decode claim."""
    _check(not any((root / name).exists() for name in
                   ['direct_decode_receipt.json', 'direct_decode_metadata.json', 'reference_cache_summary.json']),
           'Exact repack cannot also contain direct-DICOM/reference receipts.')
    source = json.loads((root / 'source_cache_summary.json').read_text())
    receipt = json.loads((root / 'repack_receipt.json').read_text())
    manifest = json.loads((root / 'cache_zip_manifest.json').read_text())
    _check(sm.get('source_kind') is None and source.get('source_kind') is None
           and sm.get('repack_complete') is True
           and not any(key in sm for key in ['original_cache5_npy_byte_identity_verified',
               'decode_metadata_checked_against_original_reference']),
           'Exact original-NPY repack cannot claim fresh DICOM provenance.')
    _check(all(sm.get(key) == value for key, value in source.items()),
           'Exact repack changed its original physical source summary.')
    _check(source.get('complete') is True and source.get('errors') == 0
           and source.get('stopped_for_walltime') is False
           and sm['logical_cache_bytes'] == sm['cache_bytes']
           and isinstance(sm['storage_bytes'], int) and sm['storage_bytes'] > 0,
           'Exact repack is incomplete or logical/storage bytes differ.')
    _check(receipt.get('verified_original_npy_sha256_and_archive_crc') is True
           and receipt.get('verified_members') == len(meta)
           and receipt.get('compression') == 'ZIP_DEFLATED'
           and receipt.get('compression_level') == 3 and receipt.get('zip64') is True,
           'Exact-original-NPY readback proof is missing.')
    _check(isinstance(receipt.get('source_kernel'), str) and bool(receipt['source_kernel'])
           and sm.get('source_kernel') == receipt['source_kernel']
           and isinstance(receipt.get('source_version_number'), int)
           and not isinstance(receipt['source_version_number'], bool) and receipt['source_version_number'] >= 0
           and (receipt.get('source_script_version_id') is None
                or isinstance(receipt['source_script_version_id'], int)
                and not isinstance(receipt['source_script_version_id'], bool)
                and receipt['source_script_version_id'] >= 0), 'Exact repack source version receipt is malformed.')
    for key, expected in {
        'source_summary_sha256': sha(root / 'source_cache_summary.json'),
        'source_metadata_sha256': fp['metadata_sha256'],
        'archive_sha256': sm['archive_sha256'], 'manifest_sha256': sm['manifest_sha256']}.items():
        _check(receipt.get(key) == expected, f'Exact repack receipt fingerprint differs: {key}')
    _check(receipt['source_summary_sha256'] == sm['source_summary_sha256']
           and sha(root / 'cache_zip_manifest.json') == sm['manifest_sha256'],
           'Exact repack source summary/manifest fingerprint differs.')
    _check(isinstance(manifest, list), 'Exact repack manifest must be a list.')
    index = {row['series']: row for row in manifest}
    _check(len(index) == len(manifest) == len(meta) and set(index) == set(meta.series),
           'Exact repack manifest coverage differs from original metadata.')
    for row in meta.itertuples():
        entry = index[row.series]
        _check(entry['member'] == f'cache384_train/{row.series}.npy'
               and entry['logical_bytes'] == int(row.cached_bytes)
               and isinstance(entry['storage_bytes'], int) and entry['storage_bytes'] > 0
               and 'source_kind' not in entry
               and entry['source_sha256_before'] == entry['archive_member_sha256_after']
               and re.fullmatch(r'[0-9a-f]{64}', entry['source_sha256_before']) is not None
               and re.fullmatch(r'[0-9a-f]{8}', entry['crc32']) is not None,
               'Exact repack original member SHA/CRC/geometry proof differs.')
    for key in ['archive_sha256', 'manifest_sha256', 'source_summary_sha256']:
        _check(re.fullmatch(r'[0-9a-f]{64}', sm[key]) is not None, f'Invalid exact repack SHA: {key}')
        fp[key] = sm[key]
    fp['repack_receipt_sha256'] = sha(root / 'repack_receipt.json')
    return fp


def cache_receipt_fingerprint(root):
    """Validate small receipts without requiring/downloading the ZIP archive.

    The reference cache's inaccessible NPY bytes are never claimed identical.
    Fresh DICOM preprocessing has matching geometry/counts and its ZIP readback
    has matching SHA/CRC, as recorded by the separate CPU job.
    """
    root = Path(root)
    sm = json.loads((root / 'cache_summary.json').read_text())
    meta = pd.read_csv(root / 'cache384_train_meta.csv')
    plan = root / 'port/plan.csv'
    _check(sm.get('complete') is True and sm.get('errors') == 0
           and sm.get('stopped_for_walltime') is False
           and sm['cached_series'] == sm['planned_series'], 'Incomplete cache receipt.')
    _check(sha(plan) == sm['plan_sha256'], 'Cache plan fingerprint mismatch.')
    planned = pd.read_csv(plan)
    pairs = dict(zip(planned.SeriesInstanceUID, planned.StudyInstanceUID))
    _check(planned.SeriesInstanceUID.is_unique and meta.series.is_unique
           and set(meta.series) == set(pairs) and len(meta) == sm['planned_series']
           and all(pairs[r.series] == r.study for r in meta.itertuples()),
           'Cache metadata coverage differs from source plan.')
    _check(meta.n.between(3, 64).all() and (meta.n_raw > 0).all()
           and int(meta.cached_bytes.sum()) == sm['cache_bytes'], 'Invalid physical cache metadata.')
    _check('error' not in meta or meta.error.isna().all(), 'Cache metadata contains decode errors.')
    fp = {'plan_sha256': sha(plan), 'metadata_sha256': sha(root / 'cache384_train_meta.csv'),
          'summary_sha256': sha(root / 'cache_summary.json')}
    if sm.get('storage_format') != 'zip_npy':
        return fp
    for name in ('archive_path', 'manifest_path'):
        _check(isinstance(sm[name], str) and Path(sm[name]).name == sm[name], 'Unsafe archive/manifest filename.')
    _check(sm['archive_path'] == 'cache384_train.zip' and sm['manifest_path'] == 'cache_zip_manifest.json',
           'Unexpected ZIP storage filenames.')
    if sm.get('source_kind') != 'competition_dicom':
        _check(sm.get('source_kind') is None and (root / 'repack_receipt.json').is_file(),
               'Unknown ZIP provenance kind; neither direct DICOM nor exact original-NPY repack.')
        return _exact_repack_fingerprint(root, sm, meta, fp)
    _check(not (root / 'repack_receipt.json').exists(), 'Direct DICOM cannot also claim exact-original-NPY repack.')
    source = json.loads((root / 'source_cache_summary.json').read_text())
    reference = json.loads((root / 'reference_cache_summary.json').read_text())
    receipt = json.loads((root / 'direct_decode_receipt.json').read_text())
    decoded = pd.DataFrame(json.loads((root / 'direct_decode_metadata.json').read_text()))
    manifest = json.loads((root / 'cache_zip_manifest.json').read_text())
    _check(sm.get('repack_complete') is True and sm.get('source_kind') == 'competition_dicom'
           and sm.get('original_cache5_npy_byte_identity_verified') is False
           and sm.get('decode_metadata_checked_against_original_reference') is True,
           'ZIP cache must honestly describe fresh DICOM preprocessing.')
    for key in ['planned_series', 'cached_series', 'errors', 'complete', 'cache_bytes',
                'plan_sha256', 'preprocess_sha256']:
        _check(source[key] == reference[key] == sm[key], f'Physical cache provenance differs: {key}')
    _check(source.get('source_kind') == 'competition_dicom'
           and source.get('original_cache5_npy_byte_identity_verified') is False
           and source.get('decode_metadata_checked_against_original_reference') is True,
           'Fresh source receipt is missing truthful decode provenance.')
    _check(sha(root / 'source_cache_summary.json') == sm['source_summary_sha256'], 'Fresh source summary changed.')
    _check(sm['logical_cache_bytes'] == sm['cache_bytes'] and isinstance(sm['storage_bytes'], int)
           and sm['storage_bytes'] > 0, 'ZIP logical/storage byte count is invalid.')
    _check(receipt.get('source_kind') == 'competition_dicom'
           and receipt.get('failed_cache5_kernel_input_attached') is False
           and receipt.get('original_cache5_npy_byte_identity_verified') is False
           and receipt.get('fresh_npy_and_archive_sha256_crc_readback_verified') is True
           and receipt.get('compression') == 'ZIP_DEFLATED' and receipt.get('compression_level') == 3
           and receipt.get('zip64') is True and receipt.get('sequential_decode') is True
           and receipt.get('retained_individual_npy_files') == 0
           and receipt.get('verified_members') == len(meta), 'Direct decode/readback integrity receipt is incomplete.')
    for receipt_key, expected in {
        'reference_cache5_metadata_sha256': fp['metadata_sha256'],
        'reference_cache5_summary_sha256': sha(root / 'reference_cache_summary.json'),
        'physical_preprocessor_sha256': sm['preprocess_sha256'], 'plan_sha256': fp['plan_sha256'],
        'archive_sha256': sm['archive_sha256'], 'manifest_sha256': sm['manifest_sha256'],
        'direct_decode_metadata_sha256': sha(root / 'direct_decode_metadata.json')}.items():
        _check(receipt.get(receipt_key) == expected, f'Direct receipt fingerprint differs: {receipt_key}')
    _check(decoded.series.is_unique and set(decoded.series) == set(meta.series), 'Direct decode UID coverage differs.')
    a, b = decoded.set_index('series').sort_index(), meta.set_index('series').sort_index()
    for key in ['study', 'n', 'n_raw', 'n_bad', 'cached_bytes', 'plane', 'estimated_raw_bytes']:
        _check(a[key].tolist() == b[key].tolist(), f'Direct decode geometry/count differs: {key}')
    gap_a, gap_b = pd.to_numeric(a.slice_gap, errors='raise'), pd.to_numeric(b.slice_gap, errors='raise')
    _check(np.array_equal(gap_a.isna().to_numpy(), gap_b.isna().to_numpy())
           and all(math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-7)
                   for a, b in zip(gap_a.fillna(0), gap_b.fillna(0))),
           'Direct decode slice spacing differs from original reference.')
    _check(sha(root / 'cache_zip_manifest.json') == sm['manifest_sha256'], 'ZIP manifest fingerprint differs.')
    _check(isinstance(manifest, list), 'ZIP manifest must be a list.')
    index = {r['series']: r for r in manifest}
    _check(len(index) == len(manifest) == len(meta) and set(index) == set(meta.series), 'ZIP manifest UID coverage differs.')
    for row in meta.itertuples():
        entry = index[row.series]
        _check(entry['member'] == f'cache384_train/{row.series}.npy'
               and entry['logical_bytes'] == int(row.cached_bytes)
               and isinstance(entry['storage_bytes'], int) and entry['storage_bytes'] > 0
               and entry.get('source_kind') == 'fresh_competition_dicom_preprocess_npy'
               and entry['source_sha256_before'] == entry['archive_member_sha256_after']
               and re.fullmatch(r'[0-9a-f]{64}', entry['source_sha256_before']) is not None
               and re.fullmatch(r'[0-9a-f]{8}', entry['crc32']) is not None,
               'ZIP member geometry/integrity receipt differs.')
    for key in ['archive_sha256', 'manifest_sha256', 'source_summary_sha256']:
        _check(re.fullmatch(r'[0-9a-f]{64}', sm[key]) is not None, f'Invalid ZIP SHA: {key}')
        fp[key] = sm[key]
    fp.update(direct_decode_receipt_sha256=sha(root / 'direct_decode_receipt.json'),
              direct_decode_metadata_sha256=sha(root / 'direct_decode_metadata.json'),
              reference_summary_sha256=sha(root / 'reference_cache_summary.json'))
    return fp


class MixedVolumeLoader:
    def __init__(self, paths):
        self.paths = paths
        self._pid = None
        self._archives = {}

    def __getstate__(self):
        # Spawn workers cannot pickle live ZipFile/file descriptors.
        return {'paths': self.paths, '_pid': None, '_archives': {}}

    def close(self):
        for zf in self._archives.values():
            zf.close()
        self._archives.clear()

    def __del__(self):
        self.close()

    def __call__(self, series):
        spec = self.paths[series]
        if isinstance(spec, str):
            return np.load(spec, mmap_mode='r', allow_pickle=False)
        kind, archive, member, n = spec
        if kind != 'zip_npy':
            raise RuntimeError(f'Unknown cache storage type: {kind}')
        pid = os.getpid()
        if self._pid != pid:
            # A fork may inherit parent descriptors; reopen independent handles
            # before any read, so workers never share a mutable ZIP file offset.
            self.close()
            self._pid = pid
        if archive not in self._archives:
            self._archives[archive] = zipfile.ZipFile(archive, 'r')
        with self._archives[archive].open(member, 'r') as f:
            volume = np.load(f, allow_pickle=False)
            if f.read(1):
                raise RuntimeError(f'Trailing bytes after cached NPY array: {series}')
        if volume.dtype != np.uint8 or volume.shape != (n, 384, 384):
            raise RuntimeError(f'Cached volume dtype/geometry changed: {series}')
        return volume


def load_cache_mixed(roots, verify_archive_hash=False):
    """Return trainer-compatible mapping/counts/summaries/fingerprints.

    Whole-archive SHA verification is optional at training startup. The direct
    CPU decoder verified each fresh NPY/archive member SHA and CRC; startup always
    verifies source receipts, manifest SHA, member CRC/index/size and geometry.
    """
    counts, mapping, summaries, fingerprints = {}, {}, [], []
    for root in map(Path, roots):
        sm = json.loads((root / 'cache_summary.json').read_text())
        meta = pd.read_csv(root / 'cache384_train_meta.csv')
        plan = root / 'port/plan.csv'
        if not sm.get('complete') or sm.get('errors') or sm['cached_series'] != sm['planned_series']:
            raise RuntimeError(f'Incomplete cache source: {root}')
        if not plan.is_file() or sha(plan) != sm['plan_sha256']:
            raise RuntimeError(f'Cache plan fingerprint mismatch: {root}')
        expected_ids = set(pd.read_csv(plan).SeriesInstanceUID)
        if not meta.series.is_unique or set(meta.series) != expected_ids or len(meta) != sm['planned_series']:
            raise RuntimeError(f'Cache metadata coverage differs from source plan: {root}')
        if 'error' in meta and meta.error.notna().any():
            raise RuntimeError(f'Cache metadata retains decode errors: {root}')
        if not meta.n.between(3, 64).all() or not (meta.n_raw > 0).all():
            raise RuntimeError(f'Invalid slice count/decode provenance: {root}')
        fp = cache_receipt_fingerprint(root)
        is_zip = sm.get('storage_format') == 'zip_npy'
        index = {}
        if is_zip:
            archive_name = sm['archive_path']
            manifest_name = sm['manifest_path']
            if Path(archive_name).name != archive_name or Path(manifest_name).name != manifest_name:
                raise RuntimeError('Archive/manifest path must be a single filename.')
            archive = root / archive_name
            manifest_path = root / manifest_name
            if not sm.get('repack_complete') or archive.stat().st_size != sm['storage_bytes']:
                raise RuntimeError(f'Compressed repack incomplete or storage size differs: {root}')
            if sha(root / 'source_cache_summary.json') != sm['source_summary_sha256']:
                raise RuntimeError(f'Original complete cache receipt changed: {root}')
            old_sm = json.loads((root / 'source_cache_summary.json').read_text())
            for key in ['planned_series', 'cached_series', 'errors', 'complete', 'cache_bytes',
                        'plan_sha256', 'preprocess_sha256']:
                if old_sm[key] != sm[key]:
                    raise RuntimeError(f'Physical cache provenance differs after repack: {key}')
            if sm['logical_cache_bytes'] != sm['cache_bytes'] or int(meta.cached_bytes.sum()) != sm['cache_bytes']:
                raise RuntimeError('Logical NPY byte count changed after storage compression.')
            if sha(manifest_path) != sm['manifest_sha256']:
                raise RuntimeError(f'Archive member manifest fingerprint differs: {root}')
            manifest = json.loads(manifest_path.read_text())
            index = {r['series']: r for r in manifest}
            if len(index) != len(manifest) or set(index) != expected_ids:
                raise RuntimeError('Archive member manifest coverage differs from plan.')
            with zipfile.ZipFile(archive, 'r') as zf:
                names = [r['member'] for r in manifest]
                if len(set(names)) != len(names) or set(zf.namelist()) != set(names) or len(zf.infolist()) != len(names):
                    raise RuntimeError('ZIP includes unexpected or duplicate members.')
                for r in manifest:
                    if r['member'] != f'cache384_train/{r["series"]}.npy':
                        raise RuntimeError('Archive member name does not match series UID.')
                    info = zf.getinfo(r['member'])
                    if info.file_size != r['logical_bytes'] or info.compress_size != r['storage_bytes']:
                        raise RuntimeError('ZIP member byte count differs from manifest.')
                    if f'{info.CRC:08x}' != r['crc32'] or r['source_sha256_before'] != r['archive_member_sha256_after']:
                        raise RuntimeError('ZIP member integrity receipt differs.')
                    if info.compress_type != zipfile.ZIP_DEFLATED or info.is_dir():
                        raise RuntimeError('Unexpected archive member storage method.')
            if verify_archive_hash and sha(archive) != sm['archive_sha256']:
                raise RuntimeError('Full archive SHA differs from the CPU repack receipt.')
        for row in meta.itertuples():
            sr = row.series
            if sr in mapping:
                raise RuntimeError(f'Duplicate series across explicitly selected cache roots: {sr}')
            if is_zip:
                member = index[sr]
                if member['logical_bytes'] != int(row.cached_bytes):
                    raise RuntimeError(f'Archive member size differs from decode metadata: {sr}')
                mapping[sr] = ('zip_npy', str(archive), member['member'], int(row.n))
            else:
                file = root / 'cache384_train' / f'{sr}.npy'
                if not file.is_file() or file.stat().st_size != int(row.cached_bytes):
                    raise RuntimeError(f'Missing/size-mismatched plain cache file: {file}')
                mapping[sr] = str(file)
            counts[sr] = int(row.n)
        summaries.append(sm)
        fingerprints.append(fp)
    if len({s['preprocess_sha256'] for s in summaries}) != 1:
        raise RuntimeError('Cache physical preprocessing fingerprints differ.')
    fingerprints.sort(key=lambda d: d['plan_sha256'])
    return mapping, counts, summaries, fingerprints

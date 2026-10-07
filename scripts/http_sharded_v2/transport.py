"""Receipt views for exact HTTP ZIP bytes; no API, signing or account auth."""
import hashlib
import json
from pathlib import Path
import shutil
import time

CAPS = [1204, 1231, 1236, 1213, 1216, 1228, 1069]
IMAGES = [12240, 12528, 12580, 12332, 12364, 12496, 10796]


def file_sha(path, deadline=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError('Receipt verification exhausted stage deadline')
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def expected_spec(manifest, root, shard, runtime, images):
    if type(shard) is not int or shard not in range(7):
        raise RuntimeError('Invalid HTTP shard')
    if runtime != CAPS[shard] or images != IMAGES[shard]:
        raise RuntimeError('HTTP shard cap or global selected-image count differs')
    source = next(row for row in manifest['source_plan'] if row['shard'] == shard)
    pin = next(row for row in manifest['cache_input_fingerprints'] if row['plan_sha256'] == source['plan_sha256'])
    summary = json.loads((Path(root) / f'cache_receipts/shard_{shard}/cache_summary.json').read_text())
    prefix = f'cache_receipts/shard_{shard}/'
    members = [{**row, 'path': row['path'][len(prefix):]} for row in manifest['members'] if row['path'].startswith(prefix)]
    if not members or len({row['path'] for row in members}) != len(members):
        raise RuntimeError('Missing or duplicate globally pinned shard receipts')
    origin = ({'kind': 'dataset', 'identifier': 'alanchoo/rsna-knee-cache-0-bin-v1',
               'version_number': 1, 'filename': 'cache384_train.zip.bin'} if shard == 0 else
              {'kind': 'kernel', 'identifier': source['kernel_id'],
               'version_number': 1, 'filename': 'cache384_train.zip'})
    return {'shard': shard, 'cache_kernel_id': source['kernel_id'], 'origin': origin,
            'archive': {'filename': 'cache384_train.zip', 'bytes': summary['storage_bytes'],
                        'sha256': pin['archive_sha256']},
            'job_filename_alias': 'cache384_train.zip.bin', 'job_id': f'shard_{shard}',
            'fingerprint': pin, 'receipt_members': members, 'storage_kind': source['storage_kind'],
            'original_npy_byte_identity_claim': False if shard == 5 else True,
            'runtime_seconds': runtime, 'expected_images': images}


def validate_spec(spec, manifest, root):
    expected = expected_spec(manifest, root, spec['shard'], spec['runtime_seconds'], spec['expected_images'])
    if spec != expected:
        raise RuntimeError('HTTP cache spec differs from globally pinned original ZIP source')
    if not 0 < spec['archive']['bytes'] <= 11_000_000_000:
        raise RuntimeError('HTTP archive size outside reviewed downloader bounds')
    for member in spec['receipt_members']:
        path = Path(member['path'])
        if path.is_absolute() or '..' in path.parts or path == Path('.'):
            raise RuntimeError('Unsafe receipt member path')


def prepare_view(root, target, spec, manifest, download, job_path, report_path, deadline,
                 allow_local_fixture=False):
    """Small receipts copy exactly; downloaded ZIP is already in owned /tmp.

    The downloader's streaming full SHA is authoritative. The unchanged core
    loader subsequently verifies metadata/receipt fingerprint and ZIP index/CRC.
    No second 10 GB hash, archive rewrite, compression or large working symlink.
    """
    started = time.monotonic()
    validate_spec(spec, manifest, root)
    target = Path(target)
    if target.exists():
        raise RuntimeError('HTTP receipt view already exists')
    source = Path(root) / f"cache_receipts/shard_{spec['shard']}"
    # Check all source receipts before copying any member or starting HTTP.
    for member in spec['receipt_members']:
        path = source / member['path']
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != member['bytes']
                or file_sha(path, deadline) != member['sha256']):
            raise RuntimeError('Global shard receipt bytes/SHA changed before view preparation')
    target.mkdir(parents=True)
    for member in spec['receipt_members']:
        destination = target / member['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / member['path'], destination)
        if destination.stat().st_size != member['bytes'] or file_sha(destination, deadline) != member['sha256']:
            raise RuntimeError('Copied HTTP receipt bytes/SHA changed')
    receipt_seconds = time.monotonic() - started
    result = download(job_path, target / spec['archive']['filename'], report_path, deadline,
                      spec['archive'], spec['origin'], spec['shard'], spec['runtime_seconds'],
                      allow_local_fixture)
    if not result.get('complete') or not result.get('download_sha256_verified'):
        raise RuntimeError('HTTP ZIP was not fully downloaded and SHA verified')
    return target, {'shard': spec['shard'], 'origin': spec['origin'], 'bytes': spec['archive']['bytes'],
                    'sha256': spec['archive']['sha256'], 'receipt_seconds': receipt_seconds,
                    'download_seconds': result['seconds'], 'full_streaming_SHA_verified': True,
                    'second_full_archive_hash': False, 'storage_kind': spec['storage_kind'],
                    'original_npy_byte_identity_claim': spec['original_npy_byte_identity_claim']}

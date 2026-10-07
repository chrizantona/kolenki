"""Explicit owner-issued compute gate; source pins come from metadata, not job."""
import math
import re
import time


def validate(approval, shard, runtime, expected_identity, source_index_sha, forecast_function, specs, now=None):
    if (approval.get('schema') != 'http_sharded_v2_owner_gate' or approval.get('shard') != shard
            or approval.get('approved') is not True or approval.get('actual_mri_pilot_pass') is not True
            or approval.get('feature_identity') != expected_identity
            or approval.get('source_index_sha256') != source_index_sha
            or approval.get('pilot_kernel') != 'alanchoo/rsna-knee-ortho-http-cache0-pilot-v2'):
        raise RuntimeError('Missing matching owner gate after actual V2 MRI pilot PASS')
    hashes = approval.get('pilot_receipt_sha256', {})
    if set(hashes) != {'pilot_report', 'http_cache_download', 'cache_dataset_transport'} or any(
            not re.fullmatch(r'[0-9a-f]{64}', str(value)) for value in hashes.values()):
        raise RuntimeError('Actual sealed pilot receipt hashes are required')
    measurements = approval.get('measured', {})
    projection = forecast_function(specs, measurements.get('http_bytes_per_second'),
                                   measurements.get('non_transport_startup_seconds'),
                                   measurements.get('gpu_images_per_second'))
    if not projection['fits_total'] or not projection['fits_every_shard']:
        raise RuntimeError('Whole HTTP route forecast exceeds fixed total or shard caps')
    quota = approval.get('fresh_quota', {})
    values = [quota.get('observed_unix_seconds'), quota.get('available_seconds'), quota.get('required_seconds')]
    if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
        raise RuntimeError('Fresh quota receipt must contain finite measured times and seconds')
    observed, available, required = values
    current = time.time() if now is None else now
    if not 0 <= current - observed <= 900 or required < runtime + 1500 or available < required:
        raise RuntimeError('Quota receipt is stale or below owner reserve including head job')
    return projection

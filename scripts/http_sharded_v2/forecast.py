"""Extrinsic whole-route forecast; no authority to start GPU work."""
import math

EXPECTED_IMAGES = [12240, 12528, 12580, 12332, 12364, 12496, 10796]
CAPS = [1204, 1231, 1236, 1213, 1216, 1228, 1069]
TOTAL_IMAGES = 85336
PILOT_CAP = 900
TOTAL_CAP = 9300


def project(specs, http_bytes_per_second, non_transport_startup_seconds, gpu_images_per_second):
    numbers = [http_bytes_per_second, non_transport_startup_seconds, gpu_images_per_second]
    if (not all(type(value) in (int, float) and math.isfinite(value) for value in numbers)
            or http_bytes_per_second <= 0 or gpu_images_per_second <= 0 or non_transport_startup_seconds < 0):
        raise RuntimeError('Invalid measured HTTP/GPU throughput or non-transport startup')
    if len(specs) != 7 or [row['shard'] for row in specs] != list(range(7)):
        raise RuntimeError('Forecast requires all seven ordered immutable archive pins')
    if [row['expected_images'] for row in specs] != EXPECTED_IMAGES:
        raise RuntimeError('Forecast global selected-image counts differ')
    if [row['runtime_seconds'] for row in specs] != CAPS:
        raise RuntimeError('Forecast original shard caps differ')
    archive_bytes = [row['archive']['bytes'] for row in specs]
    if any(type(value) is not int or value <= 0 for value in archive_bytes):
        raise RuntimeError('Invalid forecast archive byte counts')
    jobs = []
    for row in specs:
        http = row['archive']['bytes'] / http_bytes_per_second
        gpu = row['expected_images'] / gpu_images_per_second
        total = http + non_transport_startup_seconds + gpu
        jobs.append({'shard': row['shard'], 'http_seconds': http, 'gpu_seconds': gpu,
                     'non_transport_startup_seconds': non_transport_startup_seconds,
                     'projected_seconds': total, 'cap_seconds': row['runtime_seconds'],
                     'fits_cap': total <= row['runtime_seconds']})
    transport = sum(archive_bytes) / http_bytes_per_second
    startup = non_transport_startup_seconds * 7
    gpu = TOTAL_IMAGES / gpu_images_per_second
    total = PILOT_CAP + transport + startup + gpu
    return {'schema': 'http_sharded_v2_forecast', 'all_archive_bytes': sum(archive_bytes),
            'http_bytes_per_second': http_bytes_per_second, 'gpu_images_per_second': gpu_images_per_second,
            'non_transport_startup_seconds_per_job': non_transport_startup_seconds,
            'transport_seconds_all_seven': transport, 'startup_seconds_all_seven': startup,
            'GPU_feature_seconds_all_85336_images': gpu, 'pilot_reserved_seconds': PILOT_CAP,
            'projected_total_extraction_seconds': total, 'total_extraction_cap_seconds': TOTAL_CAP,
            'fits_total': total <= TOTAL_CAP, 'fits_every_shard': all(row['fits_cap'] for row in jobs),
            'head_fit_seconds': 1200, 'head_job_reserve_seconds': 1500,
            'projected_total_plus_head_job_seconds': total + 1500, 'shards': jobs,
            'launch_authorized': False,
            'remaining_gates': ['Actual MRI pilot PASS and sealed measured receipts', 'Fresh quota before every job'],
            'timer_boundary': 'Python HTTP, hashing, setup, encoder and export; pre-Python Kaggle mounting excluded'}


def from_pilot(specs, pilot, download, cache_transport, expected_identity):
    if (pilot.get('feature_identity') != expected_identity or pilot.get('shards') != [0]
            or not pilot.get('pilot_only') or pilot.get('max_seconds') != 900
            or pilot.get('head_smoke', {}).get('optimizer_updates', 0) < 1):
        raise RuntimeError('No matching actual V2 MRI pilot/head-update receipt')
    features = pilot['feature_extraction']
    if not features.get('complete') or features.get('images_encoded', 0) <= 0 or features.get('seconds', 0) <= 0:
        raise RuntimeError('No actual finite completed MRI extraction throughput')
    if (not download.get('complete') or not download.get('download_sha256_verified')
            or download.get('expected_bytes') != specs[0]['archive']['bytes']
            or download.get('expected_sha256') != specs[0]['archive']['sha256']
            or download.get('origin') != specs[0]['origin'] or download.get('seconds', 0) <= 0):
        raise RuntimeError('HTTP speed does not belong to exact completed source0 download')
    mirrors = cache_transport.get('mirrors', [])
    if (len(mirrors) != 1 or mirrors[0].get('shard') != 0 or mirrors[0].get('archive_hash_seconds', -1) < 0
            or mirrors[0].get('archive_sha256') != specs[0]['archive']['sha256']
            or mirrors[0].get('archive_bytes') != specs[0]['archive']['bytes']):
        raise RuntimeError('Missing V2 pilot second-hash timing receipt')
    # Full route streams SHA once. Keep the pilot's measured second hash inside
    # non-transport startup conservatively; do not silently assume a speedup.
    startup = pilot['budget_projection']['startup_seconds'] - download['seconds']
    if not math.isfinite(startup) or startup < 0:
        raise RuntimeError('Pilot transport/startup timing boundaries are inconsistent')
    report = project(specs, download['expected_bytes'] / download['seconds'], startup,
                     features['images_encoded'] / features['seconds'])
    report['pilot_feature_identity'] = expected_identity
    report['actual_matching_pilot_receipts_verified'] = True
    report['pilot_second_archive_hash_seconds_retained_in_startup'] = mirrors[0]['archive_hash_seconds']
    return report

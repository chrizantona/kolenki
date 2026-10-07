import hashlib,json,re,time,zlib
from pathlib import Path
import requests
from transfer_reviewed import TransferStopped,RejectAllCookies,validate_job,atomic_report,DOWNLOAD_RANGE_BYTES

def download(job_path,archive,output,deadline,expected,expected_origin,shard,runtime,allow_local_fixture=False):
    start=time.monotonic();report={'complete':False,'route':'direct_http','origin':expected_origin,
        'urls_or_credentials_written_to_output':False};session=None;stage='validate'
    archive=Path(archive)
    def check_time():
        if time.monotonic()>=deadline:raise TransferStopped('walltime_budget')
    try:
        job=json.loads(Path(job_path).read_text())
        if set(job)!={'job_id','filename','expected_bytes','expected_sha256','read_url','max_seconds','source_origin'}:
            raise TransferStopped('invalid_read_only_job_fields')
        if job['source_origin']!=expected_origin or job['job_id']!=f'shard_{shard}':raise TransferStopped('source_origin_mismatch')
        if job['expected_bytes']!=expected['bytes'] or job['expected_sha256']!=expected['sha256']:
            raise TransferStopped('immutable_cache_pin_mismatch')
        if job['max_seconds']!=runtime:raise TransferStopped('runtime_pin_mismatch')
        validation={k:v for k,v in job.items() if k!='source_origin'}
        # Constant validation-only placeholder reuses the reviewed URL/field guards.
        # This module has no upload branch and never contacts this placeholder.
        validation['put_url']='https://storage.googleapis.com/disabled-validation-only'
        validate_job(validation,allow_local_fixture)
        total=job['expected_bytes'];report.update(expected_bytes=total,expected_sha256=job['expected_sha256'])
        archive.parent.mkdir(parents=True,exist_ok=True)
        session=requests.Session();session.trust_env=False;session.headers.clear();session.auth=None
        session.cookies.set_policy(RejectAllCookies())
        stage = "download"
        digest, downloaded = hashlib.sha256(), 0
        report.update(download_range_checks=[], download_retry_count=0, download_range_max_bytes=DOWNLOAD_RANGE_BYTES)
        with archive.open("w+b") as stream:
            for offset in range(0, total, DOWNLOAD_RANGE_BYTES):
                end = min(total, offset + DOWNLOAD_RANGE_BYTES) - 1
                expected_part = end - offset + 1
                for attempt in range(3):
                    check_time()
                    stream.seek(offset)
                    stream.truncate(offset)
                    part_digest, received, crc = digest.copy(), 0, 0
                    try:
                        with session.get(job["read_url"], stream=True, timeout=(15, 60), allow_redirects=False,
                                         headers={"Accept-Encoding": "identity", "Range": f"bytes={offset}-{end}"}) as response:
                            report["download_http_status"] = response.status_code
                            if response.status_code != 206:
                                raise TransferStopped("download_range_http_failure")
                            if response.headers.get("Content-Encoding", "identity") != "identity":
                                raise TransferStopped("unexpected_transfer_encoding")
                            cr = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                            if not cr or tuple(map(int, cr.groups())) != (offset, end, total):
                                raise TransferStopped("download_content_range_mismatch")
                            length = response.headers.get("Content-Length")
                            if length is not None and int(length) != expected_part:
                                raise TransferStopped("download_header_size_mismatch")
                            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                                check_time()
                                if not chunk:
                                    continue
                                received += len(chunk)
                                if received > expected_part:
                                    raise TransferStopped("download_range_exceeded_size")
                                part_digest.update(chunk)
                                crc = zlib.crc32(chunk, crc)
                                stream.write(chunk)
                        if received != expected_part:
                            raise TransferStopped("download_range_body_size_mismatch")
                        digest = part_digest
                        downloaded += received
                        report["download_range_checks"].append({"start": offset, "end": end, "bytes": received,
                                                                "crc32_diagnostic": f"{crc:08x}"})
                        break
                    except requests.RequestException:
                        report["download_retry_count"] += 1
                        if attempt == 2:
                            raise TransferStopped("download_range_transport_attempt_budget") from None
        report.update(downloaded_bytes=downloaded, downloaded_sha256=digest.hexdigest())
        if downloaded != total or digest.hexdigest() != job["expected_sha256"]:
            raise TransferStopped("download_size_or_sha_mismatch")
        report["download_sha256_verified"] = True
        check_time()
        report['complete']=True
    except BaseException as error:
        report.update(complete=False,failure_stage=stage,failure_code=error.code if isinstance(error,TransferStopped)
                      else 'redacted_download_exception',exception_type=type(error).__name__)
    finally:
        if session is not None:session.close()
        if not report['complete']:archive.unlink(missing_ok=True)
        report['seconds']=time.monotonic()-start;atomic_report(report,output)
    if not report['complete']:raise TransferStopped('http_cache_download_failed')
    return report

"""Single-file cloud transfer using temporary URLs; no Kaggle account credential.

Preparation only. The parent creates/retains the blob upload token on the Mac.
This utility only receives the exact file's read URL and resumable PUT URL.
"""
import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
import zlib
from http.cookiejar import DefaultCookiePolicy
from pathlib import Path
from urllib.parse import urlsplit

import requests

DOWNLOAD_RANGE_BYTES = 512 * 1024 * 1024


class TransferStopped(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class RejectAllCookies(DefaultCookiePolicy):
    def set_ok(self, cookie, request):
        return False

    def return_ok(self, cookie, request):
        return False


class DeadlineReader:
    """Seekable upload body with a deadline checked on each transport read."""
    mode = "rb"

    def __init__(self, stream, check_time):
        self.stream, self.check_time = stream, check_time

    def read(self, size=-1):
        self.check_time()
        return self.stream.read(size)

    def tell(self):
        return self.stream.tell()

    def seek(self, *args):
        return self.stream.seek(*args)

    def fileno(self):
        return self.stream.fileno()


def validate_job(job, allow_local_fixture=False):
    required = {"job_id", "filename", "expected_bytes", "expected_sha256", "read_url", "put_url"}
    if set(job) - required - {"max_seconds"} or required - set(job):
        raise TransferStopped("invalid_job_fields")
    if not re.fullmatch(r"shard_[0-6]", str(job["job_id"])):
        raise TransferStopped("invalid_job_id")
    if job["filename"] != "cache384_train.zip.bin":
        raise TransferStopped("invalid_generic_filename")
    if type(job["expected_bytes"]) is not int or not 0 < job["expected_bytes"] <= 11_000_000_000:
        raise TransferStopped("invalid_file_size")
    if not re.fullmatch(r"[0-9a-f]{64}", str(job["expected_sha256"])):
        raise TransferStopped("invalid_sha256")
    if type(job.get("max_seconds", 1800)) is not int or not 1 <= job.get("max_seconds", 1800) <= 7200:
        raise TransferStopped("invalid_time_budget")
    for field in ("read_url", "put_url"):
        parsed = urlsplit(job[field])
        fixture = allow_local_fixture and parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
        allowed_hosts = {"storage.googleapis.com", "www.kaggleusercontent.com"} if field == "read_url" else {"storage.googleapis.com"}
        if not fixture and (parsed.scheme != "https" or parsed.hostname not in allowed_hosts):
            raise TransferStopped("unsupported_scoped_storage_host")
        if parsed.username or parsed.password or parsed.fragment:
            raise TransferStopped("unsafe_url_structure")
    if job["read_url"] == job["put_url"]:
        raise TransferStopped("identical_source_destination")


def received_offset(range_header, total):
    if range_header is None:
        return 0
    match = re.fullmatch(r"bytes=0-(\d+)", range_header)
    if not match or not 0 <= int(match[1]) < total:
        raise TransferStopped("invalid_resumable_range")
    return int(match[1]) + 1


def atomic_report(report, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(destination)


def transfer(job, output, scratch_parent, allow_local_fixture=False):
    # Secret values never enter the report, progress, filenames, or exceptions.
    report = {"complete": False, "cloud_account_token_used": False,
              "urls_or_upload_tokens_written_to_output": False}
    scratch = None
    stage = "validate"
    start = time.monotonic()
    session = requests.Session()
    session.trust_env = False
    session.headers.clear()
    session.auth = None
    session.cookies.set_policy(RejectAllCookies())
    deadline = None

    def check_time():
        if deadline is not None and time.monotonic() >= deadline:
            raise TransferStopped("walltime_budget")

    try:
        validate_job(job, allow_local_fixture)
        report.update(job_id=job["job_id"], filename=job["filename"],
                      expected_bytes=job["expected_bytes"], expected_sha256=job["expected_sha256"])
        total = job["expected_bytes"]
        deadline = start + job.get("max_seconds", 1800)
        parent = Path(scratch_parent)
        parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < total + 512 * 1024 * 1024:
            raise TransferStopped("insufficient_scratch_space")
        scratch = Path(tempfile.mkdtemp(prefix="cache_transfer_", dir=parent))
        archive = scratch / job["filename"]
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
        stage = "upload"
        offset = 0
        probe_statuses = []
        for attempt in range(1, 4):
            check_time()
            report["upload_attempts"] = attempt
            headers = {"Content-Length": str(total - offset)}
            if offset:
                headers["Content-Range"] = f"bytes {offset}-{total - 1}/{total}"
            try:
                with archive.open("rb", buffering=0) as stream:
                    stream.seek(offset)
                    response = session.put(job["put_url"], data=DeadlineReader(stream, check_time), headers=headers,
                                           allow_redirects=False, timeout=(15, 60))
                report["upload_http_status"] = response.status_code
                if response.status_code in (200, 201):
                    check_time()
                    report["upload_complete"] = True
                    report["complete"] = True
                    break
                if response.status_code not in (308, 503):
                    raise TransferStopped("upload_nonresumable_http_failure")
            except requests.RequestException:
                # Exception text can contain the scoped URL. Never expose it.
                report["upload_http_status"] = None
            check_time()
            response = session.put(job["put_url"], headers={"Content-Length": "0", "Content-Range": f"bytes */{total}"},
                                   allow_redirects=False, timeout=(15, 60))
            check_time()
            probe_statuses.append(response.status_code)
            if response.status_code in (200, 201):
                report.update(upload_complete=True, complete=True, upload_http_status=response.status_code)
                break
            if response.status_code != 308:
                raise TransferStopped("upload_resume_probe_failure")
            offset = received_offset(response.headers.get("Range"), total)
        report["resume_probe_http_statuses"] = probe_statuses
        if not report["complete"]:
            raise TransferStopped("upload_attempt_budget")
    except BaseException as error:
        report.update(complete=False, failure_stage=stage,
                      failure_code=error.code if isinstance(error, TransferStopped) else "redacted_unexpected_exception",
                      exception_type=type(error).__name__)
    finally:
        session.close()
        if scratch is not None:
            try:
                shutil.rmtree(scratch)
            except OSError as error:
                report.update(complete=False, cleanup_exception_type=type(error).__name__)
        report["own_temp_archive_removed"] = scratch is None or not scratch.exists()
        report["total_seconds"] = time.monotonic() - start
        atomic_report(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/transfer_report.json"))
    parser.add_argument("--scratch-parent", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    # Loading also uses a redacted failure path: malformed private JSON may have
    # URLs in its decoder message, so do not propagate that message.
    try:
        job = json.loads(args.job.read_text())
    except BaseException as error:
        result = {"complete": False, "failure_stage": "job_load", "failure_code": "invalid_private_job",
                  "exception_type": type(error).__name__, "urls_or_upload_tokens_written_to_output": False}
        atomic_report(result, args.output)
        print(json.dumps(result), flush=True)
        raise SystemExit(1)
    result = transfer(job, args.output, args.scratch_parent)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()

"""Protocol fixtures only: no Kaggle API, remote URLs, blobs, or job launch."""
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import transfer
import supervise

BODY = bytes(range(256)) * 512
URL_CANARY = "SIGNED_URL_MUST_NOT_LEAK_ABC987"
ACCOUNT_CANARY = "ACCOUNT_TOKEN_MUST_NOT_LEAVE_MAC_XYZ432"


class FixtureHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.server.request_headers.append(dict(self.headers))
        self.server.gets += 1
        if self.server.block_get:
            time.sleep(2)
        if self.server.redirect:
            self.send_response(302)
            self.send_header("Location", "https://untrusted.invalid/read?signature=" + URL_CANARY)
            self.end_headers()
            return
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not match:
            self.send_response(400)
            self.end_headers()
            return
        start, end = map(int, match.groups())
        data = BODY[start:end+1]
        self.send_response(206)
        cr_start = start + 1 if self.server.bad_download_range else start
        self.send_header("Content-Range", f"bytes {cr_start}-{end}/{len(BODY)}")
        self.send_header("Content-Length", str(len(data)))
        if self.server.set_cookie:
            self.send_header("Set-Cookie", "scoped_cookie=" + URL_CANARY + "; Path=/")
        self.end_headers()
        if self.server.fail_download_once and self.server.gets == 2:
            self.wfile.write(data[:12345])
            self.wfile.flush()
            self.close_connection = True
            return
        try:
            if self.server.trickle_get:
                for offset in range(0, len(data), 1024):
                    self.wfile.write(data[offset:offset + 1024])
                    self.wfile.flush()
                    time.sleep(0.03)
            else:
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_PUT(self):
        self.server.request_headers.append(dict(self.headers))
        length = int(self.headers.get("Content-Length", "0"))
        cr = self.headers.get("Content-Range")
        if length == 0 and cr:
            self.server.probes += 1
            if self.server.late_probe_complete:
                time.sleep(2)
                self.send_response(200)
                self.end_headers()
                return
            self.send_response(308)
            if self.server.received:
                value = "bytes=10-50" if self.server.bad_range else f"bytes=0-{len(self.server.received)-1}"
                self.send_header("Range", value)
            self.end_headers()
            return
        body = self.rfile.read(length)
        self.server.puts += 1
        if self.server.fail_first and self.server.puts == 1:
            self.server.received = body[:10000]
            self.send_response(503)
            self.end_headers()
            return
        if cr:
            self.server.resume_header = cr
            self.server.received += body
        else:
            self.server.received = body
        self.send_response(201)
        self.end_headers()


class TransferFixtures(unittest.TestCase):
    def setUp(self):
        local = Path(__file__).parent / "fixture_tmp"
        local.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=local)
        self.directory = Path(self.temp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        for key, value in {"request_headers": [], "received": b"", "puts": 0,
                           "probes": 0, "fail_first": False, "bad_range": False, "resume_header": None,
                           "gets": 0, "redirect": False, "bad_download_range": False, "fail_download_once": False,
                           "set_cookie": False, "late_probe_complete": False, "block_get": False, "trickle_get": False}.items():
            setattr(self.server, key, value)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.job = {"job_id": "shard_1", "filename": "cache384_train.zip.bin", "expected_bytes": len(BODY),
                    "expected_sha256": hashlib.sha256(BODY).hexdigest(),
                    "read_url": self.base + "/read?signature=" + URL_CANARY,
                    "put_url": self.base + "/put?upload_id=" + URL_CANARY, "max_seconds": 30}
        self.output = self.directory / "transfer_report.json"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def run_transfer(self, job=None):
        with patch.dict(os.environ, {"KAGGLE_API_TOKEN": ACCOUNT_CANARY}), patch.object(transfer, "DOWNLOAD_RANGE_BYTES", 32768):
            result = transfer.transfer(job or self.job, self.output, self.directory, allow_local_fixture=True)
        text = self.output.read_text()
        self.assertNotIn(URL_CANARY, text)
        self.assertNotIn(ACCOUNT_CANARY, text)
        self.assertNotIn("http://", text)
        self.assertEqual(list(self.directory.iterdir()), [self.output])
        self.assertTrue(result["own_temp_archive_removed"])
        for headers in self.server.request_headers:
            self.assertNotIn("Authorization", headers)
            self.assertNotIn("Cookie", headers)
            self.assertNotIn(ACCOUNT_CANARY, json.dumps(headers))
        return result

    def test_exact_download_and_seekable_upload(self):
        result = self.run_transfer()
        self.assertTrue(result["complete"])
        self.assertEqual(self.server.received, BODY)
        self.assertEqual(result["upload_http_status"], 201)
        self.assertEqual(result["upload_attempts"], 1)
        self.assertEqual(len(result["download_range_checks"]), 4)

    def test_range_disconnect_retries_same_offset_without_double_hashing(self):
        self.server.fail_download_once = True
        result = self.run_transfer()
        self.assertTrue(result["complete"])
        self.assertEqual(self.server.received, BODY)
        self.assertEqual(result["download_retry_count"], 1)
        self.assertEqual(self.server.gets, 5)
        self.assertEqual([row["start"] for row in result["download_range_checks"]], [0,32768,65536,98304])

    def test_download_content_range_mismatch_prevents_put(self):
        self.server.bad_download_range = True
        result = self.run_transfer()
        self.assertFalse(result["complete"])
        self.assertEqual(result["failure_code"], "download_content_range_mismatch")
        self.assertEqual(self.server.puts, 0)

    def test_redirect_is_rejected_without_following_untrusted_host(self):
        self.server.redirect = True
        result = self.run_transfer()
        self.assertFalse(result["complete"])
        self.assertEqual(result["failure_code"], "download_range_http_failure")
        self.assertEqual(self.server.gets, 1)
        self.assertEqual(self.server.puts, 0)

    def test_503_resume_308_content_range_matches_sdk(self):
        self.server.fail_first = True
        result = self.run_transfer()
        self.assertTrue(result["complete"])
        self.assertEqual(self.server.received, BODY)
        self.assertEqual(self.server.resume_header, f"bytes 10000-{len(BODY)-1}/{len(BODY)}")
        self.assertEqual(result["resume_probe_http_statuses"], [308])
        self.assertEqual(result["upload_attempts"], 2)

    def test_set_cookie_is_rejected_before_following_ranges_and_put(self):
        self.server.set_cookie = True
        self.server.fail_first = True
        result = self.run_transfer()
        self.assertTrue(result["complete"])
        self.assertEqual(self.server.received, BODY)
        self.assertGreater(len(self.server.request_headers), 5)

    def test_successful_resume_probe_outside_budget_is_rejected(self):
        self.server.fail_first = True
        self.server.late_probe_complete = True
        result = self.run_transfer(dict(self.job, max_seconds=1))
        self.assertFalse(result["complete"])
        self.assertEqual(result["failure_code"], "walltime_budget")
        self.assertNotIn("upload_complete", result)

    def supervised_fixture(self, budget=1, expect_timeout=True):
        child = self.directory / "fixture_child.py"
        source = (
            "import argparse,json,sys\n"
            f"sys.path.insert(0,{str(Path(__file__).parent.resolve())!r})\n"
            "import transfer\n"
            "p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--output');p.add_argument('--scratch-parent');a=p.parse_args()\n"
            "r=transfer.transfer(json.loads(open(a.job).read()),a.output,a.scratch_parent,allow_local_fixture=True)\n"
            "raise SystemExit(0 if r['complete'] else 1)\n"
        )
        child.write_text(source)
        job_path = self.directory / "private_job.json"
        job_path.write_text(json.dumps(dict(self.job, max_seconds=budget)))
        scratch = self.directory / "supervisor_scratch"
        result = supervise.supervise(child, job_path, self.output, scratch_parent=scratch)
        if expect_timeout:
            self.assertFalse(result["complete"])
            self.assertEqual(result["failure_code"], "hard_walltime_budget")
            self.assertTrue(result["child_terminated"])
            self.assertLess(result["supervisor_total_seconds"], 1.5)
        else:
            self.assertTrue(result["complete"])
            self.assertEqual(result["child_exit_code"], 0)
            self.assertEqual(result["downloaded_sha256"], hashlib.sha256(BODY).hexdigest())
            self.assertEqual(self.server.received, BODY)
        self.assertTrue(result["supervisor_owned_scratch_removed"])
        self.assertTrue(result["own_temp_archive_removed"])
        self.assertEqual(list(scratch.iterdir()), [])
        text = self.output.read_text()
        self.assertNotIn(URL_CANARY, text)
        self.assertNotIn("http://", text)
        return result

    def test_parent_allows_verified_completion_within_budget(self):
        self.supervised_fixture(budget=10, expect_timeout=False)

    def test_parent_hard_timeout_bounds_blocking_get_and_removes_scratch(self):
        self.server.block_get = True
        self.supervised_fixture()
        self.assertEqual(self.server.puts, 0)

    def test_parent_hard_timeout_bounds_trickle_body_and_removes_scratch(self):
        self.server.trickle_get = True
        self.supervised_fixture()
        self.assertEqual(self.server.puts, 0)

    def test_bad_resume_range_stops_redacted(self):
        self.server.fail_first = True
        self.server.bad_range = True
        result = self.run_transfer()
        self.assertFalse(result["complete"])
        self.assertEqual(result["failure_code"], "invalid_resumable_range")

    def test_sha_mismatch_prevents_any_upload(self):
        job = dict(self.job, expected_sha256="0" * 64)
        result = self.run_transfer(job)
        self.assertFalse(result["complete"])
        self.assertEqual(result["failure_code"], "download_size_or_sha_mismatch")
        self.assertEqual(self.server.puts, 0)

    def test_account_secret_fields_rejected(self):
        result = self.run_transfer(dict(self.job, api_token=ACCOUNT_CANARY))
        self.assertFalse(result["complete"])
        self.assertEqual(result["failure_code"], "invalid_job_fields")
        self.assertEqual(self.server.request_headers, [])

    def test_cloud_mode_rejects_http_and_unknown_hosts(self):
        with self.assertRaises(transfer.TransferStopped):
            transfer.validate_job(self.job)
        job = dict(self.job, read_url="https://example.com/read", put_url="https://storage.googleapis.com/put")
        with self.assertRaises(transfer.TransferStopped):
            transfer.validate_job(job)

    def test_source_cdn_allowed_but_not_as_upload_host(self):
        job = dict(self.job, read_url="https://www.kaggleusercontent.com/read?file_signature=CANARY",
                   put_url="https://storage.googleapis.com/put?upload_id=CANARY")
        transfer.validate_job(job)
        with self.assertRaises(transfer.TransferStopped):
            transfer.validate_job(dict(job, put_url="https://www.kaggleusercontent.com/put"))
        with self.assertRaises(transfer.TransferStopped):
            transfer.validate_job(dict(job, read_url="https://www.kaggleusercontent.com.evil.invalid/read"))
        with self.assertRaises(transfer.TransferStopped):
            transfer.validate_job(dict(job, read_url="https://user:password@www.kaggleusercontent.com/read"))

    def test_generic_names_prevent_medical_uid_outputs(self):
        result = self.run_transfer(dict(self.job, filename="1.2.3.4.zip.bin"))
        self.assertFalse(result["complete"])
        self.assertEqual(result["failure_code"], "invalid_generic_filename")

    def test_deadline_reader_checks_during_http_body_reads(self):
        path = self.directory / "body.bin"
        path.write_bytes(BODY)
        def expired():
            raise transfer.TransferStopped("walltime_budget")
        with path.open("rb") as stream:
            reader = transfer.DeadlineReader(stream, expired)
            with self.assertRaises(transfer.TransferStopped):
                reader.read(10)


if __name__ == "__main__":
    unittest.main()

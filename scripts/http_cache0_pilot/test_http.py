"""Only localhost fixtures and owned child processes; no Kaggle requests."""
import ast
import base64
import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
import zlib
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

WORK=Path(__file__).resolve().parent
REPO=WORK.parents[1]
sys.path.insert(0,str(WORK.parent/'cache_transfer'))
import transfer
from test_transfer import FixtureHandler,BODY,URL_CANARY,ACCOUNT_CANARY
sys.modules['transfer_reviewed']=transfer
import download
from parent_guard import guarded_launcher


class HTTPFixtures(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='direct_http_fixture_')
        self.root=Path(self.temp.name)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),FixtureHandler)
        for k,v in {'request_headers':[],'received':b'','puts':0,'probes':0,'fail_first':False,'bad_range':False,
                    'resume_header':None,'gets':0,'redirect':False,'bad_download_range':False,'fail_download_once':False,
                    'set_cookie':False,'late_probe_complete':False,'block_get':False,'trickle_get':False}.items():
            setattr(self.server,k,v)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.expected={'bytes':len(BODY),'sha256':hashlib.sha256(BODY).hexdigest()}
        self.job={'job_id':'shard_0','filename':'cache384_train.zip.bin','expected_bytes':len(BODY),
                  'expected_sha256':self.expected['sha256'],'read_url':f'http://127.0.0.1:{self.server.server_port}/read?signature={URL_CANARY}',
                  'max_seconds':600,'source_origin':download.ORIGIN}
        self.job_path=self.root/'read_job.json';self.archive=self.root/'cache384_train.zip.bin';self.output=self.root/'download_report.json'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()

    def run_download(self,job=None,expected=None,success=True):
        self.job_path.write_text(json.dumps(job or self.job))
        with patch.object(download,'DOWNLOAD_RANGE_BYTES',32768):
            if success:
                result=download.download(self.job_path,self.archive,self.output,time.monotonic()+10,expected or self.expected,True)
                self.assertTrue(result['complete']);self.assertEqual(self.archive.read_bytes(),BODY)
            else:
                with self.assertRaises(transfer.TransferStopped):
                    download.download(self.job_path,self.archive,self.output,time.monotonic()+10,expected or self.expected,True)
                result=json.loads(self.output.read_text());self.assertFalse(result['complete']);self.assertFalse(self.archive.exists())
        text=self.output.read_text();self.assertNotIn(URL_CANARY,text);self.assertNotIn(ACCOUNT_CANARY,text);self.assertNotIn('http://',text)
        self.assertEqual(self.server.puts,0)
        for h in self.server.request_headers:self.assertNotIn('Cookie',h);self.assertNotIn('Authorization',h)
        return result

    def test_exact_source_download_cookie_rejection_and_no_upload_branch(self):
        self.server.set_cookie=True
        result=self.run_download();self.assertEqual(len(result['download_range_checks']),4)
        self.assertEqual(result['origin'],download.ORIGIN);self.assertTrue(result['download_sha256_verified'])

    def test_partial_range_retry_preserves_authoritative_full_sha(self):
        self.server.fail_download_once=True
        result=self.run_download();self.assertEqual(result['download_retry_count'],1)

    def test_origin_mismatch_blocks_network(self):
        r=self.run_download(dict(self.job,source_origin=dict(download.ORIGIN,version_number=2)),success=False)
        self.assertEqual(r['failure_code'],'source_origin_mismatch');self.assertEqual(self.server.gets,0)

    def test_job_pin_mismatch_blocks_network(self):
        r=self.run_download(dict(self.job,expected_sha256='0'*64),success=False)
        self.assertEqual(r['failure_code'],'immutable_cache_pin_mismatch');self.assertEqual(self.server.gets,0)

    def test_actual_download_sha_mismatch_removes_incomplete_archive(self):
        job=dict(self.job,expected_sha256='0'*64);r=self.run_download(job,dict(self.expected,sha256='0'*64),success=False)
        self.assertEqual(r['failure_code'],'download_size_or_sha_mismatch')

    def test_secret_job_fields_block_network_and_are_redacted(self):
        r=self.run_download(dict(self.job,account_token=ACCOUNT_CANARY),success=False)
        self.assertEqual(r['failure_code'],'invalid_read_only_job_fields');self.assertEqual(self.server.gets,0)

    def test_range_mismatch_stops_and_removes_incomplete_archive(self):
        self.server.bad_download_range=True
        r=self.run_download(success=False);self.assertEqual(r['failure_code'],'download_content_range_mismatch')

    def test_redirect_not_followed(self):
        self.server.redirect=True
        r=self.run_download(success=False);self.assertEqual(r['failure_code'],'download_range_http_failure');self.assertEqual(self.server.gets,1)

    def test_parent_watchdog_bounds_actual_blocking_range_request(self):
        self.server.block_get=True;self.job_path.write_text(json.dumps(self.job))
        def prepare(scratch):
            download.download(self.job_path,scratch/'cache.bin',self.root/'child_download.json',time.monotonic()+5,self.expected,True)
            return [sys.executable,'-c','pass'],{}
        r=guarded_launcher(prepare,time.monotonic()+.5,self.root/'parent_report.json',self.root/'scratch')
        self.assertFalse(r['complete']);self.assertEqual(r['failure_code'],'hard_walltime_budget')
        self.assertLess(r['seconds'],1.2);self.assertTrue(r['owned_tmp_removed']);self.assertEqual(list((self.root/'scratch').iterdir()),[])
        self.assertGreater(self.server.gets,0);self.assertNotIn(URL_CANARY,(self.root/'parent_report.json').read_text())


class ParentFixtures(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory(prefix='http_parent_fixture_');self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def check_report(self,result):
        self.assertTrue(result['owned_tmp_removed']);self.assertEqual(list((self.root/'scratch').iterdir()),[])
        text=(self.root/'report.json').read_text();self.assertNotIn(URL_CANARY,text);self.assertNotIn(ACCOUNT_CANARY,text)

    def test_parent_kills_owned_nested_process_group_and_cleans_temp(self):
        marker=self.root/'worker_marker.txt'
        worker="from pathlib import Path\nimport time\np=Path("+repr(str(marker))+ ")\nfor _ in range(500):\n p.write_text(str(time.monotonic()))\n time.sleep(.01)\n"
        def prepare(scratch):
            source='import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",'+repr(worker)+'])\nprint('+repr(URL_CANARY)+',flush=True)\ntime.sleep(5)\n'
            script=scratch/'child.py';script.write_text(source)
            return [sys.executable,str(script)],{}
        result=guarded_launcher(prepare,time.monotonic()+.5,self.root/'report.json',self.root/'scratch')
        self.assertFalse(result['complete']);self.assertEqual(result['failure_code'],'hard_walltime_budget')
        self.assertTrue(result['owned_process_group_terminated']);self.assertLess(result['seconds'],1.2);self.assertTrue(marker.exists())
        after=marker.read_text();time.sleep(.15);self.assertEqual(marker.read_text(),after);self.check_report(result)

    def test_pre_child_failure_is_redacted_and_owned_scratch_cleaned(self):
        def prepare(scratch):
            (scratch/'private_partial.bin').write_bytes(BODY)
            raise RuntimeError(URL_CANARY)
        result=guarded_launcher(prepare,time.monotonic()+1,self.root/'report.json',self.root/'scratch')
        self.assertFalse(result['complete']);self.assertEqual(result['failure_code'],'redacted_parent_preparation_failure');self.check_report(result)

    def test_launcher_exit_does_not_leave_its_owned_nested_worker_running(self):
        marker=self.root/'orphan_marker.txt';group_file=self.root/'owned_group.txt'
        worker="from pathlib import Path\nimport time\np=Path("+repr(str(marker))+ ")\nfor _ in range(500):\n p.write_text(str(time.monotonic()))\n time.sleep(.01)\n"
        def prepare(scratch):
            source='import subprocess,sys,os,time\nfrom pathlib import Path\nPath('+repr(str(group_file))+').write_text(str(os.getpgrp()))\nsubprocess.Popen([sys.executable,"-c",'+repr(worker)+'])\ntime.sleep(.15)\nraise SystemExit(7)\n'
            script=scratch/'child.py';script.write_text(source)
            return [sys.executable,str(script)],{}
        try:
            result=guarded_launcher(prepare,time.monotonic()+2,self.root/'report.json',self.root/'scratch')
            self.assertFalse(result['complete']);self.assertEqual(result['child_exit_code'],7)
            self.assertTrue(result['owned_process_group_terminated']);self.assertTrue(marker.exists())
            after=marker.read_text();time.sleep(.15);self.assertEqual(marker.read_text(),after);self.check_report(result)
        finally:
            if group_file.exists():
                try:os.killpg(int(group_file.read_text()),signal.SIGKILL)
                except ProcessLookupError:pass

    def test_deadline_crossed_during_preparation_cleans_before_starting_child(self):
        def prepare(scratch):
            time.sleep(.1)
            return [sys.executable,'-c',"raise SystemExit('must not start')"],{}
        result=guarded_launcher(prepare,time.monotonic()+.05,self.root/'report.json',self.root/'scratch')
        self.assertFalse(result['complete']);self.assertEqual(result['failure_code'],'hard_walltime_budget')
        self.assertNotIn('child_exit_code',result);self.check_report(result)

    def test_blocking_preparation_is_hard_bounded_before_any_command(self):
        def prepare(scratch):
            (scratch/'partial_receipt.json').write_text('private fixture')
            time.sleep(.65)
            return [sys.executable,'-c','pass'],{}
        r=guarded_launcher(prepare,time.monotonic()+.05,self.root/'report.json',self.root/'scratch')
        self.assertFalse(r['complete']);self.assertEqual(r['failure_code'],'hard_walltime_budget')
        self.assertLess(r['seconds'],.2);self.check_report(r)

    def test_exact_generated_cache_view_and_default_temp_removed_after_sigkill(self):
        import nbformat
        n=nbformat.read(REPO/'notebooks/orthofoundation/http_cache0_pilot/train.ipynb',as_version=4)
        values={item.targets[0].id:ast.literal_eval(item.value) for item in ast.parse(n.cells[2].source).body
                if isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Name) and item.targets[0].id=='CHILD_SOURCE'}
        child=values['CHILD_SOURCE'];tree=ast.parse(child)
        fn=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='_owned_launcher_mkdtemp')
        call=next(node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                  and node.func.attr=='mkdtemp' and any(k.arg=='prefix' and isinstance(k.value,ast.Constant)
                  and k.value.value=='rsna-knee-cache-mount-' for k in node.keywords))
        exact_expression=ast.get_source_segment(child,call);self.assertIn("dir='/tmp'",exact_expression)
        marker=self.root/'view_paths.json';target=self.root/'archive_fixture.bin';target.write_bytes(BODY)
        code='import tempfile,os,time,json\nfrom pathlib import Path\n_real_mkdtemp=tempfile.mkdtemp\n'+ast.get_source_segment(child,fn)+'\ntempfile.mkdtemp=_owned_launcher_mkdtemp\n'
        code+='view=Path('+exact_expression+')\n(view/"private_receipt.json").write_text("private fixture")\n(view/"cache.zip").symlink_to('+repr(str(target))+')\n'
        code+='default=Path(tempfile.mkdtemp(prefix="private-default-fixture-"))\n(default/"private.bin").write_bytes(b"private")\n'
        code+='Path('+repr(str(marker))+').write_text(json.dumps([str(view),str(default)]))\ntime.sleep(5)\n'
        def prepare(scratch):
            script=scratch/'launcher.py';script.write_text(code)
            return [sys.executable,str(script)],{}
        r=guarded_launcher(prepare,time.monotonic()+.5,self.root/'report.json',self.root/'scratch')
        self.assertFalse(r['complete']);self.assertTrue(marker.exists());self.assertTrue(r['owned_process_group_terminated'])
        paths=[Path(x) for x in json.loads(marker.read_text())]
        self.assertTrue(all(str(p).startswith(str(self.root/'scratch')) for p in paths))
        self.assertTrue(all(not p.exists() for p in paths));self.assertTrue(target.exists());self.check_report(r)

    def test_account_environment_removed_before_verified_child_completion(self):
        def prepare(scratch):
            script=scratch/'child.py';script.write_text("import os\nassert not any(os.environ.get(k) for k in ('KAGGLE_API_TOKEN','KAGGLE_KEY','KAGGLE_USERNAME'))\n")
            return [sys.executable,str(script)],{}
        env=dict(os.environ,KAGGLE_API_TOKEN=ACCOUNT_CANARY,KAGGLE_KEY=ACCOUNT_CANARY,KAGGLE_USERNAME='privatefixture')
        result=guarded_launcher(prepare,time.monotonic()+2,self.root/'report.json',self.root/'scratch',env=env)
        self.assertTrue(result['complete']);self.check_report(result)

    def test_scientific_payload_config_and_six_hundred_second_pins_unchanged(self):
        import nbformat
        prepared=nbformat.read(REPO/'notebooks/orthofoundation/http_cache0_pilot/train.ipynb',as_version=4)
        nbformat.validate(prepared)
        parent={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(prepared.cells[2].source).body
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
        child=parent['CHILD_SOURCE'];values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(child).body
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['SETTINGS','COMPRESSED_PAYLOAD']}
        s=values['SETTINGS'];self.assertEqual(s['max_seconds'],600);self.assertEqual(s['config']['checkpoint_sha256'],'385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9')
        receipt=json.loads((REPO/'experiments/orthofoundation_http_cache0_pilot/preparation_receipt.json').read_text());self.assertEqual(s['source_sha256'],receipt['scientific_source_sha256'])
        payload=json.loads(zlib.decompress(base64.b64decode(values['COMPRESSED_PAYLOAD'])));self.assertEqual(len(payload),8)
        for name,encoded in payload.items():self.assertEqual(hashlib.sha256(base64.b64decode(encoded)).hexdigest(),s['source_sha256'][name])
        metadata=json.loads((REPO/'notebooks/orthofoundation/http_cache0_pilot/kernel-metadata.json').read_text());self.assertTrue(metadata['enable_internet']);self.assertTrue(metadata['is_private']);self.assertEqual(metadata['kernel_sources'],[])
        self.assertNotIn('alanchoo/rsna-knee-cache-0-bin-v1',metadata['dataset_sources'])


if __name__=='__main__':unittest.main()

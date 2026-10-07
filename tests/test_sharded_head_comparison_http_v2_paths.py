"""V2 aliases must record executed source paths in local and vendored layouts."""
import ast
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_sharded_head_comparison_notebook_http_v2 as BUILDER
import compare_sharded_heads_http_v2 as COMPARE
from test_sharded_fallback import fixture_plan


class VersionedSourcePaths(unittest.TestCase):
    def test_adapter_diff_is_only_import_and_actual_source_provenance_paths(self):
        original = (ROOT / "scripts/compare_sharded_heads.py").read_text()
        expected = original.replace("import sharded_frozen_extract as RUNNER",
            "from orthofoundation_http_v2 import sharded_frozen_extract as RUNNER")
        previous = ('    for name in ("sharded_frozen_extract.py", "compare_frozen_heads.py", "compare_sharded_heads.py"):\n'
                    '        source_hashes[name] = file_sha(Path(__file__).with_name(name))\n')
        actual = ('    source_hashes.update({\n'
                  '        "orthofoundation_http_v2/sharded_frozen_extract.py": file_sha(Path(RUNNER.__file__)),\n'
                  '        "compare_frozen_heads.py": file_sha(Path(__file__).with_name("compare_frozen_heads.py")),\n'
                  '        "compare_sharded_heads.py": file_sha(__file__),\n'
                  '    })\n')
        self.assertEqual(expected.count(previous), 1)
        self.assertEqual(Path(COMPARE.__file__).read_text(), expected.replace(previous, actual))
        self.assertEqual(hashlib.sha256(Path(COMPARE.RUNNER.__file__).read_bytes()).hexdigest(),
            "86265241778c40a3822c2a8d11f4f3f746a48b47101def338120dd86937ee4f2")
        old_meanmax = yaml.safe_load((ROOT / "configs/orthofoundation_meanmax.yaml").read_text())
        new_meanmax = yaml.safe_load((ROOT / "configs/orthofoundation_meanmax_http_v2.yaml").read_text())
        baseline = yaml.safe_load((ROOT / "configs/orthofoundation_frozen_http_v2.yaml").read_text())
        self.assertEqual(old_meanmax.keys(), new_meanmax.keys())
        self.assertEqual({key for key in old_meanmax if old_meanmax[key] != new_meanmax[key]},
                         {"experiment", "max_extract_seconds"})
        self.assertEqual(BUILDER.controlled_config(baseline, new_meanmax), new_meanmax)

    def test_cloud_namespace_without_v1_root_runner_completes_synthetic_head_only_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan, _, _ = fixture_plan()
            global_root = base / "global"; global_root.mkdir()
            (global_root / "global_metadata_manifest.json").write_text("{}")
            docker = base / "docker.json"; docker.write_text(json.dumps({"docker_image": "fixture_image"}))
            argv = ["builder", "--global-root", str(global_root), "--assets-manifest-sha256", "a" * 64,
                    "--docker-metadata", str(docker), "--output-dir", str(base / "prepared"), "--prepare-placeholder"]
            with patch.object(sys, "argv", argv), patch.object(BUILDER.RUNNER, "validate_global", return_value=plan), \
                    contextlib.redirect_stdout(io.StringIO()): BUILDER.main()
            notebook = nbformat.read(base / "prepared/compare.ipynb", as_version=4)
            cell = [row.source for row in notebook.cells if row.cell_type == "code"][1]
            literals = {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(cell).body
                        if isinstance(node, ast.Assign)}
            payload = json.loads(zlib.decompress(base64.b64decode(literals["COMPRESSED_PAYLOAD"])))
            vendor = base / "vendor"; vendor.mkdir()
            for name, encoded in payload.items():
                path = vendor / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(encoded))
            self.assertFalse((vendor / "sharded_frozen_extract.py").exists())
            script = vendor / "cloud_contract.py"
            script.write_text('''import contextlib,hashlib,io,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
torch.set_num_threads(1)
import compare_sharded_heads as C
assert Path(C.__file__).resolve().parent==Path(__file__).resolve().parent, (C.__file__,__file__)
assert Path(C.RUNNER.__file__).resolve().parent==Path(__file__).resolve().parent/'orthofoundation_http_v2', C.RUNNER.__file__
from test_sharded_head_comparison_http_v2 import ShardedHeadComparisonHTTPV2Contracts
base=Path(sys.argv[1]);plan,_,config,identity,provenance,previous,global_root,_=ShardedHeadComparisonHTTPV2Contracts().fixture(base)
pin=C.validate_sharded_bank(plan,previous/'features',identity,provenance)['sha256']
with patch.object(C.RUNNER,'validate_global',return_value=plan),patch.object(C.RUNNER,'extract',side_effect=AssertionError('encoder forbidden')),patch.object(C.RUNNER,'load_encoder',side_effect=AssertionError('encoder forbidden')),contextlib.redirect_stdout(io.StringIO()):
 report=C.run_comparison(previous,global_root,'synthetic_global_sha','a'*64,identity,base/'comparison',config,'cpu',expected_bank_sha256=pin)
sources=report['source_sha256']
assert sources['compare_sharded_heads.py']==hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()
assert sources['orthofoundation_http_v2/sharded_frozen_extract.py']==hashlib.sha256(Path(C.RUNNER.__file__).read_bytes()).hexdigest()
assert 'sharded_frozen_extract.py' not in sources
assert report['encoder_forward_calls']==0
assert report['production_head']['epochs']==report['weak_holdout_head']['epochs']==12
assert report['production_head']['gold_used_for_gradients']==0
print(json.dumps({'vendored_comparison_complete':True,'actual_alias_sources_hashed':True,'encoder_forward_calls':0}))
''')
            env = dict(os.environ, PYTHONPATH=str(vendor) + os.pathsep + str(ROOT / "tests"),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            result = subprocess.run([sys.executable, str(script), str(base / "bank")], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["vendored_comparison_complete"])


if __name__ == "__main__":
    unittest.main()

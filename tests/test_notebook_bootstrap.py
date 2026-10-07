"""Bootstrap evidence must exist before dependent imports and mount checks."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("orthofoundation_notebook_builder", ROOT / "scripts/build_orthofoundation_notebook.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class NotebookBootstrapDiagnostics(unittest.TestCase):
    def execute_failure(self, stage, statement, expected_type):
        with tempfile.TemporaryDirectory() as directory:
            namespace = {}
            start = BUILDER.bootstrap_start_code().replace("BOOTSTRAP_DIR='/kaggle/working'", "BOOTSTRAP_DIR=" + repr(directory))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(start, "bootstrap-start", "exec"), namespace)
                evidence = json.loads((Path(directory) / "bootstrap_start.json").read_text())
                self.assertEqual(evidence["event"], "bootstrap_start")
                code = BUILDER.instrument_bootstrap("bootstrap_checkpoint(" + repr(stage) + ")\n" + statement + "\n", "literal_payload", {"stage": "pilot"})
                with self.assertRaises(expected_type):
                    exec(compile(code, "bootstrap-launch", "exec"), namespace)
            failure = json.loads((Path(directory) / "bootstrap_failure.json").read_text())
            progress = json.loads((Path(directory) / "bootstrap_progress.json").read_text())
            self.assertEqual(failure["stage"], stage)
            self.assertEqual(progress["stage"], stage)
            self.assertEqual(failure["error_type"], expected_type.__name__)
            self.assertIn("bootstrap-launch", failure["traceback"])
            return code

    def test_import_failure_persists_start_and_failure_then_reraises(self):
        self.execute_failure("imports", "import module_that_does_not_exist_for_bootstrap_test", ModuleNotFoundError)

    def test_missing_mount_failure_has_phase_and_top_level_literals(self):
        code = self.execute_failure("input_resolution", "raise FileNotFoundError('missing synthetic mount')", FileNotFoundError)
        tree = ast.parse(code)
        values = {statement.targets[0].id: ast.literal_eval(statement.value) for statement in tree.body if isinstance(statement, ast.Assign)}
        self.assertEqual(values["COMPRESSED_PAYLOAD"], "literal_payload")
        self.assertEqual(values["SETTINGS"], {"stage": "pilot"})


if __name__ == "__main__":
    unittest.main()

"""Private parent-side deadline and cleanup for the single-file child utility."""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def save_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


def supervise(runner, job_path, output, scratch_parent="/tmp"):
    """Kill the owned child at its total budget, then remove its owned scratch.

    stdout/stderr are captured and never reproduced: a dependency exception may
    contain a temporary signed URL. The child only creates its own local files.
    """
    started = time.monotonic()
    owned_scratch = None
    report = {"complete": False, "cloud_account_token_used": False,
              "urls_or_upload_tokens_written_to_output": False,
              "supervisor_hard_timeout_enabled": True}
    try:
        try:
            job = json.loads(Path(job_path).read_text())
            budget = job.get("max_seconds", 1800)
            if type(budget) is not int or not 1 <= budget <= 7200:
                raise ValueError("Invalid time budget")
        except BaseException as error:
            report.update(failure_stage="supervisor_job_load", failure_code="invalid_private_job",
                          exception_type=type(error).__name__)
            return report
        deadline = started + budget
        report["max_seconds"] = budget
        parent = Path(scratch_parent)
        parent.mkdir(parents=True, exist_ok=True)
        owned_scratch = Path(tempfile.mkdtemp(prefix="cache_transfer_supervisor_", dir=parent))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("redacted_transfer_child", budget)
        result = subprocess.run([sys.executable, str(runner), "--job", str(job_path),
                                 "--output", str(output), "--scratch-parent", str(owned_scratch)],
                                timeout=remaining, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired("redacted_transfer_child", budget)
        if Path(output).is_file():
            child_report = json.loads(Path(output).read_text())
            if not isinstance(child_report, dict) or type(child_report.get("complete")) is not bool:
                raise ValueError("Invalid child report")
            report.update(child_report)
        else:
            report.update(failure_stage="supervisor_child", failure_code="missing_child_report")
        report["child_exit_code"] = result.returncode
        if result.returncode:
            report["complete"] = False
            report.setdefault("failure_code", "redacted_child_failure")
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for this one child before raising.
        report.update(complete=False, failure_stage="supervisor_deadline",
                      failure_code="hard_walltime_budget", child_terminated=True)
    except BaseException as error:
        report.update(complete=False, failure_stage="supervisor",
                      failure_code="redacted_supervisor_exception", exception_type=type(error).__name__)
    finally:
        if owned_scratch is not None:
            try:
                shutil.rmtree(owned_scratch)
            except OSError as error:
                report.update(complete=False, cleanup_exception_type=type(error).__name__)
        report["supervisor_owned_scratch_removed"] = owned_scratch is None or not owned_scratch.exists()
        if report.get("child_terminated"):
            report["own_temp_archive_removed"] = report["supervisor_owned_scratch_removed"]
        report["supervisor_total_seconds"] = time.monotonic() - started
        save_report(report, output)
    return report

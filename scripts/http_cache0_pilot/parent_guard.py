"""Own one launcher process group and scratch root; redact all child output."""
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path


def guarded_launcher(prepare, deadline, output, scratch_parent="/tmp", env=None):
    started = time.monotonic()
    scratch, child_pid, child_status = None, None, None
    report = {"complete": False, "route": "direct_http",
              "account_credentials_given_to_child": False,
              "raw_child_stdout_stderr_published": False,
              "hard_process_group_deadline": True}
    def stop_owned_group():
        nonlocal child_status
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
                report["owned_process_group_terminated"] = True
                if child_status is not None:
                    report["owned_worker_group_outlived_launcher"] = True
                    if report.get("complete"):
                        report.update(complete=False, failure_code="orphaned_owned_worker")
            except ProcessLookupError:
                # A deadline can race the owned child before its first setsid.
                # Never signal the inherited parent group; target this PID only.
                if child_status is None:
                    try:os.kill(child_pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            if child_status is None:
                _,child_status=os.waitpid(child_pid,0)
    try:
        parent = Path(scratch_parent)
        parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="rsna-http-cache0-", dir=parent))
        if time.monotonic()>=deadline:raise TimeoutError()
        child_env = dict(os.environ if env is None else env)
        for field in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME"):
            child_env.pop(field, None)
        child_pid=os.fork()
        if child_pid==0:
            # Every preparation/input read/copy/compile runs under the parent's
            # watchdog. The scientific command then replaces this group leader.
            try:
                os.setsid()
                own_temp=scratch/'temporary';own_temp.mkdir()
                child_env.update(TMPDIR=str(own_temp),TMP=str(own_temp),TEMP=str(own_temp),
                                 RSNA_HTTP_OWNED_TMP=str(own_temp))
                os.environ.clear();os.environ.update(child_env);tempfile.tempdir=str(own_temp)
                with (scratch/'child_capture.log').open('wb') as capture:
                    os.dup2(capture.fileno(),1);os.dup2(capture.fileno(),2)
                cmd,additions=prepare(scratch)
                child_env.update(additions)
                for field in ('KAGGLE_API_TOKEN','KAGGLE_KEY','KAGGLE_USERNAME'):child_env.pop(field,None)
                (scratch/'preparation_complete').write_text('complete')
                os.execvpe(cmd[0],cmd,child_env)
            except BaseException:
                os._exit(125)
        while True:
            waited,status=os.waitpid(child_pid,os.WNOHANG)
            if waited:
                child_status=status;break
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError()
            time.sleep(min(.01,remaining))
        if time.monotonic()>=deadline:raise TimeoutError()
        returncode=os.waitstatus_to_exitcode(child_status)
        report.update(child_exit_code=returncode, complete=returncode == 0)
        if returncode:
            report["failure_code"] = 'redacted_owned_child_failure' if (scratch/'preparation_complete').exists() else 'redacted_parent_preparation_failure'
    except TimeoutError:
        stop_owned_group()
        report.update(complete=False, failure_code="hard_walltime_budget")
    except BaseException as error:
        report.update(complete=False, failure_code="redacted_parent_preparation_failure",
                      exception_type=type(error).__name__)
    finally:
        stop_owned_group()
        if scratch is not None:
            try:
                shutil.rmtree(scratch)
            except OSError as error:
                report.update(complete=False, cleanup_exception_type=type(error).__name__)
        report["owned_tmp_removed"] = scratch is None or not scratch.exists()
        report["seconds"] = time.monotonic() - started
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(output)
    return report

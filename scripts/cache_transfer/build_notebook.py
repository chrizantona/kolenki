"""Prepare a private CPU transfer notebook; does not call any remote API."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import nbformat


def build(folder, job_dataset, slug, docker_image):
    if not re.fullmatch(r"alanchoo/[a-z0-9-]+", job_dataset):
        raise ValueError("Private job dataset must be in the authorized account")
    if not re.fullmatch(r"rsna-knee-cache-transfer-shard-[1-6]", slug):
        raise ValueError("Use one new transfer kernel per remaining shard")
    source = Path(__file__).with_name("transfer.py").read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    supervisor = Path(__file__).with_name("supervise.py").read_text()
    supervisor_digest = hashlib.sha256(supervisor.encode()).hexdigest()
    code = (
        "from pathlib import Path\nimport hashlib,json,subprocess,sys\n"
        f"SOURCE={source!r}\nSOURCE_SHA={digest!r}\nJOB_DATASET={job_dataset!r}\n"
        f"SUPERVISOR_SOURCE={supervisor!r}\nSUPERVISOR_SHA={supervisor_digest!r}\n"
        "if hashlib.sha256(SOURCE.encode()).hexdigest()!=SOURCE_SHA:raise RuntimeError('Source SHA mismatch')\n"
        "if hashlib.sha256(SUPERVISOR_SOURCE.encode()).hexdigest()!=SUPERVISOR_SHA:raise RuntimeError('Supervisor SHA mismatch')\n"
        "runner=Path('/kaggle/working/transfer.py');runner.write_text(SOURCE)\n"
        "owner,slug=JOB_DATASET.split('/')\nINPUT=Path('/kaggle/input')\n"
        "candidates=[INPUT/'datasets'/owner/slug,INPUT/slug,INPUT/owner/slug]\n"
        "roots=[p for p in candidates if (p/'transfer_job.json').is_file()]\n"
        "if not roots:\n"
        "    for pattern in [f'*/{slug}',f'*/*/{slug}',f'*/*/*/{slug}']:\n"
        "        roots.extend(p for p in INPUT.glob(pattern) if (p/'transfer_job.json').is_file())\n"
        "roots=list(dict.fromkeys(roots))\n"
        "if len(roots)!=1:raise RuntimeError('Missing or ambiguous private transfer job dataset')\n"
        "supervisor_namespace={'__name__':'private_transfer_supervisor'}\n"
        "exec(compile(SUPERVISOR_SOURCE,'private_transfer_supervisor','exec'),supervisor_namespace)\n"
        "result=supervisor_namespace['supervise'](runner,roots[0]/'transfer_job.json','/kaggle/working/transfer_report.json',scratch_parent='/tmp')\n"
        "print(json.dumps(result),flush=True)\n"
        "if not result['complete']:raise RuntimeError('Private transfer failed; see redacted receipt')\n"
    )
    ast.parse(source)
    ast.parse(supervisor)
    ast.parse(code)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        "Private, single-file CPU transfer utility. Temporary file-scoped URLs are read from a private input, "
        "never embedded in source or written to output. The account API credential and opaque blob token stay on the Mac. "
        "No GPU, MRI processing, dataset creation, or competition submission occurs here."),
        nbformat.v4.new_code_cell(code)], metadata={"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"},
                                                 "language_info": {"name": "python"}})
    nbformat.validate(notebook)
    metadata = {"id": "alanchoo/" + slug, "title": slug.replace("-", " ").title(), "code_file": "transfer.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                "competition_sources": [], "kernel_sources": [], "dataset_sources": [job_dataset],
                "docker_image": docker_image, "docker_image_pinning_type": "original"}
    if re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-") != slug:
        raise ValueError("Title must normalize to exact kernel slug")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, folder / "transfer.ipynb")
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    receipt = {"prepared_not_launched": True, "kernel_slug": slug, "job_dataset": job_dataset,
               "source_sha256": digest, "notebook_sha256": hashlib.sha256((folder / "transfer.ipynb").read_bytes()).hexdigest(),
               "supervisor_sha256": supervisor_digest,
               "contains_scoped_urls_or_account_tokens": False, "GPU": False, "internet": True}
    (folder / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--job-dataset", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--docker-image", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.folder, args.job_dataset, args.slug, args.docker_image), indent=2))

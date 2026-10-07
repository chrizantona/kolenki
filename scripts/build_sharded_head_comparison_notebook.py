#!/usr/bin/env python3
"""Prepare EXP-OF-002 for a future complete sharded attention output; no API."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import zlib

import nbformat
import yaml

import sharded_frozen_extract as RUNNER
from compare_frozen_heads import controlled_config
from build_orthofoundation_notebook import bootstrap_start_code, instrument_bootstrap

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = ("__init__.py", "cache.py", "data.py", "encoder.py", "heads.py", "run.py", "train.py")

LAUNCH_BODY = r'''bootstrap_checkpoint('imports')
from pathlib import Path
import base64,hashlib,json,os,re,shutil,subprocess,sys,tarfile,time,zlib
bootstrap_checkpoint('sealed_source_bank_pin')
if not SETTINGS['comparison_launch_allowed'] or not isinstance(SETTINGS['source_bank_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',SETTINGS['source_bank_sha256']):
    raise RuntimeError('Placeholder only: root must seal completed immutable attention full-bank SHA256 and rebuild before comparison')
CODE=Path('/kaggle/working/sharded_compare_source'); CODE.mkdir(exist_ok=True)
bootstrap_checkpoint('vendoring')
for name,encoded in json.loads(zlib.decompress(base64.b64decode(COMPRESSED_PAYLOAD))).items():
    path=CODE/name; path.parent.mkdir(exist_ok=True); path.write_bytes(base64.b64decode(encoded))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SETTINGS['source_sha256'][name]:
        raise RuntimeError('Vendored comparison source SHA differs: '+name)
(CODE/'config.json').write_text(json.dumps(SETTINGS['config'],indent=2))
INPUT=Path('/kaggle/input')
def find_input(identifier,marker,kind='datasets'):
    owner,slug=identifier.split('/')
    candidates=[INPUT/kind/owner/slug,INPUT/slug,INPUT/'kernels'/owner/slug,INPUT/owner/slug]
    found=[path for path in candidates if (path/marker).is_file()]
    if not found:
        for pattern in [f'*/{slug}',f'*/*/{slug}',f'*/*/*/{slug}']:
            found.extend(path for path in INPUT.glob(pattern) if (path/marker).is_file())
    found=list(dict.fromkeys(found))
    if len(found)!=1:raise RuntimeError(f'Missing/ambiguous private comparison input {identifier}: {found}')
    return found[0]
bootstrap_checkpoint('global_metadata_resolution')
metadata=find_input(SETTINGS['metadata_dataset'],'global_metadata_manifest.json')
manifest_file=metadata/'global_metadata_manifest.json'
if hashlib.sha256(manifest_file.read_bytes()).hexdigest()!=SETTINGS['global_manifest_sha256']:
    raise RuntimeError('Global metadata manifest differs from prepared comparison pin')
manifest=json.loads(manifest_file.read_text()); archive=metadata/manifest['archive']['filename']
if archive.stat().st_size!=manifest['archive']['bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['archive']['sha256']:
    raise RuntimeError('Small global metadata archive bytes/SHA differ')
DATA=Path('/kaggle/working/global_metadata'); DATA.mkdir(exist_ok=True)
bootstrap_checkpoint('global_metadata_unpack')
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers(); files=[]
    for member in members:
        path=Path(member.name)
        if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            raise RuntimeError('Unsafe global metadata archive member')
        if member.isfile():files.append(member.name)
    if len(files)!=len(set(files)) or set(files)!={row['path'] for row in manifest['members']}:
        raise RuntimeError('Global metadata archive members differ')
    tar.extractall(DATA,filter='data')
shutil.copyfile(manifest_file,DATA/'global_metadata_manifest.json')
bootstrap_checkpoint('completed_source_bank_resolution')
previous=find_input(SETTINGS['previous_kernel'],'run/training_report.json','notebooks')/'run'
env=dict(os.environ); env['PYTHONPATH']=str(CODE)+os.pathsep+env.get('PYTHONPATH','')
env['OMP_NUM_THREADS']='1'; env['OPENBLAS_NUM_THREADS']='1'
env['RSNA_SHARDED_COMPARE_BOOTSTRAP_START']=str(BOOTSTRAP_MONOTONIC)
if SETTINGS['device']=='cuda':
    bootstrap_checkpoint('gpu_selection')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for explicitly selected T4 comparison')
    free=[torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    env['CUDA_VISIBLE_DEVICES']=str(max(range(len(free)),key=lambda index:free[index]))
cmd=[sys.executable,str(CODE/'compare_sharded_heads.py'),'--previous-run-dir',str(previous),'--global-root',str(DATA),
     '--global-manifest-sha256',SETTINGS['global_manifest_sha256'],'--assets-manifest-sha256',SETTINGS['assets_manifest_sha256'],
     '--expected-feature-identity',SETTINGS['source_feature_identity'],'--output-dir','/kaggle/working/run',
     '--source-bank-sha256',SETTINGS['source_bank_sha256'],
     '--config',str(CODE/'config.json'),'--device',SETTINGS['device'],'--max-seconds',str(SETTINGS['max_seconds'])]
bootstrap_checkpoint('launch')
remaining=SETTINGS['max_seconds']-(time.monotonic()-BOOTSTRAP_MONOTONIC)
if remaining<=0:raise RuntimeError('Comparison Python-stage budget exhausted before child')
subprocess.run(cmd,env=env,check=True,timeout=remaining)
bootstrap_checkpoint('report')
report=json.loads(Path('/kaggle/working/run/training_report.json').read_text())
if not report['head_training_complete'] or report['encoder_forward_calls']!=0:
    raise RuntimeError('Fixed head-only sharded comparison incomplete')
print(json.dumps({'training_complete':True,'baseline_attention':report['baseline_attention'],
 'meanmax_weak_holdout_metrics':report['weak_holdout_metrics'],'meanmax_gold_metrics':report['gold_metrics'],
 'macro_auc_delta':report['macro_auc_delta_meanmax_minus_attention'],
 'per_label_auc_delta':report['per_label_auc_delta_meanmax_minus_attention'],'total_seconds':report['total_seconds']},indent=2),flush=True)
bootstrap_checkpoint('complete')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-root", type=Path, required=True)
    parser.add_argument("--assets-manifest-sha256", required=True, help="Existing encoder asset pin; no assets attached")
    parser.add_argument("--docker-metadata", type=Path, required=True)
    parser.add_argument("--previous-kernel", default="alanchoo/rsna-knee-ortho-dataset-merge-heads")
    parser.add_argument("--metadata-dataset", default="alanchoo/rsna-knee-ortho-global-metadata-v1")
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "configs/orthofoundation_frozen.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orthofoundation_meanmax.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--owner", default="alanchoo")
    parser.add_argument("--slug", default="rsna-knee-ortho-sharded-meanmax")
    parser.add_argument("--output-dir", type=Path)
    pin_group = parser.add_mutually_exclusive_group(required=True)
    pin_group.add_argument("--source-bank-sha256", help="Trusted pin sealed by root from completed immutable attention output")
    pin_group.add_argument("--prepare-placeholder", action="store_true", help="Non-executable preparation until the real full-bank pin exists")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.assets_manifest_sha256):
        raise RuntimeError("Existing encoder manifest SHA256 must be explicitly pinned")
    if args.source_bank_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", args.source_bank_sha256):
        raise RuntimeError("External attention source-bank SHA256 pin is invalid")
    baseline = yaml.safe_load(args.baseline_config.read_text())
    config = controlled_config(baseline, yaml.safe_load(args.config.read_text()))
    global_sha = RUNNER.file_sha(args.global_root / "global_metadata_manifest.json")
    plan = RUNNER.validate_global(args.global_root, global_sha)
    identity, _ = RUNNER.make_identity(plan, baseline, args.assets_manifest_sha256)
    title = args.slug.replace("-", " ").title()
    if re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") != args.slug:
        raise RuntimeError("Comparison title must normalize to exact referenced slug")
    if args.previous_kernel == args.owner + "/" + args.slug:
        raise RuntimeError("Comparison cannot read its own output")
    paths = {"orthofoundation/" + name: ROOT / "src/orthofoundation" / name for name in CORE_MODULES}
    for name in ("sharded_frozen_extract.py", "compare_frozen_heads.py", "compare_sharded_heads.py"):
        paths[name] = ROOT / "scripts" / name
    payload = {name: base64.b64encode(path.read_bytes()).decode() for name, path in paths.items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    compressed = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    settings = {"config": config, "source_sha256": hashes, "source_feature_identity": identity,
        "global_manifest_sha256": global_sha, "assets_manifest_sha256": args.assets_manifest_sha256,
        "metadata_dataset": args.metadata_dataset, "previous_kernel": args.previous_kernel, "device": args.device,
        "encoder_forward_calls": 0, "max_seconds": 1500, "head_fit_budget_seconds": 1200, "validation_reserve_seconds": 300,
        "source_bank_required_complete": True, "source_bank_ready_verified_during_preparation": False}
    settings.update(source_bank_sha256=args.source_bank_sha256, comparison_launch_allowed=args.source_bank_sha256 is not None,
        source_bank_pin_origin="Root must seal full completed immutable attention output before building executable comparison; builder never measures expected pin from current bank")
    code = instrument_bootstrap(LAUNCH_BODY, compressed, settings)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        "# EXP-OF-002: attention versus mean+max on the complete sharded bank\n\n"
        "Preparation only; this follow-up requires completed attention heads and all 4,407 merged studies. "
        "Native sharded provenance, all seven cache pins, global selection and both source heads are verified before fitting. "
        "Requires an independently sealed full attention-source bank SHA256, binding scored and unscored training rows. "
        "Both attention checkpoints replay weak-holdout/Gold probabilities against their original CSV with declared FP32 device tolerances. "
        "A placeholder with no real bank pin refuses execution before any data resolution or fitting. "
        "Only head_kind and experiment change; 12 epochs, split, seed and optimizer stay fixed. "
        "Same seed does not guarantee identical minibatch order because initialization consumes RNG differently. "
        "No MRI/encoder/weight assets mounted; the existing bank and small private metadata are sufficient. "
        "Weak AUC measures teacher agreement; Gold58 is diagnostic and excluded from gradients/checkpoint selection. "
        "The 1,500-second Python-stage cap reserves 300 seconds for validation and 1,200 for the unchanged head fit; "
        "pre-Python Kaggle mounting is outside this timer. Prediction/UID outputs stay private."),
        nbformat.v4.new_code_cell("import time\nBOOTSTRAP_MONOTONIC=time.monotonic()\n" + bootstrap_start_code()),
        nbformat.v4.new_code_cell(code)], metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.13"}})
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code": ast.parse(cell.source)
    metadata = {"id": args.owner + "/" + args.slug, "title": title, "code_file": "compare.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": args.device == "cuda", "enable_tpu": False,
        "enable_internet": False, "competition_sources": [], "dataset_sources": [args.metadata_dataset],
        "kernel_sources": [args.previous_kernel], "docker_image": json.loads(args.docker_metadata.read_text())["docker_image"],
        "docker_image_pinning_type": "original"}
    if args.device == "cuda": metadata["machine_shape"] = "NvidiaTeslaT4"
    directory = args.output_dir or ROOT / "artifacts/kaggle" / args.slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "compare.ipynb").write_text(nbformat.writes(notebook))
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "stage_config.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps({"directory": str(directory), "source_feature_identity": identity, "device": args.device,
                      "source_bank_ready": False, "preparation_only": True,
                      "comparison_launch_allowed": settings['comparison_launch_allowed']}), flush=True)


if __name__ == "__main__":
    main()

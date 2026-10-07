#!/usr/bin/env python3
"""Prepare private Kaggle sharded jobs; no API calls and no patient rows embedded."""
import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import zlib

import nbformat
import yaml

from build_orthofoundation_notebook import bootstrap_start_code, instrument_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sharded_frozen_runner", ROOT / "scripts/sharded_frozen_extract.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def title_for_slug(slug):
    """Kaggle new-kernel creation derives the actual slug from its title."""
    title = slug.replace("-", " ").title()
    normalized = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if normalized != slug:
        raise RuntimeError("Kaggle title must normalize to the exact referenced kernel slug")
    return title


LAUNCH_BODY = r'''bootstrap_checkpoint('imports')
from pathlib import Path
import base64,hashlib,json,math,os,shutil,subprocess,sys,tarfile,time,zlib
CODE=Path('/kaggle/working/sharded_source'); CODE.mkdir(exist_ok=True)
bootstrap_checkpoint('vendoring')
for name,encoded in json.loads(zlib.decompress(base64.b64decode(COMPRESSED_PAYLOAD))).items():
    destination=CODE/name; destination.parent.mkdir(exist_ok=True); destination.write_bytes(base64.b64decode(encoded))
    if hashlib.sha256(destination.read_bytes()).hexdigest()!=SETTINGS['source_sha256'][name]:
        raise RuntimeError('Vendored sharded source SHA mismatch: '+name)
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
    if len(found)!=1:raise RuntimeError(f'Missing/ambiguous private source {identifier}: {found}')
    return found[0]
def unpack(archive,target,expected,allowed_members=None):
    if archive.stat().st_size!=expected['bytes'] or hashlib.sha256(archive.read_bytes()).hexdigest()!=expected['sha256']:
        raise RuntimeError('Private archive size/SHA differs from pin')
    target.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers(); files=[]
        for member in members:
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
                raise RuntimeError('Unsafe private archive member')
            if member.isfile():files.append(member.name)
        if len(files)!=len(set(files)) or (allowed_members is not None and set(files)!=set(allowed_members)):
            raise RuntimeError('Private archive members differ from manifest')
        tar.extractall(target,filter='data')
bootstrap_checkpoint('global_metadata_resolution')
metadata=find_input(SETTINGS['metadata_dataset'],'global_metadata_manifest.json')
manifest_file=metadata/'global_metadata_manifest.json'
if hashlib.sha256(manifest_file.read_bytes()).hexdigest()!=SETTINGS['global_manifest_sha256']:
    raise RuntimeError('Global private manifest differs from prepared pin')
manifest=json.loads(manifest_file.read_text())
DATA=Path('/kaggle/working/global_metadata')
bootstrap_checkpoint('global_metadata_unpack')
unpack(metadata/manifest['archive']['filename'],DATA,manifest['archive'],[row['path'] for row in manifest['members']])
shutil.copyfile(manifest_file,DATA/'global_metadata_manifest.json')
cmd=[sys.executable,str(CODE/'sharded_frozen_extract.py'),'--mode',SETTINGS['mode'],
     '--global-root',str(DATA),'--global-manifest-sha256',SETTINGS['global_manifest_sha256'],
     '--assets-manifest-sha256',SETTINGS['assets_manifest_sha256'],'--config-json',str(CODE/'config.json'),
     '--output-dir','/kaggle/working/run','--max-seconds',str(SETTINGS['max_seconds']),
     '--planned-extraction-jobs',str(SETTINGS['planned_extraction_jobs'])]
if SETTINGS['mode'] in ('pilot','extract'):
    bootstrap_checkpoint('encoder_assets_resolution')
    assets=find_input(SETTINGS['assets_dataset'],'asset_manifest.json')
    asset_file=assets/'asset_manifest.json'
    if hashlib.sha256(asset_file.read_bytes()).hexdigest()!=SETTINGS['assets_manifest_sha256']:
        raise RuntimeError('Encoder asset manifest differs from prepared pin')
    asset_manifest=json.loads(asset_file.read_text()); dinov3=asset_manifest['dinov3_source']
    bootstrap_checkpoint('encoder_source_unpack')
    source_archive=assets/dinov3['filename']
    unpack(source_archive,CODE,{'bytes':source_archive.stat().st_size,'sha256':dinov3['sha256']})
    bootstrap_checkpoint('owned_cache_resolution')
    roots=[find_input(identifier,'cache_summary.json','notebooks') for identifier in SETTINGS['cache_kernel_ids']]
    cmd+=['--shards',*[str(shard) for shard in SETTINGS['shards']],'--cache-roots',*[str(root) for root in roots],
          '--dinov3-repo',str(CODE/dinov3['archive_root']),'--checkpoint',str(assets/asset_manifest['checkpoint']['filename']),
          '--assets-manifest',str(asset_file)]
    if SETTINGS['resume_kernel']:
        prior=find_input(SETTINGS['resume_kernel'],'run/partial_manifest.json','notebooks')
        cmd+=['--resume-partial',str(prior/'run')]
else:
    bootstrap_checkpoint('partial_features_resolution')
    roots=[find_input(identifier,'run/partial_manifest.json','notebooks') for identifier in SETTINGS['partial_kernels']]
    cmd+=['--partial-roots',*[str(root/'run') for root in roots]]
bootstrap_checkpoint('gpu_selection')
import torch
if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU; refuse CPU fallback')
free=[torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
env=dict(os.environ); env['PYTHONPATH']=str(CODE)+os.pathsep+env.get('PYTHONPATH','')
env['OMP_NUM_THREADS']='1'; env['OPENBLAS_NUM_THREADS']='1'
env['CUDA_VISIBLE_DEVICES']=str(max(range(len(free)),key=lambda index:free[index]))
env['RSNA_SHARDED_BOOTSTRAP_START']=str(BOOTSTRAP_MONOTONIC)
print(json.dumps({'mode':SETTINGS['mode'],'shards':SETTINGS['shards'],'max_seconds':SETTINGS['max_seconds'],
                  'launch':cmd,'gpu_free_gb':[value/1e9 for value in free]}),flush=True)
bootstrap_checkpoint('launch')
remaining=SETTINGS['max_seconds']-(time.monotonic()-BOOTSTRAP_MONOTONIC)
if remaining<=0:raise RuntimeError('Private launcher Python-stage budget exhausted before child launch')
subprocess.run(cmd,env=env,check=True,timeout=remaining)
bootstrap_checkpoint('report')
name={'pilot':'pilot_report.json','extract':'extraction_report.json','merge-train':'training_report.json'}[SETTINGS['mode']]
report=json.loads((Path('/kaggle/working/run')/name).read_text())
if SETTINGS['mode']=='extract' and not report['extraction_complete_for_shards']:raise RuntimeError('Shard feature extraction incomplete')
if SETTINGS['mode']=='merge-train' and not report['head_training_complete']:raise RuntimeError('Merged head training incomplete')
print(json.dumps(report,indent=2),flush=True)
bootstrap_checkpoint('complete')
'''


def create_notebook(directory, settings, metadata):
    modules = sorted((ROOT / "src/orthofoundation").glob("*.py"))
    files = {"orthofoundation/" + path.name: path for path in modules}
    files["sharded_frozen_extract.py"] = ROOT / "scripts/sharded_frozen_extract.py"
    payload = {name: base64.b64encode(path.read_bytes()).decode() for name, path in files.items()}
    settings["source_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    compressed = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    code = instrument_bootstrap(LAUNCH_BODY, compressed, settings)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        "# RSNA Knee explicit sharded fallback: " + settings["mode"] + "\n\n"
        "All-seven metadata and global longest-series selection are validated before local archive filtering. "
        "A shard produces explicitly partial frozen features; it is never called full training. "
        "Merge requires 4,407 studies, 21,334 selected slot-series and 85,336 slices before fresh heads train. "
        "Same 224px/.92 FOV/K4/author_no_rope extraction, fixed 12 epochs and Gold58 exclusion. "
        "Private medical tables are read from the small attached metadata asset, never embedded here."),
        nbformat.v4.new_code_cell("import time\nBOOTSTRAP_MONOTONIC=time.monotonic()\n" + bootstrap_start_code()), nbformat.v4.new_code_cell(code)],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12.13"}})
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.ipynb").write_text(nbformat.writes(notebook))
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "stage_config.json").write_text(json.dumps(settings, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "extract", "merge-train", "all"), required=True)
    parser.add_argument("--global-root", type=Path, required=True, help="Private locally extracted small metadata for preparation")
    parser.add_argument("--assets-manifest", type=Path, required=True)
    parser.add_argument("--docker-metadata", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orthofoundation_frozen.yaml")
    parser.add_argument("--metadata-dataset", default="alanchoo/rsna-knee-ortho-global-metadata-v1")
    parser.add_argument("--assets-dataset", default="alanchoo/orthofoundation-l-assets-v1")
    parser.add_argument("--owner", default="alanchoo")
    parser.add_argument("--shards", type=int, nargs="+", default=[0])
    parser.add_argument("--partial-kernels", nargs="+")
    parser.add_argument("--resume-kernel")
    parser.add_argument("--planned-extraction-jobs", type=int, default=7)
    parser.add_argument("--max-seconds", type=int, help="May lower, never exceed the proportional preparation cap")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/kaggle/sharded")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    global_sha = RUNNER.file_sha(args.global_root / "global_metadata_manifest.json")
    plan = RUNNER.validate_global(args.global_root, global_sha)
    assets_sha = RUNNER.file_sha(args.assets_manifest)
    RUNNER.make_identity(plan, config, assets_sha)
    docker = json.loads(args.docker_metadata.read_text())["docker_image"]
    stages = [("pilot", [0]), *[("extract", [shard]) for shard in range(7)], ("merge-train", [])] if args.stage == "all" else [(args.stage, args.shards if args.stage != "merge-train" else [])]
    default_partials = [args.owner + f"/rsna-knee-ortho-shard-{shard}" for shard in range(7)]
    for mode, shards in stages:
        if len(set(shards)) != len(shards) or not set(shards).issubset(range(7)):
            raise RuntimeError("Shard group must be distinct and within 0..6")
        group = "-".join(map(str, sorted(shards)))
        slug = "rsna-knee-ortho-sharded-pilot" if mode == "pilot" else ("rsna-knee-ortho-shard-" + group if mode == "extract" else "rsna-knee-ortho-merge-heads")
        image_count = sum(int(RUNNER.owned_mask(plan, uid, set(shards)).sum()) for uid in plan["ids"]) * 4 if shards else 0
        cap = RUNNER.allocated_extraction_cap(image_count) if shards else RUNNER.HEAD_JOB_RESERVE_SECONDS
        budget = RUNNER.PILOT_BUDGET_SECONDS if mode == "pilot" else cap
        if args.max_seconds is not None:
            if args.max_seconds <= 0 or args.max_seconds > budget:
                raise RuntimeError("Requested runtime exceeds proportional hard cap")
            budget = args.max_seconds
        resume = args.resume_kernel if mode == "extract" else None
        if args.stage == "all" and mode == "extract" and shards == [0]:
            resume = args.owner + "/rsna-knee-ortho-sharded-pilot"
        if resume == args.owner + "/" + slug:
            raise RuntimeError("Resume kernel cannot be the output itself")
        cache_ids = [plan["kernel_by_shard"][shard] for shard in sorted(shards)]
        partials = args.partial_kernels or default_partials if mode == "merge-train" else []
        settings = {"mode": mode, "shards": sorted(shards), "config": config, "global_manifest_sha256": global_sha,
                    "assets_manifest_sha256": assets_sha, "metadata_dataset": args.metadata_dataset,
                    "assets_dataset": args.assets_dataset, "cache_kernel_ids": cache_ids, "partial_kernels": partials,
                    "resume_kernel": resume, "max_seconds": budget, "expected_images_for_shards": image_count,
                    "planned_extraction_jobs": args.planned_extraction_jobs,
                    "total_frozen_budget_seconds": config["max_extract_seconds"], "head_fit_budget_seconds": 1200,
                    "head_job_reserve_seconds": RUNNER.HEAD_JOB_RESERVE_SECONDS}
        datasets = [args.metadata_dataset] + ([args.assets_dataset] if mode != "merge-train" else [])
        metadata = {"id": args.owner + "/" + slug, "title": title_for_slug(slug),
                    "code_file": "train.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
                    "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
                    "competition_sources": [], "dataset_sources": datasets,
                    "kernel_sources": partials if mode == "merge-train" else cache_ids + ([resume] if resume else []),
                    "docker_image": docker, "docker_image_pinning_type": "original"}
        directory = args.output_root / slug
        create_notebook(directory, settings, metadata)
        print(json.dumps({"mode": mode, "shards": shards, "directory": str(directory), "max_seconds": budget,
                          "input_kernel_count": len(metadata["kernel_sources"]), "expected_images": image_count}), flush=True)


if __name__ == "__main__":
    main()

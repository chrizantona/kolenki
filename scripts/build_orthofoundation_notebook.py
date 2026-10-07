#!/usr/bin/env python3
"""Generate a private, offline Kaggle notebook; never contact Kaggle or embed labels.

Code is vendored with SHA pins. Weights, report-derived labels, and verified cache
receipts come from explicitly attached private assets. Generated notebooks live
under artifacts/ (gitignored), rather than publishing competition tables to Git.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]


def bootstrap_start_code():
    """Stdlib-only first cell: prove Python started before ML/input dependencies."""
    return r'''print('{"event":"bootstrap_start","diagnostic_schema":1}',flush=True)
import json,os,time,traceback
BOOTSTRAP_DIR='/kaggle/working'
BOOTSTRAP_STAGE='notebook_start'
os.makedirs(BOOTSTRAP_DIR,exist_ok=True)
def bootstrap_checkpoint(stage):
    global BOOTSTRAP_STAGE
    BOOTSTRAP_STAGE=stage
    event={'diagnostic_schema':1,'event':'bootstrap_checkpoint','stage':stage,'unix_seconds':time.time()}
    with open(os.path.join(BOOTSTRAP_DIR,'bootstrap_progress.json'),'w') as stream:
        json.dump(event,stream,indent=2)
    with open(os.path.join(BOOTSTRAP_DIR,'bootstrap_events.jsonl'),'a') as stream:
        stream.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)
def bootstrap_save_failure(error):
    event={'diagnostic_schema':1,'event':'bootstrap_failure','stage':BOOTSTRAP_STAGE,
           'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
           'unix_seconds':time.time()}
    try:
        with open(os.path.join(BOOTSTRAP_DIR,'bootstrap_failure.json'),'w') as stream:
            json.dump(event,stream,indent=2)
    except BaseException as save_error:
        print(json.dumps({'event':'bootstrap_failure_save_error','error_type':type(save_error).__name__}),flush=True)
    print(json.dumps(event),flush=True)
bootstrap_checkpoint('notebook_start')
with open(os.path.join(BOOTSTRAP_DIR,'bootstrap_start.json'),'w') as stream:
    json.dump({'diagnostic_schema':1,'event':'bootstrap_start','unix_seconds':time.time()},stream,indent=2)
'''


def instrument_bootstrap(body, compressed, settings):
    """Keep payload/settings as top-level literals; preserve failures and reraises."""
    code = "COMPRESSED_PAYLOAD=" + repr(compressed) + "\nSETTINGS=" + repr(settings) + "\n"
    code += "try:\n" + textwrap.indent(body, "    ")
    code += "except BaseException as bootstrap_error:\n    bootstrap_save_failure(bootstrap_error)\n    raise\n"
    ast.parse(code)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orthofoundation_frozen.yaml")
    parser.add_argument("--cache-plan", type=Path, required=True)
    parser.add_argument("--docker-metadata", type=Path, required=True)
    parser.add_argument("--preprocess-sha256", required=True)
    parser.add_argument("--assets-dataset", default="alanchoo/orthofoundation-l-assets-v1")
    parser.add_argument("--training-inputs-dataset", help="Default: labels/receipts colocated with encoder assets")
    parser.add_argument("--owner", default="alanchoo")
    parser.add_argument("--slug")
    parser.add_argument("--title")
    parser.add_argument("--resume-kernel", help="Prior exact-config feature output, optionally the pilot")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["extraction_variant"] not in ("author_no_rope", "canonical_dinov3_rope"):
        raise RuntimeError("Extraction variant must be explicit")
    cache_ids = [row["kernel_id"] for row in json.loads(args.cache_plan.read_text())]
    if len(cache_ids) != 7 or len(set(cache_ids)) != 7:
        raise RuntimeError("Exactly seven distinct completed cache shard sources are required")
    docker = json.loads(args.docker_metadata.read_text())["docker_image"]
    modules = sorted((ROOT / "src/orthofoundation").glob("*.py"))
    payload = {"orthofoundation/" + path.name: base64.b64encode(path.read_bytes()).decode() for path in modules}
    hashes = {"orthofoundation/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in modules}
    compressed = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    settings = {"stage": args.stage, "config": config, "source_sha256": hashes, "cache_kernel_ids": cache_ids,
                "assets_dataset": args.assets_dataset, "training_inputs_dataset": args.training_inputs_dataset or args.assets_dataset,
                "preprocess_sha256": args.preprocess_sha256, "resume_kernel": args.resume_kernel}
    code = r'''bootstrap_checkpoint('imports')
from pathlib import Path
import base64,hashlib,json,os,subprocess,sys,tarfile,zlib
bootstrap_checkpoint('vendoring')
CODE=Path('/kaggle/working/source'); CODE.mkdir(exist_ok=True)
for name,encoded in json.loads(zlib.decompress(base64.b64decode(COMPRESSED_PAYLOAD))).items():
    target=CODE/name; target.parent.mkdir(exist_ok=True); target.write_bytes(base64.b64decode(encoded))
    if hashlib.sha256(target.read_bytes()).hexdigest()!=SETTINGS['source_sha256'][name]:
        raise RuntimeError('Vendored source SHA mismatch: '+name)
(CODE/'config.json').write_text(json.dumps(SETTINGS['config'],indent=2))
bootstrap_checkpoint('input_resolution')
INPUT=Path('/kaggle/input')
def find_named_input(identifier,marker,kind='datasets'):
    owner,slug=identifier.split('/')
    candidates=[INPUT/kind/owner/slug,INPUT/slug,INPUT/'kernels'/owner/slug,INPUT/owner/slug]
    matches=[path for path in candidates if (path/marker).is_file()]
    if not matches:
        for pattern in [f'*/{slug}',f'*/*/{slug}',f'*/*/*/{slug}']:
            matches.extend(path for path in INPUT.glob(pattern) if (path/marker).is_file())
    matches=list(dict.fromkeys(matches))
    if len(matches)!=1:raise RuntimeError(f'Expected one {identifier} source with {marker}: {matches}')
    return matches[0]
competition=[path for path in [INPUT/'competitions/rsna-knee-abnormality-detection',INPUT/'rsna-knee-abnormality-detection']
             if (path/'train.csv').is_file() and (path/'train_series.csv').is_file()]
if len(competition)!=1:raise RuntimeError('Missing/ambiguous competition source')
assets=find_named_input(SETTINGS['assets_dataset'],'asset_manifest.json')
training=find_named_input(SETTINGS['training_inputs_dataset'],'labels_v0.csv')
manifest=json.loads((assets/'asset_manifest.json').read_text())
expected=SETTINGS['config']
if manifest['checkpoint']['sha256']!=expected['checkpoint_sha256'] or manifest['checkpoint']['bytes']!=expected['checkpoint_bytes']:
    raise RuntimeError('Asset manifest checkpoint differs from frozen experiment')
bootstrap_checkpoint('vendor_unpack')
archive=assets/manifest['dinov3_source']['filename']
if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['dinov3_source']['sha256']:
    raise RuntimeError('DINOv3 source archive SHA mismatch')
with tarfile.open(archive,'r:gz') as tar:
    # Reject links and traversal even before Python's extraction filter.
    for member in tar.getmembers():
        parts=Path(member.name).parts
        if member.issym() or member.islnk() or Path(member.name).is_absolute() or '..' in parts:
            raise RuntimeError('Unsafe DINOv3 source archive member')
    tar.extractall(CODE,filter='data')
bootstrap_checkpoint('cache_resolution')
roots=[find_named_input(identifier,'cache_summary.json','notebooks') for identifier in SETTINGS['cache_kernel_ids']]
bootstrap_checkpoint('gpu_selection')
env=dict(os.environ); env['PYTHONPATH']=str(CODE)+os.pathsep+env.get('PYTHONPATH','')
env['OMP_NUM_THREADS']='1'; env['OPENBLAS_NUM_THREADS']='1'
import torch
if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU; refuse CPU fallback')
free=[torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
env['CUDA_VISIBLE_DEVICES']=str(max(range(len(free)),key=lambda index:free[index]))
cmd=[sys.executable,'-m','orthofoundation.run','--stage',SETTINGS['stage'],'--config',str(CODE/'config.json'),
     '--data-dir',str(competition[0]),'--labels',str(training/'labels_v0.csv'),
     '--cache-roots',*[str(path) for path in roots],'--cache-fingerprints',str(training/'cache_input_fingerprints.json'),
     '--preprocess-sha256',SETTINGS['preprocess_sha256'],'--dinov3-repo',str(CODE/manifest['dinov3_source']['archive_root']),
     '--checkpoint',str(assets/manifest['checkpoint']['filename']),'--assets-manifest',str(assets/'asset_manifest.json'),
     '--output-dir','/kaggle/working/run']
if SETTINGS['resume_kernel']:
    previous=find_named_input(SETTINGS['resume_kernel'],'run/provenance.json','notebooks')
    cmd+=['--resume-features',str(previous/'run/features')]
print(json.dumps({'launch':cmd,'gpu_free_gb':[value/1e9 for value in free]}),flush=True)
bootstrap_checkpoint('launch')
subprocess.run(cmd,env=env,check=True)
bootstrap_checkpoint('report')
report_file=Path('/kaggle/working/run')/('pilot_report.json' if SETTINGS['stage']=='pilot' else 'training_report.json')
report=json.loads(report_file.read_text())
if SETTINGS['stage']=='full' and not report['training_complete']:raise RuntimeError('Full training did not finish')
print(json.dumps(report,indent=2),flush=True)
bootstrap_checkpoint('complete')
'''
    code = instrument_bootstrap(code, compressed, settings)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        "# RSNA Knee OrthoFoundation " + args.stage + "\n\n"
        "New frozen DINOv3-L knee encoder, normalized CLS per MRI slice, freshly initialized 12-target study head. "
        "Primary forward follows the author's no-RoPE wrapper; this is explicitly recorded. "
        "The pilot verifies strict weights, finite features, one finite head update, throughput and memory. "
        "Full mode stops if measured extraction exceeds its budget. Gold studies are excluded from gradients; "
        "fixed final epochs are used without Gold checkpoint selection. No leaderboard gain is assumed."),
        nbformat.v4.new_code_cell(bootstrap_start_code()),
        nbformat.v4.new_code_cell(code)], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                   "language_info": {"name": "python", "version": "3.12.13"}})
    nbformat.validate(notebook)
    slug = args.slug or "rsna-knee-orthofoundation-" + args.stage
    if args.resume_kernel == args.owner + "/" + slug:
        raise RuntimeError("Resume source cannot be the output kernel itself")
    directory = args.output_dir or ROOT / "artifacts/kaggle" / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train.ipynb").write_text(nbformat.writes(notebook))
    metadata = {"id": args.owner + "/" + slug, "title": args.title or "RSNA Knee OrthoFoundation " + args.stage.title(),
                "code_file": "train.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
                "competition_sources": ["rsna-knee-abnormality-detection"],
                "dataset_sources": list(dict.fromkeys([settings["assets_dataset"], settings["training_inputs_dataset"]])),
                "kernel_sources": cache_ids + ([args.resume_kernel] if args.resume_kernel else []),
                "docker_image": docker, "docker_image_pinning_type": "original"}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "stage_config.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps({"stage": args.stage, "directory": str(directory), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()

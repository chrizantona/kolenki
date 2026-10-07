#!/usr/bin/env python3
"""Build a private offline T4 follow-up using the completed full feature output.

This only generates files in gitignored artifacts/. It does not push a notebook,
request GPU time, expose labels, or copy features/weights into the repository.
"""
import argparse
import ast
import base64
import hashlib
import json
import zlib
from pathlib import Path

import nbformat
import yaml

from compare_frozen_heads import controlled_config

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = ("__init__.py", "cache.py", "data.py", "encoder.py", "heads.py", "run.py", "train.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orthofoundation_meanmax.yaml")
    parser.add_argument("--baseline-config", type=Path, default=ROOT / "configs/orthofoundation_frozen.yaml")
    parser.add_argument("--docker-metadata", type=Path, required=True)
    parser.add_argument("--previous-kernel", default="alanchoo/rsna-knee-orthofoundation-full")
    parser.add_argument("--assets-dataset", default="alanchoo/orthofoundation-l-assets-v1")
    parser.add_argument("--owner", default="alanchoo")
    parser.add_argument("--slug", default="rsna-knee-orthofoundation-meanmax")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = controlled_config(yaml.safe_load(args.baseline_config.read_text()), yaml.safe_load(args.config.read_text()))
    if args.previous_kernel == args.owner + "/" + args.slug:
        raise RuntimeError("Output kernel cannot be its own input")
    docker = json.loads(args.docker_metadata.read_text())["docker_image"]
    paths = {"orthofoundation/" + name: ROOT / "src/orthofoundation" / name for name in CORE_MODULES}
    paths["compare_frozen_heads.py"] = ROOT / "scripts/compare_frozen_heads.py"
    payload = {name: base64.b64encode(path.read_bytes()).decode() for name, path in paths.items()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    compressed = base64.b64encode(zlib.compress(json.dumps(payload).encode(), 9)).decode()
    settings = {"config": config, "source_sha256": hashes, "previous_kernel": args.previous_kernel,
                "assets_dataset": args.assets_dataset, "encoder_forward_calls": 0, "head_budget_seconds": 1200}
    code = r'''from pathlib import Path
import base64,hashlib,json,os,subprocess,sys,zlib
CODE=Path('/kaggle/working/source'); CODE.mkdir(exist_ok=True)
for name,encoded in json.loads(zlib.decompress(base64.b64decode(COMPRESSED_PAYLOAD))).items():
    target=CODE/name; target.parent.mkdir(exist_ok=True); target.write_bytes(base64.b64decode(encoded))
    if hashlib.sha256(target.read_bytes()).hexdigest()!=SETTINGS['source_sha256'][name]:
        raise RuntimeError('Vendored source SHA mismatch: '+name)
(CODE/'config.json').write_text(json.dumps(SETTINGS['config'],indent=2))
INPUT=Path('/kaggle/input')
def find_named_input(identifier,marker,kind):
    owner,slug=identifier.split('/')
    candidates=[INPUT/kind/owner/slug,INPUT/slug,INPUT/'kernels'/owner/slug,INPUT/owner/slug]
    matches=[path for path in candidates if (path/marker).is_file()]
    if not matches:
        for pattern in [f'*/{slug}',f'*/*/{slug}',f'*/*/*/{slug}']:
            matches.extend(path for path in INPUT.glob(pattern) if (path/marker).is_file())
    matches=list(dict.fromkeys(matches))
    if len(matches)!=1:raise RuntimeError(f'Expected one {identifier} input with {marker}: {matches}')
    return matches[0]
competition=[path for path in [INPUT/'competitions/rsna-knee-abnormality-detection',INPUT/'rsna-knee-abnormality-detection']
             if (path/'train.csv').is_file() and (path/'train_series.csv').is_file()]
if len(competition)!=1:raise RuntimeError('Missing/ambiguous competition input')
previous=find_named_input(SETTINGS['previous_kernel'],'run/training_report.json','notebooks')/'run'
assets=find_named_input(SETTINGS['assets_dataset'],'labels_v0.csv','datasets')
env=dict(os.environ); env['PYTHONPATH']=str(CODE)+os.pathsep+env.get('PYTHONPATH','')
env['OMP_NUM_THREADS']='1'; env['OPENBLAS_NUM_THREADS']='1'
import torch
if not torch.cuda.is_available():raise RuntimeError('No CUDA GPU; refuse CPU fallback')
free=[torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
env['CUDA_VISIBLE_DEVICES']=str(max(range(len(free)),key=lambda index:free[index]))
cmd=[sys.executable,str(CODE/'compare_frozen_heads.py'),'--previous-run-dir',str(previous),
     '--labels',str(assets/'labels_v0.csv'),'--data-dir',str(competition[0]),'--config',str(CODE/'config.json'),
     '--assets-manifest',str(assets/'asset_manifest.json'),'--output-dir','/kaggle/working/run','--device','cuda']
print(json.dumps({'launch':cmd,'encoder_forward_calls':0,'head_budget_seconds':1200,
                  'gpu_free_gb':[value/1e9 for value in free]}),flush=True)
subprocess.run(cmd,env=env,check=True)
report=json.loads(Path('/kaggle/working/run/training_report.json').read_text())
if not report['training_complete'] or report['encoder_forward_calls']!=0:
    raise RuntimeError('Head-only fixed training did not complete')
print(json.dumps({'training_complete':True,'baseline_attention':report['baseline_attention'],
                  'meanmax_weak_holdout_metrics':report['weak_holdout_metrics'],
                  'meanmax_gold_metrics':report['gold_metrics'],
                  'macro_auc_delta_meanmax_minus_attention':report['macro_auc_delta_meanmax_minus_attention']},indent=2),flush=True)
'''
    code = "COMPRESSED_PAYLOAD=" + repr(compressed) + "\nSETTINGS=" + repr(settings) + "\n" + code
    ast.parse(code)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        "# RSNA Knee: frozen OrthoFoundation attention versus mean+max\n\n"
        "Head-only controlled comparison on the completed EXP-OF-001 feature bank. "
        "The original feature identity, labels, split, seed 42, and fixed 12 epochs are retained. "
        "All 4407 feature files and the prior checkpoint receipts are verified before training. "
        "Fresh weak-holdout and production mean+max heads are trained; all 58 Gold studies stay outside gradients. "
        "No MRI decoding, encoder forward, external download, or Gold checkpoint selection. "
        "Weak-holdout AUC measures teacher agreement and Gold AUC is diagnostic. Outputs are private."),
        nbformat.v4.new_code_cell(code)], metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.13"}})
    nbformat.validate(notebook)
    directory = args.output_dir or ROOT / "artifacts/kaggle" / args.slug
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"id": args.owner + "/" + args.slug, "title": "RSNA Knee OrthoFoundation MeanMax Comparison",
                "code_file": "compare.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
                "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": [args.assets_dataset],
                "kernel_sources": [args.previous_kernel], "docker_image": docker, "docker_image_pinning_type": "original"}
    (directory / "compare.ipynb").write_text(nbformat.writes(notebook))
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (directory / "stage_config.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps({"directory": str(directory), "metadata": metadata, "source_sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare isolated Ortho test inference; never blend or submit predictions."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orthofoundation_inference.contracts import load_production_head, require, require_pin, verify_frozen_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    gate = parser.add_mutually_exclusive_group(required=True)
    gate.add_argument('--production-head-sha256', help='Externally sealed ROOT production head.pt SHA, never derived from current file')
    gate.add_argument('--prepare-placeholder', action='store_true', help='Explicit non-executable placeholder; refuses before weights or DICOM reads')
    parser.add_argument('--production-head', type=Path)
    parser.add_argument('--ROOT-training-seal', type=Path, dest='seal')
    parser.add_argument('--ROOT-training-seal-sha256', dest='seal_sha')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--split', default='test')
    parser.add_argument('--assets-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-seconds', type=int, default=3600)
    args = parser.parse_args()
    require(not args.prepare_placeholder, 'Placeholder only: verified future production head/ROOT seal required before any weights or DICOM decoding')
    require_pin(args.production_head_sha256, 'production head'); require_pin(args.seal_sha, 'training seal')
    require(all(value is not None for value in (args.production_head, args.seal, args.data_dir, args.assets_dir, args.output_dir)),
            'Explicit production checkpoint, ROOT seal, test metadata/DICOM directory, assets and output are required')
    config, sources, _ = verify_frozen_sources()
    head, admission = load_production_head(args.production_head, args.production_head_sha256, args.seal, args.seal_sha, config)
    # Heavy imports happen only after every externally pinned admission check.
    from orthofoundation_inference.pipeline import run_inference
    report = run_inference(args.data_dir, args.split, args.output_dir, head, admission, config, sources, args.assets_dir, args.max_seconds)
    import json
    print(json.dumps({'inference_complete': True, 'n_test_studies': report['n_test_studies'],
        'encoded_images': report['encoded_images'], 'seconds': report['seconds'],
        'output_files': ['_ortho.csv', '_ortho_provenance.json'], 'automatic_blend_or_submission': False}), flush=True)


if __name__ == '__main__': main()

"""External pins and production-only admission; no encoder or DICOM loading."""
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
TRAINING_IDENTITY = '7f0e16dc1c34e8c0920f567db2ebb68f55c470c09ce77dd7c0aa238cd5ec6c77'
CONFIG_SHA = 'f3c4334f97df527d22dd2b0da3b6220ec466c1333efe7e99a471b955a74ba919'
ASSETS_SHA = '6feeb03c976483c2b91cca667f4fdbefa998d3620a93eb6006de22a13cbdeb4f'
ENCODER_SHA = '385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9'
ENCODER_BYTES = 1213056638
ROOT_SEAL_HELPER_SHA = 'dcb6e90aa90c14e22f901d0aedc05b592ae4d9198014a12d450b2ee7e3675ef4'
SOURCE8 = {
    'orthofoundation/__init__.py': 'f3df43207f5d18497b4f698aa91c746bf81dcd4edb74b81784136acd8aa52cf1',
    'orthofoundation/cache.py': '700767576eb7b6d9dfa22f77bdf2088411ba4bdf6036c80db58bf608c96f7485',
    'orthofoundation/data.py': 'f3b63ab922f0452c065126dc2ef1c87442a0b988dff29d25c201666a8a5310aa',
    'orthofoundation/encoder.py': '86defa5f943bfc161603c7b377bbf08a9d5c2a270d01b1df5c039ab59cff7ecb',
    'orthofoundation/heads.py': 'c814b1bcbf1ac1322b717c1f1e6ba7b9173357e956e9104b2c77d0351a833f1f',
    'orthofoundation/run.py': '740c2e2599516b1876e7d7226bb0f11c51205cbdaec0b6eebf44dc740c659e10',
    'orthofoundation/train.py': '928121a030aabf4fc58594da967ad98ff43a7ce915474d69a909ab09652a7dc1',
    'sharded_frozen_extract.py': '86265241778c40a3822c2a8d11f4f3f746a48b47101def338120dd86937ee4f2',
}
PREPROCESS_SHA = '68601ad84ec3a0f030b5c2d37cc0d657cd3e12b17f7130e7385fba9648d324b6'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_pin(pin, description):
    require(isinstance(pin, str) and re.fullmatch(r'[0-9a-f]{64}', pin) is not None,
            f'Missing/invalid external ROOT {description} SHA256; preparation placeholder cannot run')


def snapshot(path, expected_sha=None, maximum_bytes=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Required ordinary pinned file missing or linked')
    if maximum_bytes is not None:
        require(0 < path.stat().st_size <= maximum_bytes, 'Pinned input exceeds its bounded size')
    data = path.read_bytes(); actual = hashlib.sha256(data).hexdigest()
    if expected_sha is not None:
        require_pin(expected_sha, 'input')
        require(actual == expected_sha, 'Pinned input snapshot SHA differs from external ROOT pin')
    return data, actual


def verify_frozen_sources(root=ROOT):
    root = Path(root); receipts = {}
    for name, pin in SOURCE8.items():
        path = root / ('src/' + name if name.startswith('orthofoundation/')
                       else 'scripts/orthofoundation_http_v2/' + name)
        _, receipts[name] = snapshot(path, pin)
    _, receipts['convnext_reader/preprocess.py'] = snapshot(root / 'src/convnext_reader/preprocess.py', PREPROCESS_SHA)
    data, config_sha = snapshot(root / 'configs/orthofoundation_frozen_http_v2.yaml', CONFIG_SHA)
    import yaml
    return yaml.safe_load(data), receipts, config_sha


def load_production_head(head_path, head_sha, seal_path, seal_sha, config):
    """Bind one small checkpoint snapshot to an independently sealed ROOT proof."""
    require_pin(head_sha, 'production head'); require_pin(seal_sha, 'training seal')
    require(Path(head_path).name == 'head.pt', 'Only fixed-final production head.pt is accepted; no smoke/CV fallback')
    seal_bytes, _ = snapshot(seal_path, seal_sha, maximum_bytes=1024 * 1024)
    seal = json.loads(seal_bytes)
    require(seal.get('schema') == 'ROOT_http_v2_attention_full_bank_seal_v1'
            and seal.get('ROOT_actual_full_merge_PASS') is True
            and seal.get('feature_identity') == TRAINING_IDENTITY
            and seal.get('validation_helper_sha256') == ROOT_SEAL_HELPER_SHA
            and seal.get('source_kernel') == 'alanchoo/rsna-knee-ortho-http-merge-heads-v2'
            and seal.get('kernel_version_origin_verified_separately_by_ROOT') is True
            and type(seal.get('root_confirmed_kernel_version')) is int and seal['root_confirmed_kernel_version'] > 0,
            'A reviewed ROOT seal of the complete fixed-final attention output is required')
    require(seal.get('downloaded_source8_sha256') == SOURCE8 and seal.get('approved_config_yaml_sha256') == CONFIG_SHA
            and seal.get('assets_manifest_sha256') == ASSETS_SHA, 'ROOT seal source/config/encoder provenance differs')
    require(seal.get('counts') == {'studies': 4407, 'weak': 4349, 'Gold': 58, 'CV_gradient': 3479,
            'weak_holdout': 870, 'production_gradient': 4349, 'selected_slots': 21334, 'images_k4': 85336},
            'ROOT seal does not certify complete training coverage and Gold exclusion')
    bank = seal.get('feature_bank_receipt', {})
    require(bank.get('studies') == 4407 and bank.get('images_k4') == 85336
            and bank.get('selected_slot_series') == 21334 and bank.get('sha256') == seal.get('source_bank_sha256'),
            'ROOT full source-bank seal differs')
    require_pin(seal.get('source_bank_sha256'), 'complete training source bank')
    require(seal.get('both_heads') == {'epochs': 12, 'CV_optimizer_updates': 660, 'production_optimizer_updates': 816,
            'Gold_gradient_rows': 0, 'Gold_checkpoint_selection': False}, 'ROOT head schedule or Gold policy differs')
    receipt = seal.get('checkpoint_receipts', {}).get('head.pt', {})
    require(receipt.get('filename') == 'head.pt' and receipt.get('sha256') == head_sha
            and type(receipt.get('bytes')) is int and 0 < receipt['bytes'] <= 16 * 1024 * 1024,
            'External production pin differs from ROOT checkpoint receipt')
    head_bytes, _ = snapshot(head_path, head_sha, maximum_bytes=16 * 1024 * 1024)
    require(len(head_bytes) == receipt['bytes'], 'Production checkpoint bytes differ from ROOT receipt')
    import torch
    from orthofoundation.data import LABELS
    from orthofoundation.heads import StudyHead
    saved = torch.load(BytesIO(head_bytes), map_location='cpu', weights_only=True)
    require(set(saved) == {'model', 'config', 'labels', 'feature_identity', 'training_report'}
            and saved['config'] == config and config['head_kind'] == 'attention'
            and saved['labels'] == LABELS and saved['feature_identity'] == TRAINING_IDENTITY,
            'Production checkpoint config/labels/training identity differs')
    trained = saved['training_report']
    require(trained.get('epochs') == 12 and trained.get('n_gradient_studies') == 4349
            and trained.get('optimizer_updates') == 816 and trained.get('gold_used_for_gradients') == 0
            and trained.get('fixed_final_epoch') is True and trained.get('gold_checkpoint_selection') is False,
            'Checkpoint is not the fixed12epoch all4349weak production head')
    require_pin(trained.get('gradient_ids_sha256'), 'production gradient manifest')
    require(trained.get('head_code_sha256') == SOURCE8['orthofoundation/heads.py']
            and trained.get('training_code_sha256') == SOURCE8['orthofoundation/train.py'], 'Production head training source differs')
    history = trained.get('history', [])
    require(len(history) == 12 and all(row.get('epoch') == j + 1 and row.get('updates') == (j + 1) * 68
            and isinstance(row.get('loss'), (int, float)) and math.isfinite(row['loss']) for j, row in enumerate(history)),
            'Production fixed-final history/update count or finite loss differs')
    require(isinstance(saved['model'], dict) and all(isinstance(value, torch.Tensor) and value.dtype == torch.float32
            and torch.isfinite(value).all().item() for value in saved['model'].values()), 'Invalid/nonfinite or non-FP32 production state')
    head = StudyHead(dim=config['head_dim'], kind='attention', dropout=config['head_dropout'])
    head.load_state_dict(saved['model'], strict=True); head.eval().requires_grad_(False)
    return head, {'production_head_sha256': head_sha, 'ROOT_training_seal_sha256': seal_sha,
                  'parent_training_feature_identity': TRAINING_IDENTITY, 'source_training_bank_sha256': seal['source_bank_sha256'],
                  'fixed_epochs': 12, 'weak_gradient_studies': 4349, 'Gold_gradient_studies': 0, 'Gold_checkpoint_selection': False}

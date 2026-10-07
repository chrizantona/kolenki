"""Load the real DINOv3 architecture offline, with strict checkpoint validation."""
import hashlib
import sys
from pathlib import Path

import torch


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_encoder(repo, checkpoint, expected_sha, expected_bytes, device="cuda"):
    checkpoint, repo = Path(checkpoint), Path(repo)
    if checkpoint.stat().st_size != expected_bytes or file_sha(checkpoint) != expected_sha:
        raise RuntimeError("OrthoFoundation checkpoint size/SHA differs from pinned release")
    if not (repo / "hubconf.py").is_file() or not (repo / "LICENSE.md").is_file():
        raise RuntimeError("Pinned offline DINOv3 source or its license is missing")
    # Passing a custom filename to weights= hits DINOv3's filename-hash parser.
    # Build with pretrained=False; apply the released state explicitly and strictly.
    sys.path.insert(0, str(repo))
    from dinov3.hub.backbones import dinov3_vitl16
    # Import class before meta context so module-scope buffers are initialized normally.
    from dinov3.models.vision_transformer import DinoVisionTransformer  # noqa: F401
    with torch.device("meta"):
        model = dinov3_vitl16(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state or not all(key.startswith("backbone.") for key in state):
        raise RuntimeError("Released checkpoint must contain backbone.* tensors only")
    state = {key.removeprefix("backbone."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False).to(device)
    return model


@torch.inference_mode()
def encode(model, images, variant="author_no_rope"):
    if variant == "author_no_rope":
        # Match the released OrthoFoundation nets/dinov3_vit.py forward_vit:
        # prepare CLS/storage/patch tokens, each block with rope=None, final norm.
        tokens, _ = model.prepare_tokens_with_masks(images)
        for block in model.blocks:
            tokens = block(tokens)
        result = model.norm(tokens)[:, 0]
    elif variant == "canonical_dinov3_rope":
        result = model.forward_features(images)["x_norm_clstoken"]
    else:
        raise ValueError(f"Unknown extraction forward variant: {variant}")
    if result.ndim != 2 or result.shape[-1] != 1024 or not torch.isfinite(result).all():
        raise RuntimeError("Invalid DINOv3-L CLS features")
    return result

"""Heads trained from scratch on frozen per-slice 1024-dimensional features."""
import torch
from torch import nn

from .data import N_SLOTS, LABELS


class StudyHead(nn.Module):
    def __init__(self, input_dim=1024, dim=256, kind="attention", dropout=.2):
        super().__init__()
        if kind not in ("attention", "meanmax"):
            raise ValueError(f"Unknown head: {kind}")
        self.kind = kind
        self.proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.slot = nn.Embedding(N_SLOTS, dim)
        self.position = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        if kind == "attention":
            self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, len(LABELS)))
            self.class_weight = nn.Parameter(torch.randn(len(LABELS), dim) * .02)
            self.class_bias = nn.Parameter(torch.zeros(len(LABELS)))
        else:
            self.classifier = nn.Linear(dim * 2, len(LABELS))

    def forward(self, features, mask, positions):
        if features.ndim != 4 or features.shape[1] != N_SLOTS:
            raise ValueError("Expected features [B,6,K,D]")
        b, s, k, _ = features.shape
        if mask.shape != (b, s) or positions.shape != (b, s, k):
            raise ValueError("Mask or position shape differs from features")
        if (~mask.any(dim=1)).any():
            raise ValueError("Empty studies must never enter the classifier")
        h = self.proj(features)
        h = h + self.slot(torch.arange(s, device=h.device))[None, :, None] + self.position(positions[..., None])
        h = h.reshape(b, s * k, -1)
        valid = mask[:, :, None].expand(b, s, k).reshape(b, s * k)
        if self.kind == "meanmax":
            mean = h.masked_fill(~valid[..., None], 0).sum(dim=1) / valid.sum(dim=1, keepdim=True)
            maximum = h.masked_fill(~valid[..., None], -torch.inf).max(dim=1).values
            return self.classifier(torch.cat((mean, maximum), dim=-1))
        scores = self.attention(h).float().masked_fill(~valid[..., None], -torch.inf)
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("btc,btd->bcd", weights.to(h.dtype), h)
        return (pooled * self.class_weight).sum(dim=-1) + self.class_bias

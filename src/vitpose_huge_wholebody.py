"""Pure-PyTorch ViTPose-Huge wholebody implementation.

The upstream ViTPose-Huge wholebody checkpoint (from ViTAE-Transformer/ViTPose,
also mirrored at JunkyByte/easy_ViTPose) has the architecture:

    backbone = plain ViT-H
        patch_embed: Conv2d(3, 1280, kernel=16, stride=16)
        pos_embed:  Parameter (1, 193, 1280)   # 16*12 patches + 1 cls token
        blocks[32]:
            norm1 -> attn.qkv(1280->3840) -> attn.proj(1280->1280) ->
            norm2 -> mlp.fc1(1280->5120) -> mlp.fc2(5120->1280)
        last_norm: LayerNorm(1280)
    keypoint_head = ClassicHead (2 x deconv upsample + 1x1 conv)
        deconv_layers.0:  ConvTranspose2d(1280, 256, 4, 2, 1)
        deconv_layers.1:  BatchNorm2d(256)
        deconv_layers.3:  ConvTranspose2d(256, 256, 4, 2, 1)
        deconv_layers.4:  BatchNorm2d(256)
        final_layer:      Conv2d(256, 133, 1)

Total ~637 M params.  Input 256x192 (HxW)  ->  feature map 16x12  ->
two 2x upsamples  ->  64x48 heatmap, 133 channels.

We reimplement just enough to load the weights and run forward, with no
third-party Python imports beyond torch/numpy.  The state_dict key
namespace already matches our module names exactly, so loading is a
direct `load_state_dict(..., strict=True)` after `state_dict =
ckpt["state_dict"]`.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Plain ViT-H backbone (matches `backbone.*` keys in the checkpoint)
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _ViTBackbone(nn.Module):
    """Plain ViT backbone matching `backbone.*` keys of vitpose-h-wholebody."""

    def __init__(
        self,
        img_size: tuple[int, int] = (256, 192),
        patch_size: int = 16,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        H, W = img_size
        self.grid_h = H // patch_size
        self.grid_w = W // patch_size
        num_patches = self.grid_h * self.grid_w
        # +1 for the cls token (kept in pos_embed even though we never read it).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size,
                                          stride=patch_size)
        self.blocks = nn.ModuleList([
            _Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.last_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, C, h, w) feature map ready for the deconv head."""
        B = x.shape[0]
        x = self.patch_embed.proj(x)                  # (B, C, h, w)
        h, w = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)              # (B, h*w, C)
        # Skip cls token in pos_embed (index 0), use the patch portion.
        x = x + self.pos_embed[:, 1:1 + h * w]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        x = x.transpose(1, 2).reshape(B, -1, h, w)
        return x


# ---------------------------------------------------------------------------
# Classic keypoint head (matches `keypoint_head.*` keys)
# ---------------------------------------------------------------------------

class _ClassicHead(nn.Module):
    """Two deconv upsamples + 1x1 conv to N keypoint heatmaps."""

    def __init__(self, in_channels: int = 1280, num_keypoints: int = 133):
        super().__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.deconv_layers(x))


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ViTPoseHugeWholeBody(nn.Module):
    """ViTPose-Huge wholebody (137 kpt) heatmap predictor."""

    def __init__(self, num_keypoints: int = 133,
                 img_size: tuple[int, int] = (256, 192)):
        super().__init__()
        self.img_size = img_size
        self.backbone = _ViTBackbone(img_size=img_size)
        self.keypoint_head = _ClassicHead(in_channels=1280,
                                           num_keypoints=num_keypoints)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.keypoint_head(self.backbone(x))

    # ------------------------------------------------------------------ #
    # Checkpoint loading
    # ------------------------------------------------------------------ #
    def load_upstream_pth(self, path: str | Path) -> None:
        """Load weights from the upstream `vitpose-h-wholebody.pth`."""
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        sd = ckpt.get("state_dict", ckpt)
        # The upstream key namespace already matches our module names, so a
        # strict load works. If a mismatch appears we report it explicitly.
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing or unexpected:
            # Trim batch-tracking and other ignorable mismatches.
            missing = [m for m in missing if "num_batches_tracked" not in m]
            unexpected = [u for u in unexpected if "num_batches_tracked" not in u]
            if missing or unexpected:
                raise RuntimeError(
                    f"state_dict mismatch — missing={missing[:4]}... "
                    f"unexpected={unexpected[:4]}..."
                )


# ---------------------------------------------------------------------------
# Heatmap → keypoints (with DARK-style sub-pixel refinement)
# ---------------------------------------------------------------------------

def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    output_size_xy: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax + 2-pixel Taylor-expansion sub-pixel refinement.

    Args:
        heatmaps: (N, K, h, w) tensor.
        output_size_xy: (W, H) of the input image that the heatmap covers.
            The 64x48 heatmap is rescaled back to (W, H).

    Returns:
        kpts_xy: (N, K, 2) keypoint coords in the input-image frame.
        scores:  (N, K) per-keypoint confidence (peak heatmap value).
    """
    N, K, H, W_hm = heatmaps.shape
    flat = heatmaps.view(N, K, -1)
    scores, idx = flat.max(dim=2)
    xs = (idx % W_hm).float()
    ys = (idx // W_hm).float()

    # Sub-pixel refinement: shift by 0.25 * sign(gradient) (classic
    # DARK-style trick; cheap version that doesn't require a Gaussian
    # blur on CPU).
    for n in range(N):
        for k in range(K):
            x = int(xs[n, k].item())
            y = int(ys[n, k].item())
            if 1 <= x < W_hm - 1 and 1 <= y < H - 1:
                dx = heatmaps[n, k, y, x + 1] - heatmaps[n, k, y, x - 1]
                dy = heatmaps[n, k, y + 1, x] - heatmaps[n, k, y - 1, x]
                xs[n, k] += 0.25 * torch.sign(dx)
                ys[n, k] += 0.25 * torch.sign(dy)

    out_w, out_h = output_size_xy
    xs = xs * (out_w / W_hm)
    ys = ys * (out_h / H)
    kpts = torch.stack([xs, ys], dim=-1)
    return kpts, scores

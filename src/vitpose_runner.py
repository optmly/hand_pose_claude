"""Per-video ViTPose-Huge wholebody pass + per-frame hand-keypoint lookup.

The wholebody output has 133 keypoints; indices 91..111 are the LEFT hand
(21 kpts) and 112..132 are the RIGHT hand (21 kpts). We feed the full
frame at stretched 192x256 (no bbox crop) and decode each 48x64 heatmap
with UDP / DARK sub-pixel refinement (ported from easy_ViTPose).

Used by `pose_video_v2.py` as a backup after MP VIDEO and MP IMAGE rerun
both fail to produce a candidate that passes the mask-hull gates --
typically the gloved-hand clips that MP cannot detect at all.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from vitpose_huge_wholebody import ViTPoseHugeWholeBody  # type: ignore
except ImportError:  # when imported as a package, retry with the absolute name
    from src.vitpose_huge_wholebody import ViTPoseHugeWholeBody  # noqa: E402

DEFAULT_CKPT = Path(
    "~/.cache/huggingface/hub/models--JunkyByte--easy_ViTPose/"
    "snapshots/e83805274e89428969355ec4afffcbc413e79188/"
    "torch/wholebody/vitpose-h-wholebody.pth"
).expanduser()

LEFT_HAND_IDX = list(range(91, 112))   # 21 left-hand kpts
RIGHT_HAND_IDX = list(range(112, 133)) # 21 right-hand kpts

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────── UDP / DARK heatmap decode ────────────────────────
# Ported from easy_ViTPose / mmpose: 11x11 Gaussian blur + log + Newton step.

def _get_max_preds(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """argmax per-keypoint. (N, K, H, W) -> preds (N, K, 2), maxvals (N, K, 1)."""
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = np.argmax(flat, axis=2)
    maxvals = np.amax(flat, axis=2)[..., None]
    preds = np.zeros((N, K, 2), dtype=np.float32)
    preds[..., 0] = idx % W
    preds[..., 1] = idx // W
    preds = np.where(maxvals > 0.0, preds, -1.0)
    return preds, maxvals


def _post_dark_udp(coords: np.ndarray, batch_heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """DARK sub-pixel refinement (mmpose / easy_ViTPose port)."""
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N), f"B={B}, N={N}"
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    padded = np.pad(batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='edge').flatten()
    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_     = padded[index]
    ix1    = padded[index + 1]
    iy1    = padded[index + W + 2]
    ix1y1  = padded[index + W + 3]
    ix1_y1_= padded[index - W - 3]
    ix1_   = padded[index - 1]
    iy1_   = padded[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1).reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords


def _udp_decode_full_frame(heatmaps: np.ndarray, W_img: int, H_img: int,
                            kernel: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Decode (1, K, H_hm, W_hm) -> (K, 2) kpt coords in full-frame image
    coordinates, plus (K,) per-kp scores."""
    H_hm, W_hm = heatmaps.shape[-2], heatmaps.shape[-1]
    preds, maxvals = _get_max_preds(heatmaps)
    preds = _post_dark_udp(preds.copy(), heatmaps.astype(np.float32, copy=True), kernel=kernel)
    scale_x = W_img / float(W_hm - 1)
    scale_y = H_img / float(H_hm - 1)
    kpts_xy = preds[0].copy()
    kpts_xy[:, 0] = kpts_xy[:, 0] * scale_x
    kpts_xy[:, 1] = kpts_xy[:, 1] * scale_y
    scores = maxvals[0, :, 0]
    return kpts_xy.astype(np.float32), scores.astype(np.float32)


# ──────────────────────── per-video pass ────────────────────────

class ViTPoseRunner:
    """Lazy per-video ViTPose-Huge wholebody backup.

    Loads the model on first use. ViTPose is the pose cascade's last-resort
    backup, consulted only on frames where every MediaPipe path fails, so
    inference runs ON DEMAND per (video, frame) via `get_for_side` and is
    memoized — never eagerly over the whole clip.
    """

    def __init__(self, ckpt: str | Path = DEFAULT_CKPT,
                 device: str = "cuda", dtype: str = "float16",
                 input_hw: tuple[int, int] = (256, 192)):
        self.ckpt = Path(ckpt)
        self.device = device
        self.dtype = dtype
        self.input_h, self.input_w = input_hw
        self._model = None
        self._torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        # video_path -> {frame_idx -> {'lm_L','lm_R','sc_L','sc_R'}}, lazily filled
        self._cache: dict[Path, dict] = {}

    def _load_model(self):
        if self._model is not None:
            return
        if not self.ckpt.exists():
            raise FileNotFoundError(f"ViTPose checkpoint not found: {self.ckpt}")
        torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
        model = ViTPoseHugeWholeBody(num_keypoints=133, img_size=(self.input_h, self.input_w))
        model.load_upstream_pth(str(self.ckpt))
        model.eval().to(self.device).to(torch_dtype)
        self._model = model
        self._torch_dtype = torch_dtype

    def _infer_frame(self, rgb: np.ndarray) -> dict:
        """Run a single ViT-Huge forward + UDP/DARK decode on one RGB frame.
        Returns {'lm_L','lm_R','sc_L','sc_R'} ((21,2)/(21,) float32) in the
        frame's own pixel coordinates."""
        self._load_model()
        H_img, W_img = rgb.shape[:2]
        img_in = cv2.resize(rgb, (self.input_w, self.input_h),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        img_in = ((img_in - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)[None]
        x = torch.from_numpy(img_in).to(self.device).to(self._torch_dtype)
        with torch.inference_mode():
            heatmaps = self._model(x)
        kpts, scores = _udp_decode_full_frame(
            heatmaps.float().cpu().numpy(), W_img=W_img, H_img=H_img, kernel=11,
        )
        return {
            "lm_L": kpts[LEFT_HAND_IDX].astype(np.float32),
            "lm_R": kpts[RIGHT_HAND_IDX].astype(np.float32),
            "sc_L": scores[LEFT_HAND_IDX].astype(np.float32),
            "sc_R": scores[RIGHT_HAND_IDX].astype(np.float32),
        }

    def run_video(self, video_path: Path, max_frames: int | None = None) -> dict:
        """LAZY registration (no eager inference). ViTPose is the cascade's
        last-resort backup, consulted only on frames where every MP path
        fails. Running it eagerly over the whole clip wastes a full-clip
        ViT-Huge forward + DARK decode on frames never read. So this just
        loads the model + inits the per-frame cache; actual inference happens
        on demand in `get_for_side`."""
        self._load_model()
        self._cache.setdefault(video_path, {})
        return self._cache[video_path]

    def get_for_side(self, video_path: Path, frame_idx: int, side: str,
                      rgb: np.ndarray | None = None
                      ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (kpts_21x2, per_kp_scores_21) for the requested side at this
        frame, or (None, None). Computes ViTPose lazily for `frame_idx` and
        memoizes per (video, frame). If `rgb` (the already-decoded full frame
        in video_path's coordinate frame) is given, it is used directly to
        avoid re-reading the video; otherwise the frame is seeked + read."""
        if side not in ("left", "right"):
            return None, None
        fc = self._cache.setdefault(video_path, {})
        if frame_idx not in fc:
            frame_rgb = rgb
            if frame_rgb is None:
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, bgr = cap.read()
                cap.release()
                if not ok:
                    return None, None
                frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            fc[frame_idx] = self._infer_frame(frame_rgb)
        d = fc[frame_idx]
        if side == "left":
            return d["lm_L"], d["sc_L"]
        return d["lm_R"], d["sc_R"]

    def unload(self):
        """Free model GPU memory between videos to keep the working set small."""
        self._model = None
        torch.cuda.empty_cache()

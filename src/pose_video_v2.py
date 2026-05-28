"""Hand-pose estimation v2 (MP primary, gates anchored to tracker mask hulls).

Stage 1: MediaPipe HandLandmarker (VIDEO mode, num_hands=4, low confidence
0.20 / 0.20) runs once per video; each detection is gated per tracker
obj_id against:

  - wrist-near-hull        wrist (lm 0) must be inside the obj_id's mask
                           convex hull OR within 0.2 * hull diagonal of
                           its border. (The tracker masks are wrist-trimmed,
                           so the wrist often falls just OUTSIDE the mask,
                           which is why we don't gate on wrist-IN-mask.)
  - kpts-in-hull           >= MIN_KPTS_IN_HULL_FRAC of the 21 keypoints must
                           lie inside the mask convex hull (positive
                           pointPolygonTest).
  - size sanity            pose bbox area must not exceed POSE_BBOX_MAX_RATIO
                           times the mask bbox area. Catches "giant
                           hallucinated skeleton" when the hand goes
                           partially off-screen.
  - temporal continuity    once a pose is accepted for an obj_id, any
                           subsequent acceptance whose per-keypoint mean
                           displacement exceeds CONSISTENCY_MAX_PX_FRAC of
                           the image diagonal is rejected (with a short
                           override window of CONT_OVERRIDE_FRAMES).

If MP VIDEO produced no qualifying detection, run MP IMAGE mode on a
50%-expanded square crop around the mask bbox. Same gate suite applies.

Wearer L / R label per obj_id comes from the tracker's
`wearer_handedness_by_obj_id` (set by `src/label_tracking_handedness.py`),
not from MP's per-frame handedness output (MP's handedness is unreliable
on ego-centric clips).

Outputs go to `outputs/pose_v<N>/`:
  <stem>_pose.json   per-frame keypoints, gate diagnostics, source label
  <stem>_pose.mp4    overlay video (tracker mask + hull + skeleton + L/R
                      tag + frame # in top-right)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pycocotools import mask as coco_mask

# ViTPose backup is imported lazily on first need (lots of torch import time)
# from .vitpose_runner import ViTPoseRunner

MP_MODEL_PATHS = [
    Path("/home/jingjin/.cache/mediapipe/hand_landmarker.task"),
    Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task",
]

# Detection / gating parameters
MP_NUM_HANDS_VIDEO = 4
MP_NUM_HANDS_IMAGE = 1
MP_CONF = 0.20

# Wrist-near-hull: wrist may be outside the mask hull by at most this
# fraction of the hull's bbox diagonal (the wrist-trim cuts the wrist off
# the mask, so the actual wrist landmark lands just outside).
WRIST_HULL_MAX_DIST_FRAC = 0.25
# Kpts-in-hull: this fraction of the 21 keypoints must lie inside the
# mask convex hull (strict pointPolygonTest, no buffer). Loose enough to
# admit extended fingers / open hands that reach slightly past the mask.
MIN_KPTS_IN_HULL_FRAC = 0.50
# Expand the convex hull by this factor (about its centroid) before the
# kpts-in-hull check. Fingertips frequently extend a few px past the SAM-mask
# hull on otherwise-correct MP detections; a slight outward dilation lets
# those poses through without loosening the threshold itself.
KPTS_HULL_EXPAND_FRAC = 0.20
# Additional gate: minimum fraction of keypoints that must lie inside the
# ACTUAL tracker mask (not the convex hull). Catches candidates that
# geometrically fall inside the hull but are on a different subject (e.g.,
# coworker hand near the wearer's mask region). 0.0 disables the check.
MASK_KPTS_MIN_FRAC = 0.0
# Size sanity: a candidate pose whose bbox area exceeds this multiple of
# the mask bbox area is treated as a hallucination.
POSE_BBOX_MAX_RATIO = 2.5
# Minimum acceptable pose-bbox / mask-bbox area ratio. Catches both
# degenerate MP image-rerun detections that collapse all 21 kpts to a tiny
# cluster (~20 px diag in a ~180 px mask, ratio ~0.012) AND MP-video
# tracked-in detections that come back compressed when the underlying hand
# is much larger (rgb_03 f183 obj1: pose 90 px in 185 px mask, ratio 0.23).
# 99.3% of legitimate accepted poses across 20 videos have ratio >= 0.30;
# mp_video's 5th percentile is 0.47, vitpose is 0.42. 0.30 catches the
# compressed cases without touching the bulk of legitimate detections.
POSE_BBOX_MIN_RATIO = 0.30
# MP-video size gates (using pose-keypoints convex-hull area / mask
# convex-hull area). VIDEO mode tracking can lock onto a tucked-finger
# pose from prior frames; if the resulting hull-area ratio is below the
# threshold, fall through to MP video IMAGE mode (no tracking). IMAGE
# mode applies the same threshold; below it falls through to the rest
# of the cascade (mp_image_rerun, wide, vitpose) which use the lower
# POSE_BBOX_MIN_RATIO floor.
# After the cascade, runs of compressed-MP-video frames that are
# <= COMPRESSED_INTERP_MAX_LEN AND bracketed by uncompressed mp_video
# accepts are linearly interpolated, overriding the cascade output.
MP_VIDEO_MIN_AREA_RATIO = 0.45
MP_VIDEO_IMAGE_MIN_AREA_RATIO = 0.45
COMPRESSED_INTERP_MAX_LEN = 7
# Temporal jump cap (fraction of image diagonal) for per-kp mean
# displacement vs the most recently accepted pose for the same obj_id.
CONSISTENCY_MAX_PX_FRAC = 0.10
# How long a stretch of rejections may "carry forward" the previous pose
# instead of leaving the frame empty.
CONT_CARRYFWD_MAX_FRAMES = 10
# IMAGE-mode rerun crop expansion (fraction of mask bbox, square) — legacy.
RERUN_BBOX_EXPAND = 0.50
# Mask-dilation expansion: kernel radius as a fraction of mask bbox diagonal.
# 0.10 gives roughly +50% mask area for a typical hand mask. The dilated mask
# becomes the crop (tight) AND the binary mask is used to zero out background
# pixels in the crop fed to MP image -- removes co-worker hands / scene clutter.
MASK_RERUN_DILATE_FRAC = 0.10
# Wide-crop MP image fallback: square crop expanded by this fraction of the
# mask bbox's longer side, no background zeroing. Gives MP enough surrounding
# context (wrist, forearm, scene anchor) to lock onto closed-fist / dorsal
# hand views that the tight mask-zeroed crop misses.
MP_WIDE_CROP_EXPAND_FRAC = 0.75
# Single-resolution MP video: downsample each frame to this short-edge size
# before feeding it to MP. None = use source resolution. 1024 keeps the
# detector cheap while remaining well above MP's internal 256x192 working
# size. Landmark coords are rescaled back into source-frame coordinates.
MP_VIDEO_SHORT_EDGE = 1024
# Minimum mask area for which we even bother running a pose detector.
MIN_MASK_AREA_PX = 2000

# Visualization
MASK_COLORS_BGR = [(255, 80, 0), (0, 80, 255)]    # obj 0 cool, obj 1 warm
HULL_COLORS_BGR = [(255, 220, 0), (0, 220, 255)]
SKEL_COLOR = (255, 0, 255)
KP_COLOR = (255, 0, 255)
WRIST_COLOR = (0, 255, 255)
LABEL_TEXT_COLOR = (0, 255, 255)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


# ──────────────────────── small helpers ────────────────────────

def _find_mp_model() -> Path:
    for p in MP_MODEL_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("hand_landmarker.task not found")


def make_mp(running_mode: mp_vision.RunningMode, num_hands: int) -> mp_vision.HandLandmarker:
    base = mp_python.BaseOptions(model_asset_path=str(_find_mp_model()))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=MP_CONF,
        min_hand_presence_confidence=MP_CONF,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def decode_mask_rle(rle: dict) -> np.ndarray:
    raw = {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")}
    return coco_mask.decode(raw).astype(bool)


def largest_cc_hull(mask: np.ndarray) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    """Return (hull_contour_Nx1x2_int32, hull_bbox_diag_px, component_mask).

    Hull comes from the largest connected component of `mask`.
    """
    if mask is None or mask.sum() == 0:
        return None, 0.0, None
    m_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    if num <= 1:
        return None, 0.0, None
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    comp = (labels == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0.0, None
    pts = np.vstack(contours).reshape(-1, 2).astype(np.int32)
    hull = cv2.convexHull(pts)
    hpts = hull.reshape(-1, 2).astype(np.float32)
    diag = float(np.hypot(hpts[:, 0].max() - hpts[:, 0].min(),
                          hpts[:, 1].max() - hpts[:, 1].min()))
    return hull, diag, comp.astype(bool)


def hull_area(pts) -> float:
    """Convex hull area of a 2D point set (e.g., 21 hand keypoints).
    Returns 0.0 if fewer than 3 points or all collinear."""
    a = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(a) < 3:
        return 0.0
    h = cv2.convexHull(a)
    return float(cv2.contourArea(h))


def mask_hull_area(hull) -> float:
    """Convex hull area of the mask hull (already computed by
    `largest_cc_hull`)."""
    if hull is None:
        return 0.0
    return float(cv2.contourArea(hull.astype(np.int32)))


def bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def kp_bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def expand_to_square_crop(bbox, image_w: int, image_h: int,
                          expand_frac: float = RERUN_BBOX_EXPAND):
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + expand_frac)
    half = side / 2.0
    return (
        max(0, int(round(cx - half))),
        max(0, int(round(cy - half))),
        min(image_w, int(round(cx + half))),
        min(image_h, int(round(cy + half))),
    )


def mp_pts_to_image(landmarks, image_w: int, image_h: int) -> np.ndarray:
    return np.array([[lm.x * image_w, lm.y * image_h] for lm in landmarks], dtype=np.float32)


def mp_pts_in_crop_to_image(landmarks, crop) -> np.ndarray:
    sx1, sy1, sx2, sy2 = crop
    cw, ch = sx2 - sx1, sy2 - sy1
    return np.array([[sx1 + lm.x * cw, sy1 + lm.y * ch] for lm in landmarks], dtype=np.float32)


def dilate_mask_to_crop(mask: np.ndarray,
                         expand_frac: float = MASK_RERUN_DILATE_FRAC
                         ) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """Return (dilated_mask_HxW_bool, dilated_bbox_xyxy).

    Dilates the input mask with a circular kernel of radius = expand_frac
    * mask_bbox_diagonal. ~+50% area when expand_frac == 0.10 for a typical
    hand mask. Returns (None, (0,0,0,0)) for an empty mask.
    """
    if mask is None or mask.sum() == 0:
        return None, (0, 0, 0, 0)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, (0, 0, 0, 0)
    diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    k = max(3, int(round(expand_frac * diag)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    dy, dx = np.where(dilated)
    if len(dx) == 0:
        return None, (0, 0, 0, 0)
    bbox = (int(dx.min()), int(dy.min()), int(dx.max()) + 1, int(dy.max()) + 1)
    return dilated, bbox


# ──────────────────────── gates ────────────────────────

def gate_wrist_near_hull(wrist_xy: np.ndarray, hull: np.ndarray, hull_diag: float
                          ) -> tuple[bool, float, float]:
    """Wrist must be inside the hull or within WRIST_HULL_MAX_DIST_FRAC * hull_diag of it."""
    if hull is None or hull_diag <= 0:
        return False, 0.0, 0.0
    # pointPolygonTest with measureDist=True: > 0 inside, == 0 on edge, < 0 outside.
    dist = float(cv2.pointPolygonTest(
        hull.astype(np.float32), (float(wrist_xy[0]), float(wrist_xy[1])), True
    ))
    thresh = -WRIST_HULL_MAX_DIST_FRAC * hull_diag  # most negative dist allowed
    return (dist >= thresh), dist, thresh


def _expand_hull(hull: np.ndarray, expand_frac: float) -> np.ndarray:
    """Scale a convex hull about its centroid by (1 + expand_frac)."""
    pts = hull.reshape(-1, 2).astype(np.float32)
    centroid = pts.mean(axis=0, keepdims=True)
    return ((pts - centroid) * (1.0 + expand_frac) + centroid).reshape(hull.shape)


def snap_pose_to_mask(held_kpts: np.ndarray, mask_hull: np.ndarray,
                       hull_diag: float) -> np.ndarray:
    """Translate + scale a held pose so its centroid aligns with the mask
    hull centroid and its bbox diagonal matches the hull diagonal. Used
    during carryforward when the underlying detector failed but the mask
    still locates the hand -- keeps the held skeleton visually anchored to
    the actual hand location instead of drifting at the stale frame's
    coordinates.
    """
    if mask_hull is None or hull_diag <= 0:
        return held_kpts
    held = held_kpts.astype(np.float32)
    held_centroid = held.mean(axis=0)
    bbox_w = float(held[:, 0].max() - held[:, 0].min())
    bbox_h = float(held[:, 1].max() - held[:, 1].min())
    held_diag = float(np.hypot(bbox_w, bbox_h))
    mask_centroid = mask_hull.reshape(-1, 2).astype(np.float32).mean(axis=0)
    scale = (hull_diag / held_diag) if held_diag > 1e-3 else 1.0
    return (held - held_centroid) * scale + mask_centroid


def gate_kpts_in_hull(kpts: np.ndarray, hull: np.ndarray) -> tuple[bool, float]:
    """At least MIN_KPTS_IN_HULL_FRAC of the 21 kpts must lie inside the
    convex hull, expanded by KPTS_HULL_EXPAND_FRAC about its centroid."""
    if hull is None:
        return False, 0.0
    h = _expand_hull(hull, KPTS_HULL_EXPAND_FRAC).astype(np.float32)
    inside = 0
    for (x, y) in kpts:
        if cv2.pointPolygonTest(h, (float(x), float(y)), False) >= 0:
            inside += 1
    frac = inside / max(len(kpts), 1)
    return (frac >= MIN_KPTS_IN_HULL_FRAC), frac


def gate_size_sanity(kpts: np.ndarray, mask_hull) -> tuple[bool, float]:
    """Pose-keypoints convex-hull area must be within
    [POSE_BBOX_MIN_RATIO, POSE_BBOX_MAX_RATIO] * mask convex-hull area.
    The upper bound rejects poses that span far beyond the hand mask
    (wrong subject); the lower bound rejects degenerate MP image
    detections that collapse all 21 kpts into a tiny cluster.

    Uses convex hulls (not bounding boxes) so the ratio reflects actual
    spatial extent, not axis-aligned bbox area which is biased by hand
    orientation."""
    mask_a = mask_hull_area(mask_hull)
    if mask_a <= 0:
        return False, 0.0
    pose_a = hull_area(kpts)
    ratio = pose_a / mask_a
    return (POSE_BBOX_MIN_RATIO <= ratio <= POSE_BBOX_MAX_RATIO), ratio


def gate_temporal_jump(kpts: np.ndarray, prev_kpts: np.ndarray | None,
                       image_diag: float, prev_age: int = 0
                       ) -> tuple[bool, float]:
    """Mean per-kp displacement vs prev_kpts must be within a gap-scaled
    threshold. Base threshold is CONSISTENCY_MAX_PX_FRAC * image_diag per
    adjacent frame; for an older prev_pose (separated by `prev_age` failed
    or skipped frames), scale by sqrt(1 + prev_age) under a random-walk
    motion model. With prev_age=0 this matches the original adjacent-frame
    bound; with prev_age=10 the bound is ~3.3x larger, accommodating real
    hand motion across short stretches where MP/MP-image/ViTPose all
    failed to produce a fresh detection."""
    if prev_kpts is None:
        return True, 0.0
    jump = float(np.linalg.norm(kpts - prev_kpts, axis=1).mean())
    base = CONSISTENCY_MAX_PX_FRAC * image_diag
    thresh = base * (1.0 + max(prev_age, 0)) ** 0.5
    return (jump <= thresh), jump


# ──────────────────────── MP wrappers ────────────────────────

def run_mp_video_frame_image(landmarker, frame_rgb: np.ndarray,
                              out_W: int | None = None, out_H: int | None = None
                              ) -> list[dict]:
    """Same as `run_mp_video_frame` but uses `detect()` (IMAGE mode, no
    tracking). First backup when VIDEO-mode tracking returns a too-
    compressed pose; without the tracking bias IMAGE mode often produces
    a fuller pose."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.ascontiguousarray(frame_rgb))
    res = landmarker.detect(img)
    H, W = frame_rgb.shape[:2]
    target_w = out_W if out_W is not None else W
    target_h = out_H if out_H is not None else H
    out = []
    for i, lms in enumerate(res.hand_landmarks):
        pts = mp_pts_to_image(lms, target_w, target_h)
        score = float(res.handedness[i][0].score) if (res.handedness and i < len(res.handedness)) else 0.0
        handed = res.handedness[i][0].category_name if (res.handedness and i < len(res.handedness)) else None
        kp_conf = np.full(21, score, dtype=np.float32)
        out.append({"keypoints": pts, "score": score, "handedness": handed,
                     "kp_confidences": kp_conf})
    return out


def run_mp_video_frame(landmarker, frame_rgb: np.ndarray, ts_ms: int,
                        out_W: int | None = None, out_H: int | None = None
                        ) -> list[dict]:
    """Run MP video on `frame_rgb`. If out_W/out_H are given, scale the
    normalized landmarks into (out_W, out_H) rather than the frame's own
    dimensions -- useful when feeding MP a downsampled frame but you want
    kpts in the source-resolution coordinate frame."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.ascontiguousarray(frame_rgb))
    res = landmarker.detect_for_video(img, int(ts_ms))
    H, W = frame_rgb.shape[:2]
    target_w = out_W if out_W is not None else W
    target_h = out_H if out_H is not None else H
    out = []
    for i, lms in enumerate(res.hand_landmarks):
        pts = mp_pts_to_image(lms, target_w, target_h)
        score = float(res.handedness[i][0].score) if (res.handedness and i < len(res.handedness)) else 0.0
        handed = res.handedness[i][0].category_name if (res.handedness and i < len(res.handedness)) else None
        # MP HandLandmarker doesn't expose per-keypoint confidences; the
        # hand-level score is the best we have. Broadcast it to (21,).
        kp_conf = np.full(21, score, dtype=np.float32)
        out.append({"keypoints": pts, "score": score, "handedness": handed,
                     "kp_confidences": kp_conf})
    return out


def run_mp_image_crop(landmarker, frame_rgb: np.ndarray, crop) -> dict | None:
    """Legacy bbox-only image rerun (kept for compatibility)."""
    sx1, sy1, sx2, sy2 = crop
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    sub = np.ascontiguousarray(frame_rgb[sy1:sy2, sx1:sx2])
    if sub.size == 0:
        return None
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=sub)
    res = landmarker.detect(img)
    if not res.hand_landmarks:
        return None
    pts = mp_pts_in_crop_to_image(res.hand_landmarks[0], crop)
    score = float(res.handedness[0][0].score) if res.handedness else 0.0
    handed = res.handedness[0][0].category_name if res.handedness else None
    kp_conf = np.full(21, score, dtype=np.float32)
    return {"keypoints": pts, "score": score, "handedness": handed,
             "kp_confidences": kp_conf}


def run_mp_image_masked(landmarker, frame_rgb: np.ndarray, mask: np.ndarray
                         ) -> dict | None:
    """MP image rerun on a mask-derived crop with background zeroed out.

    Steps:
      1. Dilate the input mask by ~10% of its bbox diagonal (≈ +50% area).
      2. Crop to the dilated mask's bbox.
      3. Zero pixels outside the dilated mask within the crop. This focuses
         MP on the hand and removes co-worker hands / scene clutter that
         could otherwise compete with the wearer hand for the detection.
    """
    dilated, bbox = dilate_mask_to_crop(mask)
    if dilated is None:
        return None
    sx1, sy1, sx2, sy2 = bbox
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    sub = frame_rgb[sy1:sy2, sx1:sx2].copy()
    sub_mask = dilated[sy1:sy2, sx1:sx2]
    sub[~sub_mask] = 0
    if sub.size == 0:
        return None
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.ascontiguousarray(sub))
    res = landmarker.detect(img)
    if not res.hand_landmarks:
        return None
    pts = mp_pts_in_crop_to_image(res.hand_landmarks[0], bbox)
    score = float(res.handedness[0][0].score) if res.handedness else 0.0
    handed = res.handedness[0][0].category_name if res.handedness else None
    kp_conf = np.full(21, score, dtype=np.float32)
    return {"keypoints": pts, "score": score, "handedness": handed,
             "kp_confidences": kp_conf}


def run_mp_image_wide(landmarker, frame_rgb: np.ndarray, mask: np.ndarray,
                       expand_frac: float = MP_WIDE_CROP_EXPAND_FRAC
                       ) -> dict | None:
    """MP image rerun on a WIDER square crop around the mask, without background
    zeroing. Falls back from `run_mp_image_masked` when the tight + zeroed crop
    is too context-starved (small hands, closed-fist / dorsal views).

    Square crop = mask bbox expanded by `expand_frac` * max(bw, bh) on each
    side, then squared. No mask multiplication — MP sees the surrounding
    scene as anchor.
    """
    if mask is None or mask.sum() == 0:
        return None
    H, W = frame_rgb.shape[:2]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    bw, bh = (x2 - x1), (y2 - y1)
    pad = int(round(expand_frac * max(bw, bh)))
    # Expand then square
    sx1, sy1 = x1 - pad, y1 - pad
    sx2, sy2 = x2 + pad, y2 + pad
    side = max(sx2 - sx1, sy2 - sy1)
    cx = (sx1 + sx2) // 2
    cy = (sy1 + sy2) // 2
    half = side // 2
    sx1, sy1 = cx - half, cy - half
    sx2, sy2 = sx1 + side, sy1 + side
    sx1 = max(0, sx1); sy1 = max(0, sy1)
    sx2 = min(W, sx2); sy2 = min(H, sy2)
    if sx2 - sx1 < 16 or sy2 - sy1 < 16:
        return None
    sub = np.ascontiguousarray(frame_rgb[sy1:sy2, sx1:sx2])
    if sub.size == 0:
        return None
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=sub)
    res = landmarker.detect(img)
    if not res.hand_landmarks:
        return None
    pts = mp_pts_in_crop_to_image(res.hand_landmarks[0], (sx1, sy1, sx2, sy2))
    score = float(res.handedness[0][0].score) if res.handedness else 0.0
    handed = res.handedness[0][0].category_name if res.handedness else None
    kp_conf = np.full(21, score, dtype=np.float32)
    return {"keypoints": pts, "score": score, "handedness": handed,
             "kp_confidences": kp_conf}


# ──────────────────────── per-frame selection ────────────────────────

def gate_kpts_in_mask(kpts: np.ndarray, mask: np.ndarray) -> tuple[bool, float]:
    """Fraction of kpts lying inside the actual tracker mask (not the hull).
    Stricter than gate_kpts_in_hull: discriminates coworker-hand candidates
    that fall inside the wearer mask's hull region but not on the actual
    wearer-hand pixels.
    """
    if mask is None or mask.sum() == 0:
        return False, 0.0
    H, W = mask.shape[:2]
    inside = 0
    for (x, y) in kpts:
        xi = int(round(float(x))); yi = int(round(float(y)))
        if 0 <= xi < W and 0 <= yi < H and bool(mask[yi, xi]):
            inside += 1
    frac = inside / max(len(kpts), 1)
    return (frac >= MASK_KPTS_MIN_FRAC), frac


def gate_candidate(kpts: np.ndarray, mask_hull: np.ndarray, hull_diag: float,
                    mask_bbox, mask: np.ndarray | None = None
                    ) -> tuple[bool, dict]:
    """Run the wrist + kpts + size gates on a candidate pose against an obj_id's
    mask. If `mask` is given AND MASK_KPTS_MIN_FRAC > 0, also requires a
    minimum fraction of kpts inside the actual mask (not just the hull).
    Returns (passes, gate_diag_dict)."""
    diag = {}
    wpass, wdist, wthresh = gate_wrist_near_hull(kpts[0], mask_hull, hull_diag)
    diag["wrist_dist_to_hull_px"] = wdist
    diag["wrist_dist_thresh_px"] = wthresh
    diag["wrist_near_hull"] = wpass
    if not wpass:
        return False, diag
    kpass, kfrac = gate_kpts_in_hull(kpts, mask_hull)
    diag["kpts_in_hull_frac"] = kfrac
    diag["kpts_in_hull"] = kpass
    if not kpass:
        return False, diag
    spass, sratio = gate_size_sanity(kpts, mask_hull)
    diag["pose_bbox_to_mask_bbox_ratio"] = sratio
    diag["size_sanity"] = spass
    if not spass:
        return False, diag
    if MASK_KPTS_MIN_FRAC > 0 and mask is not None:
        mpass, mfrac = gate_kpts_in_mask(kpts, mask)
        diag["kpts_in_mask_frac"] = mfrac
        diag["kpts_in_mask"] = mpass
        if not mpass:
            return False, diag
    return True, diag


def select_best_for_obj(mp_dets: list[dict], mask_hull, hull_diag, mask_bbox,
                         mask: np.ndarray | None = None
                         ) -> tuple[dict | None, list[dict]]:
    """Return the best-scoring MP detection that passes all gates for this obj_id,
    along with per-candidate gate diagnostics."""
    diag_per_cand = []
    qualifying = []
    for i, det in enumerate(mp_dets):
        passes, gd = gate_candidate(det["keypoints"], mask_hull, hull_diag, mask_bbox, mask=mask)
        diag_per_cand.append({"cand_idx": i, "score": det["score"], **gd})
        if passes:
            qualifying.append(det)
    if not qualifying:
        return None, diag_per_cand
    qualifying.sort(key=lambda d: -d["score"])
    return qualifying[0], diag_per_cand


# ──────────────────────── render ────────────────────────

def overlay_mask(img, mask, color_bgr):
    if mask is None or mask.sum() == 0:
        return img
    color = np.array(color_bgr, dtype=np.uint8)
    img[mask] = (0.55 * img[mask] + 0.45 * color).astype(np.uint8)
    return img


def draw_mask_hull(img, hull, color_bgr):
    if hull is None:
        return
    cv2.polylines(img, [hull.astype(np.int32)], isClosed=True,
                  color=color_bgr, thickness=3, lineType=cv2.LINE_AA)


def draw_skeleton(img, pts, _color_unused):
    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, ipts[a], ipts[b], SKEL_COLOR, 2, lineType=cv2.LINE_AA)
    for i, p in enumerate(ipts):
        cv2.circle(img, p, 7, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, p, 5, WRIST_COLOR if i == 0 else KP_COLOR, -1, lineType=cv2.LINE_AA)


def draw_frame_number(img, fidx: int):
    H, W = img.shape[:2]
    text = f"f{fidx}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    x = W - tw - 12
    y = th + 12
    cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


# ──────────────────────── orchestration ────────────────────────

def interpolate_short_compressed_runs(per_frame: list[dict]) -> list[dict]:
    """For each obj_id, find runs of consecutive frames whose entry has
    `mp_video_compressed_ratio` set (the cascade rejected MP video VIDEO
    for being too compressed). Runs of length <= COMPRESSED_INTERP_MAX_LEN
    that are bracketed on BOTH sides by frames whose source is 'mp_video'
    (uncompressed accepts) get linearly interpolated; the backup-cascade
    decision is overwritten with the interpolation. Longer runs keep the
    backup cascade result.

    Returns a log of interpolation events.
    """
    log: list[dict] = []
    if not per_frame:
        return log
    n = len(per_frame)
    def get_entry(fi: int, oid: int) -> dict | None:
        for h in per_frame[fi]["hands"]:
            if h["obj_id"] == oid:
                return h
        return None
    for oid in (0, 1):
        i = 0
        while i < n:
            e = get_entry(i, oid)
            if e is None or "mp_video_compressed_ratio" not in e:
                i += 1
                continue
            run_start = i
            while i < n:
                e2 = get_entry(i, oid)
                if e2 is None or "mp_video_compressed_ratio" not in e2:
                    break
                i += 1
            run_end = i - 1
            run_len = run_end - run_start + 1
            if run_len > COMPRESSED_INTERP_MAX_LEN:
                continue
            prev_idx = None
            for k in range(run_start - 1, -1, -1):
                pe = get_entry(k, oid)
                if pe is not None and pe.get("source") == "mp_video" and pe.get("keypoints") is not None:
                    prev_idx = k
                    break
            next_idx = None
            for k in range(run_end + 1, n):
                ne = get_entry(k, oid)
                if ne is not None and ne.get("source") == "mp_video" and ne.get("keypoints") is not None:
                    next_idx = k
                    break
            if prev_idx is None or next_idx is None:
                continue
            prev_kp = np.asarray(get_entry(prev_idx, oid)["keypoints"], dtype=np.float32)
            next_kp = np.asarray(get_entry(next_idx, oid)["keypoints"], dtype=np.float32)
            span = next_idx - prev_idx
            for fi in range(run_start, run_end + 1):
                t = (fi - prev_idx) / span
                interp_kp = (1.0 - t) * prev_kp + t * next_kp
                e = get_entry(fi, oid)
                e["keypoints"] = interp_kp.tolist()
                e["source"] = "interpolated"
                e["rejected_reason"] = None
                e["kp_confidences"] = [0.05] * 21
            log.append({
                "obj_id": oid, "run": [run_start, run_end],
                "run_len": run_len, "prev_idx": prev_idx, "next_idx": next_idx,
            })
    for rec in per_frame:
        for h in rec["hands"]:
            h.pop("mp_video_compressed_ratio", None)
    return log


def process_one_video(video_path: Path, track_dir: Path, out_dir: Path,
                       mp_video, mp_image, max_frames: int | None = None,
                       vitpose=None,
                       mp_video_short_edge: int | None = None,
                       mp_video_image=None) -> dict:
    stem = video_path.stem
    frames_json_path = track_dir / f"{stem}_track.frames.json"
    track_json_path = track_dir / f"{stem}_track.json"
    if not frames_json_path.exists():
        return {"video": video_path.name, "error": "frames.json missing"}
    frames_meta = json.loads(frames_json_path.read_text())
    track_meta = json.loads(track_json_path.read_text()) if track_json_path.exists() else {}
    handedness_by_oid = {int(k): v for k, v in track_meta.get("wearer_handedness_by_obj_id", {}).items()}
    W, H = frames_meta["size"]
    fps = frames_meta["fps"]
    image_diag = float(np.hypot(W, H))

    cap = cv2.VideoCapture(str(video_path))
    total_frames = len(frames_meta["frames"])
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    prev_pose: dict[int, np.ndarray | None] = {0: None, 1: None}
    prev_pose_age: dict[int, int] = {0: 0, 1: 0}     # frames since last accepted
    per_frame: list[dict] = []

    # If ViTPose backup is available, run it once over the (truncated) clip
    # and cache per-frame left/right hand keypoints. We only consult it when
    # both MP VIDEO and MP IMAGE rerun have failed for an obj_id.
    vit_active = vitpose is not None
    if vit_active:
        vitpose.run_video(video_path, max_frames=total_frames)

    out_video = out_dir / f"{stem}_pose.mp4"
    # Writer is created after the cascade loop + post-process (two-pass design).

    for fidx in range(total_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ts_ms = int(round(fidx * 1000.0 / fps))

        # Tracker masks for this frame
        f_entry = frames_meta["frames"][fidx]
        masks: dict[int, np.ndarray] = {}
        mask_bboxes: dict[int, list[float]] = {}
        hulls: dict[int, np.ndarray] = {}
        hull_diags: dict[int, float] = {}
        for h in f_entry["hands"]:
            if h.get("mask_rle") is None:
                continue
            m = decode_mask_rle(h["mask_rle"])
            if m.sum() == 0:
                continue
            oid = int(h["obj_id"])
            masks[oid] = m
            mask_bboxes[oid] = h["bbox"]
            hull, diag, _ = largest_cc_hull(m)
            hulls[oid] = hull
            hull_diags[oid] = diag

        # MP VIDEO pass at a single (possibly downsampled) resolution.
        # Landmark coords are rescaled into source-frame coordinates via
        # the out_W / out_H args.
        if mp_video_short_edge is not None and min(W, H) > mp_video_short_edge:
            scale = mp_video_short_edge / float(min(W, H))
            new_w, new_h = int(round(W * scale)), int(round(H * scale))
            rgb_for_mp = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            rgb_for_mp = rgb
        mp_dets = run_mp_video_frame(mp_video, rgb_for_mp, ts_ms, out_W=W, out_H=H)
        # Lazy IMAGE-mode detections; computed only when at least one
        # obj's VIDEO-mode candidate fails the compression gate.
        mp_dets_image_cache: list[dict] | None = None

        frame_record = {"frame": fidx, "hands": []}
        accepted_pts: dict[int, np.ndarray] = {}
        for oid in (0, 1):
            entry = {
                "obj_id": oid,
                "source": None,
                "keypoints": None,
                "kp_confidences": None,
                "wearer_handedness": handedness_by_oid.get(oid),
                "mp_score": None,
                "rejected_reason": None,
                "gate_diag": None,
                "candidates_diag": None,
            }
            if oid not in masks or masks[oid].sum() < MIN_MASK_AREA_PX:
                entry["rejected_reason"] = "mask_too_small_or_missing"
                # Reset prev_pose age too — but allow short carry-forward if we
                # had a recent accepted pose
                frame_record["hands"].append(entry)
                continue

            # Try each detector path in order. Each candidate must pass both
            # the hull gates AND the temporal-jump check; the first one that
            # passes both wins. This lets MP image / wide / ViTPose rescue
            # frames where MP video found a hull-passing candidate that then
            # failed the temporal jump (e.g. MP picked a wrong hand from
            # among multiple in-frame, while MP image masked or wide was
            # forced to look in the right region).
            mp_video_best, diag_per_cand = select_best_for_obj(
                mp_dets, hulls[oid], hull_diags[oid], mask_bboxes[oid],
                mask=masks[oid],
            )
            entry["candidates_diag"] = diag_per_cand

            best = None
            attempt_fails: list[str] = []

            def _try(cand: dict | None, source_name: str, gate_diag_key: str,
                      no_det_tag: str, gate_tag: str, jump_tag: str) -> dict | None:
                """Run gate_candidate + temporal_jump on `cand`. On pass,
                set entry['source'] and return cand. On fail, append a tag
                to attempt_fails and return None."""
                if cand is None:
                    attempt_fails.append(no_det_tag)
                    return None
                passes, gd = gate_candidate(
                    cand["keypoints"], hulls[oid], hull_diags[oid], mask_bboxes[oid],
                    mask=masks[oid],
                )
                if gate_diag_key:
                    entry[gate_diag_key] = gd
                if not passes:
                    attempt_fails.append(gate_tag)
                    return None
                jump_ok, jp = gate_temporal_jump(
                    cand["keypoints"], prev_pose[oid], image_diag,
                    prev_age=prev_pose_age[oid],
                )
                if not jump_ok:
                    attempt_fails.append(f"{jump_tag}_{jp:.0f}px")
                    return None
                entry["source"] = source_name
                return cand

            # 1. MP video best candidate. Additionally check size: if the
            #    accepted pose's area is < MP_VIDEO_MIN_AREA_RATIO of mask
            #    area, treat as compressed (tucked-finger from VIDEO-mode
            #    tracking) and fall through. The post-pass can interpolate
            #    short compressed runs, while longer runs use the backup
            #    cascade results.
            best = _try(mp_video_best, "mp_video", "gate_diag",
                         "mp_video_no_candidate", "mp_video_gate_fail",
                         "mp_video_temporal_jump")
            if best is not None:
                mask_a = mask_hull_area(hulls[oid])
                pose_a = hull_area(best["keypoints"])
                area_ratio = pose_a / mask_a if mask_a > 0 else 1.0
                if area_ratio < MP_VIDEO_MIN_AREA_RATIO:
                    attempt_fails.append(f"mp_video_compressed_{area_ratio:.2f}")
                    entry["mp_video_compressed_ratio"] = round(area_ratio, 3)
                    entry["source"] = None
                    best = None
            # 2. MP video IMAGE mode (no tracking) — first backup. Pick its
            #    best candidate matching this obj's hull, then apply the
            #    same hull-area size gate at MP_VIDEO_IMAGE_MIN_AREA_RATIO.
            if best is None and mp_video_image is not None:
                if mp_dets_image_cache is None:
                    mp_dets_image_cache = run_mp_video_frame_image(
                        mp_video_image, rgb_for_mp, out_W=W, out_H=H,
                    )
                mp_img_best, _ = select_best_for_obj(
                    mp_dets_image_cache, hulls[oid], hull_diags[oid],
                    mask_bboxes[oid], mask=masks[oid],
                )
                best = _try(mp_img_best, "mp_video_image", "video_image_gate_diag",
                             "mp_video_image_no_candidate",
                             "mp_video_image_gate_fail",
                             "mp_video_image_temporal_jump")
                if best is not None:
                    mask_a = mask_hull_area(hulls[oid])
                    pose_a = hull_area(best["keypoints"])
                    area_ratio = pose_a / mask_a if mask_a > 0 else 1.0
                    if area_ratio < MP_VIDEO_IMAGE_MIN_AREA_RATIO:
                        attempt_fails.append(f"mp_video_image_compressed_{area_ratio:.2f}")
                        entry["source"] = None
                        best = None
            # 2. MP image rerun (mask-zeroed tight crop)
            if best is None:
                best = _try(run_mp_image_masked(mp_image, rgb, masks[oid]),
                             "mp_image_rerun", "gate_diag",
                             "image_rerun_no_detection", "image_rerun_gate_fail",
                             "image_rerun_temporal_jump")
            # 3. MP image wide (no background zeroing)
            if best is None:
                best = _try(run_mp_image_wide(mp_image, rgb, masks[oid]),
                             "mp_image_rerun_wide", "wide_gate_diag",
                             "wide_no_detection", "wide_gate_fail",
                             "wide_temporal_jump")
            # 4. ViTPose backup (requires wearer L/R label)
            if best is None and vit_active:
                side = handedness_by_oid.get(oid)
                if side in ("left", "right"):
                    v_kpts, v_kp_scores = vitpose.get_for_side(video_path, fidx, side)
                    v_cand = None
                    if v_kpts is not None and v_kpts.shape == (21, 2):
                        v_mean = float(v_kp_scores.mean()) if v_kp_scores is not None else 0.0
                        entry["vitpose_mean_score"] = v_mean
                        v_cand = {"keypoints": v_kpts, "score": v_mean,
                                   "handedness": side.capitalize(),
                                   "kp_confidences": v_kp_scores}
                    best = _try(v_cand, "vitpose_huge", "vitpose_gate_diag",
                                 "vit_no_detection", "vit_gate_fail",
                                 "vit_temporal_jump")

            if best is not None:
                entry["keypoints"] = best["keypoints"]
                entry["mp_score"] = best.get("score")
                entry["kp_confidences"] = best.get("kp_confidences")
                entry["rejected_reason"] = None
                prev_pose[oid] = best["keypoints"]
                prev_pose_age[oid] = 0
            else:
                # All paths failed. Record the chain for diagnostics, then
                # carryforward if we still have a recent accepted pose.
                entry["rejected_reason"] = "_then_".join(attempt_fails) or "no_paths_tried"
                if prev_pose[oid] is not None and prev_pose_age[oid] < CONT_CARRYFWD_MAX_FRAMES:
                    entry["source"] = "carryforward"
                    entry["keypoints"] = snap_pose_to_mask(
                        prev_pose[oid], hulls[oid], hull_diags[oid],
                    )
                    entry["kp_confidences"] = np.full(21, 0.05, dtype=np.float32)
                    prev_pose_age[oid] += 1
                else:
                    # Window exhausted: release stale prev_pose so the next
                    # fresh detection can re-acquire.
                    entry["keypoints"] = None
                    prev_pose[oid] = None
                    prev_pose_age[oid] = CONT_CARRYFWD_MAX_FRAMES

            if entry["keypoints"] is not None:
                accepted_pts[oid] = entry["keypoints"]
            frame_record["hands"].append(entry)

        per_frame.append(frame_record)

    cap.release()

    # ── Post-process: short compressed-MP-video runs ───────────────────
    # Identify runs of consecutive frames where MP video VIDEO was
    # rejected for compression (mp_video_compressed_ratio set on entry).
    # Short runs (<= COMPRESSED_INTERP_MAX_LEN) bracketed by uncompressed
    # mp_video accepts get linearly interpolated, overwriting the
    # backup-cascade decision.
    interp_log = interpolate_short_compressed_runs(per_frame)

    # ── Re-render mp4 using final (post-processed) entries ─────────────
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap = cv2.VideoCapture(str(video_path))
    for fidx in range(total_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        f_entry = frames_meta["frames"][fidx]
        masks: dict[int, np.ndarray] = {}
        for h in f_entry["hands"]:
            if h.get("mask_rle") is None:
                continue
            m = decode_mask_rle(h["mask_rle"])
            if m.sum() > 0:
                masks[int(h["obj_id"])] = m
        hulls = {}
        for oid, m in masks.items():
            hull, _, _ = largest_cc_hull(m)
            hulls[oid] = hull
        rec = per_frame[fidx]
        overlay = bgr.copy()
        for oid, m in masks.items():
            overlay = overlay_mask(overlay, m, MASK_COLORS_BGR[oid % 2])
            draw_mask_hull(overlay, hulls[oid], HULL_COLORS_BGR[oid % 2])
        for h in rec["hands"]:
            kp = h.get("keypoints")
            if kp is None:
                continue
            oid = int(h["obj_id"])
            pts = np.asarray(kp, dtype=np.float32)
            draw_skeleton(overlay, pts, MASK_COLORS_BGR[oid % 2])
            wx, wy = int(round(pts[0][0])), int(round(pts[0][1]))
            tag = handedness_by_oid.get(oid)
            tag_short = "L" if tag == "left" else "R" if tag == "right" else "?"
            src = h.get("source") or "-"
            text = f"{tag_short} obj{oid} {src}"
            cv2.putText(overlay, text, (wx + 14, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, text, (wx + 14, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LABEL_TEXT_COLOR, 2, cv2.LINE_AA)
        draw_frame_number(overlay, fidx)
        writer.write(overlay)
    cap.release()
    writer.release()

    # Serialize keypoints + confidences as lists
    serial = []
    for rec in per_frame:
        hands_out = []
        for h in rec["hands"]:
            hh = dict(h)
            if isinstance(hh.get("keypoints"), np.ndarray):
                hh["keypoints"] = hh["keypoints"].tolist()
            if isinstance(hh.get("kp_confidences"), np.ndarray):
                hh["kp_confidences"] = [float(c) for c in hh["kp_confidences"]]
            hands_out.append(hh)
        serial.append({"frame": rec["frame"], "hands": hands_out})

    summary = {
        "video": video_path.name,
        "size": [W, H],
        "fps": fps,
        "n_frames_processed": total_frames,
        "wearer_handedness_by_obj_id": handedness_by_oid,
        "params": {
            "mp_num_hands_video": MP_NUM_HANDS_VIDEO,
            "mp_conf": MP_CONF,
            "wrist_hull_max_dist_frac": WRIST_HULL_MAX_DIST_FRAC,
            "min_kpts_in_hull_frac": MIN_KPTS_IN_HULL_FRAC,
            "pose_bbox_max_ratio": POSE_BBOX_MAX_RATIO,
            "consistency_max_px_frac": CONSISTENCY_MAX_PX_FRAC,
            "cont_carryfwd_max_frames": CONT_CARRYFWD_MAX_FRAMES,
            "min_mask_area_px": MIN_MASK_AREA_PX,
            "mask_rerun_dilate_frac": MASK_RERUN_DILATE_FRAC,
            "mp_video_short_edge": MP_VIDEO_SHORT_EDGE,
        },
        "frames": serial,
    }
    (out_dir / f"{stem}_pose.json").write_text(json.dumps(summary))
    return {"video": video_path.name, "pose_video": out_video.name,
            "pose_json": f"{stem}_pose.json"}


def latest_track_dir(base: Path) -> Path:
    cands = sorted(base.glob("track_v*"), key=lambda p: int(p.name.split("v")[-1]))
    if not cands:
        raise FileNotFoundError(f"no track_v* dirs in {base}")
    return cands[-1]


def pick_next_pose_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        d = base / f"pose_v{n}"
        if not d.exists():
            d.mkdir()
            return d
        n += 1


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-dir", default=None, help="Tracker output dir. Default: latest outputs/track_v*.")
    ap.add_argument("--source-dir", default=None,
                    help="Directory containing the source mp4s. Default: data/.")
    ap.add_argument("--videos", nargs="*", default=None, help="Video stems (e.g. rgb_01).")
    ap.add_argument("--num", type=int, default=10,
                    help="If --videos is unset, process the first N rgb_*.mp4 from --source-dir.")
    ap.add_argument("--max-sec", type=float, default=10.0,
                    help="Cap each video to N seconds (default 10).")
    ap.add_argument("--output-base", default="outputs")
    ap.add_argument("--vitpose", action="store_true",
                    help="Enable ViTPose-Huge wholebody as a final backup "
                         "(after MP VIDEO + MP IMAGE rerun). Used only on "
                         "frames where MP failed and the obj_id has a known "
                         "wearer L/R label.")
    ap.add_argument("--vitpose-ckpt", default=None,
                    help="Override ViTPose checkpoint path.")
    ap.add_argument("--mask-kpts-min-frac", type=float, default=None,
                    help=f"Override MASK_KPTS_MIN_FRAC. Min fraction of keypoints "
                         f"that must lie inside the actual tracker mask (not just "
                         f"the convex hull). 0.0 disables. Default: {MASK_KPTS_MIN_FRAC}.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.mask_kpts_min_frac is not None:
        global MASK_KPTS_MIN_FRAC
        MASK_KPTS_MIN_FRAC = float(args.mask_kpts_min_frac)
    root = Path(__file__).resolve().parents[1]
    track_dir = Path(args.track_dir).resolve() if args.track_dir else latest_track_dir(root / args.output_base)
    source_dir = Path(args.source_dir).resolve() if args.source_dir else (root / "data")
    out_dir = pick_next_pose_dir(root / args.output_base)
    print(f"Track dir:  {track_dir}")
    print(f"Source dir: {source_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Per-video cap: {args.max_sec}s")

    if args.videos:
        stems = args.videos
    else:
        all_mp4 = sorted(source_dir.glob("rgb_*.mp4"))
        stems = [p.stem for p in all_mp4[: args.num]]
    print(f"Videos to process: {len(stems)}")

    mp_image = make_mp(mp_vision.RunningMode.IMAGE, num_hands=MP_NUM_HANDS_IMAGE)

    vitpose = None
    if args.vitpose:
        # Defer the import so the script starts fast when ViTPose is unused.
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vitpose_runner import ViTPoseRunner, DEFAULT_CKPT  # type: ignore
        ckpt = Path(args.vitpose_ckpt).expanduser() if args.vitpose_ckpt else DEFAULT_CKPT
        print(f"ViTPose backup enabled. Checkpoint: {ckpt}")
        vitpose = ViTPoseRunner(ckpt=ckpt)

    summary = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        video_path = source_dir / f"{stem}.mp4"
        if not video_path.exists():
            print(f"  [{i}/{len(stems)}] {stem}: source mp4 missing, skip")
            summary.append({"video": stem, "error": "source missing"})
            continue
        tic = time.time()
        try:
            # MP VIDEO mode needs a fresh landmarker per video (monotonic ts_ms).
            mp_video = make_mp(mp_vision.RunningMode.VIDEO, num_hands=MP_NUM_HANDS_VIDEO)
            # IMAGE-mode landmarker used as first backup when VIDEO-mode
            # tracking returns a compressed pose.
            mp_video_image = make_mp(mp_vision.RunningMode.IMAGE, num_hands=MP_NUM_HANDS_VIDEO)
            cap = cv2.VideoCapture(str(video_path))
            fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            max_frames = int(round(args.max_sec * fps_native))
            res = process_one_video(
                video_path, track_dir, out_dir, mp_video, mp_image,
                max_frames=max_frames, vitpose=vitpose,
                mp_video_short_edge=MP_VIDEO_SHORT_EDGE,
                mp_video_image=mp_video_image,
            )
            mp_video.close()
            mp_video_image.close()
            if vitpose is not None:
                # Drop this video's cache to avoid retaining all frames in RAM.
                vitpose._cache.pop(video_path, None)
            summary.append(res)
            print(f"  [{i}/{len(stems)}] {stem}: done in {time.time()-tic:.1f}s")
        except Exception as e:
            import traceback
            print(f"  [{i}/{len(stems)}] {stem}: ERROR {e!r}")
            traceback.print_exc()
            summary.append({"video": stem, "error": repr(e)})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {time.time()-t0:.1f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()

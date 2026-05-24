"""SAM 2 video tracking of hands, seeded by SAM 3 "hand from above" detections.

For each video in data/*.mp4:
  1. Build PAIRED reseed indices: a pair at the start of every RESEED_INTERVAL_SEC,
     with the second member PAIR_OFFSET_FRAMES after the first.
     e.g. (0, 50), (150, 200), (300, 350), ...
  2. At each seed frame run SAM 3 with prompt "hand from above". SAM 3 outputs
     instance masks + boxes + scores. Apply NMS on the boxes (handles SAM 3's
     occasional duplicate masks per concept) and the max-area filter.
  3. obj_id assignment maintains identity across seeds: the first seed with hands
     gets x-sorted obj_ids (leftmost -> 0, rightmost -> 1); subsequent seeds
     greedily match new boxes to existing obj_ids by centroid distance. Unmatched
     new boxes get the lowest free obj_id (up to 2).
  4. Feed the SAM 3 *masks* (not bboxes) into the SAM 2 video predictor via
     `add_new_mask`, so SAM 2 propagates from a clean mask anchored to the
     actual hand rather than re-segmenting a bbox that may contain other
     prominent objects (toys, tissues, etc.).
  5. Propagate FORWARD across the whole video.
  6. For each pair (a, b), propagate BACKWARD from b to a, overriding masks in
     [a, b-1]. This covers gaps when one seed of the pair missed a hand.
  7. Write an MP4 with masks overlaid + a per-frame COCO-RLE-mask JSON.

All SAM 2 ops run under bfloat16 autocast (avoids a known dtype-mismatch crash
in memory_attention on some videos).

Outputs go to outputs/track_v<N>/ where <N> auto-increments.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pycocotools import mask as coco_mask
from sam2.sam2_video_predictor import SAM2VideoPredictor
from scipy.optimize import linear_sum_assignment
from transformers import Sam3Model, Sam3Processor

SAM3_MODEL_ID = "facebook/sam3"
SAM2_VIDEO_MODEL_ID = "facebook/sam2-hiera-base-plus"

SAM3_PROMPT = "hand from above"
# Fallback prompt used when post-pass detects that obj_0 and obj_1 collapsed
# onto the same blob (one hand was lost). The whole video is re-tracked with
# this prompt, which often picks up the actual hand pair more reliably in
# scenes where "hand from above" anchored on a non-hand region first.
FALLBACK_SAM3_PROMPT = "egocentric first person's hands"
SAM3_SCORE_THRESHOLD = 0.50
SAM3_MASK_THRESHOLD = 0.50
# Fallback prompt is more specific and SAM 3 scores it noticeably lower; needs
# its own relaxed thresholds or it returns 0 candidates. Empirical: rgb_33 at
# every seed frame returns 0 at 0.50 but 1-2 verified hands at 0.20.
FALLBACK_SAM3_SCORE_THRESHOLD = 0.20
FALLBACK_SAM3_MASK_THRESHOLD = 0.20
SAM3_NMS_IOU = 0.50  # SAM 3 occasionally emits overlapping duplicate masks for the same hand
MAX_HAND_BBOX_AREA_FRAC = 0.20
MAX_HANDS = 2  # wearer has at most two hands

RESEED_INTERVAL_SEC = 5.0
PAIR_OFFSET_FRAMES = 50          # second seed of each pair is this many frames after the first
MASK_COLORS_BGR = [(255, 80, 0), (0, 80, 255)]  # obj 0 = blue-ish, obj 1 = red-ish (BGR)

# SAM 2 video-predictor memory offload. Frames at 1920x1440 ~= 8 MB each on
# the GPU; a 1300-frame clip would push >10 GB just for the video tensor.
# These thresholds switch on CPU offload past comfortable budgets on a 24 GB
# 4090, trading throughput for memory headroom.
OFFLOAD_VIDEO_FRAME_THRESHOLD = 600
OFFLOAD_STATE_FRAME_THRESHOLD = 2000

# For longer videos we pre-decode the mp4 into a JPEG folder so SAM 2 can
# stream frames (async_loading_frames=True). The mp4-path code path in SAM 2
# loads all frames into one big tensor upfront, which OOMs anything over a
# couple thousand frames on a 4090 even with offload_video_to_cpu=True.
JPEG_STREAMING_FRAME_THRESHOLD = 1000

# Even with JPEG streaming + offload, SAM 2's AsyncVideoFrameLoader caches
# every loaded frame in CPU RAM and never evicts. At 1024x1024x3xfloat32
# (~12 MB per frame post-resize), beyond ~2000 frames the host runs out of
# RAM (~62 GB on this box) and the kernel OOM-kills the process. For long
# videos we pre-truncate to MAX_TRACK_SEC seconds via ffmpeg before handing
# the (now-shorter) mp4 to the tracker. Set this to None / very large to
# disable truncation.
MAX_TRACK_SEC_DEFAULT = 60.0

# Mask-collapse guard: when two obj_ids' masks share too much IoU OR their
# centroids are too close for too long, both are likely locked on the same hand.
# Zero out the smaller mask until the next reseed splits them apart.
MASK_COLLAPSE_IOU = 0.30                  # IoU >= this counts as overlapping
MASK_COLLAPSE_CENTROID_PX_FRAC = 0.05     # OR centroid distance < this * image diagonal
MASK_COLLAPSE_MIN_RUN = 3                 # require this many consecutive overlapping frames

# Lost-track detection: scan every N frames; if MP can't anchor a hand landmark
# inside the obj_id's mask, the mask is on a non-hand. Trigger SAM 3 re-detection
# for that obj_id and re-seed SAM 2 with the matched detection.
LOST_TRACK_CHECK_INTERVAL = 15            # check every K frames (~0.5s at 30fps)
LOST_TRACK_MIN_LOST_RUN = 2               # require at least 2 consecutive failed checks

# Wrist-trim (helps when SAM 3 segments long gloves / forearm along with the hand)
MP_MODEL_PATHS = [
    Path("/home/jingjin/.cache/mediapipe/hand_landmarker.task"),
    Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task",
]
WRIST_TRIM_CROP_EXPAND = 0.50         # expand mask bbox by this much before squaring+cropping for MP
WRIST_TRIM_MIN_RETAIN_FRAC = 0.30     # if trim would remove >70% of mask, skip (sanity)
WRIST_TRIM_MARGIN_FRAC = -0.06        # shift cut towards the forearm by |this| * bbox diagonal,
                                      # so the wrist itself and a small buffer of forearm are kept
                                      # (negative = away from palm)

# Seed verification: reject SAM 3 detections that don't look like hands
SEED_VERIFY_MIN_SOLIDITY = 0.55       # mask_area / convex_hull_area (hands ~ 0.6-0.85)
SEED_VERIFY_MAX_ASPECT = 3.5          # bbox long/short ratio (hands rarely > 3:1)

# Full-collapse detection: after the tracking pipeline finishes, if obj_0 and
# obj_1 end up on essentially the same blob (>= 95% IoU on BOTH bbox and mask)
# for at least FULL_COLLAPSE_MIN_FRAMES frames, redo the entire video with
# FALLBACK_SAM3_PROMPT. Replaces in-pass salvage for rgb_33-style failures.
FULL_COLLAPSE_IOU = 0.95
FULL_COLLAPSE_MIN_FRAMES = 3

# Mask-spike filter: a short-run mask anomaly where the centroid jumps more
# than ~10% of the image diagonal (or area inflates >2x) and recovers within
# MAX_RUN frames is treated as a tracking spike. The spike frames are replaced
# with the pre-spike mask. Targets rgb_09 (right-hand cross-screen jump ~1s)
# and rgb_21 (left-hand inflated to entire arm ~1s).
MASK_SPIKE_CENTROID_JUMP_FRAC = 0.10
MASK_SPIKE_AREA_RATIO = 2.0
MASK_SPIKE_MAX_RUN = 20
# When the spike filter would replace mask[oid] at frame f with a mask that
# overlaps the OTHER obj_id's mask at f by >= this IoU, skip the replacement.
# Prevents the filter from creating a collapse (rgb_03 ~26s: replacing obj_1's
# seed-LEFT mask at f800 with the prior-frame RIGHT mask collided with obj_0's
# already-RIGHT mask, yielding a 5-frame double-bbox).
MASK_SPIKE_COLLISION_IOU = 0.30

# Seed-frame label-swap fix: Hungarian at the seed step uses last_box_per_obj
# (the prior seed's box) and can misassign obj_ids when the hands moved a lot
# between seeds. We detect this AT EACH SEED FRAME by comparing the seed
# masks at sf to the propagated masks at sf-1: if obj_0/obj_1 at sf are
# closer to each other's prior-frame positions than to their own, we swap
# labels at the seed frame only (1-frame correction, doesn't propagate
# forward, so SAM 2's later revert keeps things consistent).
SEED_LABEL_SWAP_MARGIN_PX = 20.0

# Cross-person hand filter: in ego-centric video the wearer's hands almost
# always sit in the middle / lower half of the frame (the wearer's body is
# behind/below the camera, hands extend forward). A mask whose centroid sits
# in the TOP CROSS_PERSON_TOP_FRAC of the frame for at least
# CROSS_PERSON_MIN_RUN consecutive frames is almost always a hand belonging
# to another person reaching across the workspace from the opposite side
# (rgb_17 obj_0 from f680+: SAM 2 drift latched onto a co-worker's hand).
# Zero out those frames so the visualization shows "lost" rather than the
# wrong hand.
CROSS_PERSON_TOP_FRAC = 0.20
CROSS_PERSON_MIN_RUN = 5


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes, scores, iou_thresh):
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) < iou_thresh for j in keep):
            keep.append(int(i))
    return keep


def indices_under_max_area(boxes, image_size):
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    W, H = image_size
    image_area = float(W) * float(H)
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return np.where(widths * heights / image_area <= MAX_HAND_BBOX_AREA_FRAC)[0]


def load_sam3(device):
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, device_map=device).eval()
    return proc, model


def _find_mp_model() -> Path:
    for p in MP_MODEL_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("hand_landmarker.task not found")


def load_mp_image() -> mp_vision.HandLandmarker:
    base = mp_python.BaseOptions(model_asset_path=str(_find_mp_model()))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def expand_to_square_crop_xyxy(bbox, image_w: int, image_h: int, expand_frac: float):
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


def seed_is_hand_like(
    mask: np.ndarray, bbox: np.ndarray, frame_rgb: np.ndarray, mp_image: mp_vision.HandLandmarker
) -> tuple[bool, str]:
    """Decide whether a SAM 3 candidate is a real hand.

    Pass criteria (any one is sufficient):
      - MP HandLandmarker IMAGE mode finds a hand whose **wrist landmark falls
        inside this candidate's SAM 3 mask** (not just somewhere in the wider
        crop where a different nearby hand could trigger a false confirmation), OR
      - the mask is hand-shaped: solidity >= SEED_VERIFY_MIN_SOLIDITY AND
        bbox aspect ratio <= SEED_VERIFY_MAX_ASPECT.
    Returns (passes, reason).
    """
    if mask.sum() == 0:
        return False, "empty_mask"
    H, W = mask.shape[:2]
    sx1, sy1, sx2, sy2 = expand_to_square_crop_xyxy(bbox, W, H, WRIST_TRIM_CROP_EXPAND)
    mp_found_hand_outside = False
    if sx2 > sx1 and sy2 > sy1:
        crop = np.ascontiguousarray(frame_rgb[sy1:sy2, sx1:sx2])
        if crop.size > 0:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            result = mp_image.detect(mp_img)
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                wrist_x = int(round(lms[0].x * (sx2 - sx1) + sx1))
                wrist_y = int(round(lms[0].y * (sy2 - sy1) + sy1))
                if 0 <= wrist_x < W and 0 <= wrist_y < H and bool(mask[wrist_y, wrist_x]):
                    return True, "mp_wrist_in_mask"
                # Also accept if any of palm-base landmarks (5, 9, 13, 17) is in mask
                for li in (5, 9, 13, 17):
                    px = int(round(lms[li].x * (sx2 - sx1) + sx1))
                    py = int(round(lms[li].y * (sy2 - sy1) + sy1))
                    if 0 <= px < W and 0 <= py < H and bool(mask[py, px]):
                        return True, f"mp_landmark_{li}_in_mask"
                mp_found_hand_outside = True

    # Shape fallback for gloves / occlusions where MP can't anchor inside the mask.
    ys, xs = np.where(mask)
    pts = np.column_stack([xs, ys]).astype(np.int32)
    if len(pts) < 8:
        return False, "too_few_pts"
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull))
    mask_area = float(mask.sum())
    solidity = mask_area / hull_area if hull_area > 0 else 0.0
    bw = float(bbox[2] - bbox[0])
    bh = float(bbox[3] - bbox[1])
    short, long_ = (bw, bh) if bw <= bh else (bh, bw)
    aspect = long_ / max(short, 1.0)
    note = " (mp_hand_was_elsewhere)" if mp_found_hand_outside else ""
    if solidity >= SEED_VERIFY_MIN_SOLIDITY and aspect <= SEED_VERIFY_MAX_ASPECT:
        return True, f"shape ok (sol={solidity:.2f}, ar={aspect:.2f}){note}"
    return False, f"shape rejected (sol={solidity:.2f}, ar={aspect:.2f}){note}"


def _mp_finds_hand_in_mask(
    mp_image: mp_vision.HandLandmarker,
    mask: np.ndarray,
    frame_rgb: np.ndarray,
) -> bool:
    """Run MP on a crop around the mask; return True if MP's wrist or a palm-base
    landmark falls inside the mask. Same anchoring rule as `seed_is_hand_like`.
    """
    if mask is None or mask.sum() == 0:
        return False
    H, W = mask.shape[:2]
    bbox = mask_bbox(mask)
    if bbox is None:
        return False
    sx1, sy1, sx2, sy2 = expand_to_square_crop_xyxy(bbox, W, H, WRIST_TRIM_CROP_EXPAND)
    if sx2 <= sx1 or sy2 <= sy1:
        return False
    crop = np.ascontiguousarray(frame_rgb[sy1:sy2, sx1:sx2])
    if crop.size == 0:
        return False
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = mp_image.detect(mp_img)
    if not result.hand_landmarks:
        return False
    lms = result.hand_landmarks[0]
    cw, ch = sx2 - sx1, sy2 - sy1
    for li in (0, 5, 9, 13, 17):  # wrist + palm-base knuckles
        px = int(round(lms[li].x * cw + sx1))
        py = int(round(lms[li].y * ch + sy1))
        if 0 <= px < W and 0 <= py < H and bool(mask[py, px]):
            return True
    return False


def find_lost_track_segments(
    masks_per_frame: dict[int, dict[int, np.ndarray]],
    video_path: Path,
    mp_image: mp_vision.HandLandmarker,
    reseed_frames: set[int],
) -> dict[int, list[tuple[int, int]]]:
    """Return {obj_id: [(start_lost_frame, ...), ...]} for tracks that no longer have a hand.

    Sampled every LOST_TRACK_CHECK_INTERVAL frames. A "lost" run is at least
    LOST_TRACK_MIN_LOST_RUN consecutive failed checks. The start of the run
    is what we'll use as the SAM 3 re-detection frame.
    """
    sample_frames = sorted(
        f for f in masks_per_frame.keys()
        if f % LOST_TRACK_CHECK_INTERVAL == 0
    )
    if not sample_frames:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    needed = set(sample_frames)
    frame_cache: dict[int, np.ndarray] = {}
    fidx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fidx in needed:
            frame_cache[fidx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fidx += 1
    cap.release()
    lost_runs: dict[int, list[tuple[int, int]]] = {}
    streak: dict[int, int | None] = {}
    for f in sample_frames:
        rgb = frame_cache.get(f)
        if rgb is None:
            continue
        per_obj = masks_per_frame.get(f, {})
        for oid in (0, 1):
            mask = per_obj.get(oid)
            if mask is None or mask.sum() == 0:
                # Mask already empty (collapse guard or natural loss) — skip.
                if streak.get(oid) is not None:
                    streak[oid] = None
                continue
            if f in reseed_frames:
                # Don't second-guess true reseed frames.
                streak[oid] = None
                continue
            ok_hand = _mp_finds_hand_in_mask(mp_image, mask, rgb)
            if not ok_hand:
                s = streak.get(oid)
                if s is None:
                    streak[oid] = f  # start of a lost run
            else:
                if streak.get(oid) is not None:
                    lost_runs.setdefault(oid, []).append((streak[oid], f))
                    streak[oid] = None
    # Close out runs still open at the end
    for oid, s in streak.items():
        if s is not None:
            lost_runs.setdefault(oid, []).append((s, sample_frames[-1] + 1))
    # Require minimum run length
    out: dict[int, list[tuple[int, int]]] = {}
    for oid, runs in lost_runs.items():
        for start, end in runs:
            # Number of sampled checks in [start, end)
            n_checks = sum(1 for f in sample_frames if start <= f < end)
            if n_checks >= LOST_TRACK_MIN_LOST_RUN:
                out.setdefault(oid, []).append((start, end))
    return out


def redetect_lost_tracks(
    masks_per_frame: dict[int, dict[int, np.ndarray]],
    video_path: Path,
    sam2_video: SAM2VideoPredictor,
    state,
    sam3_proc, sam3_model,
    mp_image: mp_vision.HandLandmarker,
    reseed_frames: set[int],
    image_size: tuple[int, int],
    device: str,
    prompt: str = SAM3_PROMPT,
    score_threshold: float = SAM3_SCORE_THRESHOLD,
    mask_threshold: float = SAM3_MASK_THRESHOLD,
) -> dict:
    """Detect lost-track runs, re-seed via SAM 3 at the run-start frame, and
    re-propagate SAM 2 from there. Returns a log dict for the JSON.

    SAM 3 must run OUTSIDE the bfloat16 autocast (it returns BF16 tensors that
    .cpu().numpy() can't materialize). SAM 2 ops (`add_new_mask`,
    `propagate_in_video`) are wrapped in autocast separately.
    """
    W, H = image_size
    log = {"checked": True, "lost_segments": [], "redetections": []}
    lost_runs = find_lost_track_segments(masks_per_frame, video_path, mp_image, reseed_frames)
    if not lost_runs:
        return log
    interventions: list[tuple[int, int]] = []
    for oid, runs in lost_runs.items():
        for start, end in runs:
            interventions.append((start, oid))
            log["lost_segments"].append({"obj_id": int(oid), "start": int(start), "end": int(end)})
    interventions.sort()

    # Pre-read each intervention frame
    needed = sorted({s for s, _ in interventions})
    frame_rgb_at: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    fidx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fidx in needed:
            frame_rgb_at[fidx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fidx += 1
    cap.release()

    # Stage 1: SAM 3 re-detect (no autocast)
    new_seeds: list[tuple[int, int, np.ndarray]] = []  # (frame, obj_id, trimmed_mask)
    for f, lost_oid in interventions:
        rgb = frame_rgb_at.get(f)
        if rgb is None:
            continue
        boxes, scores, masks = run_sam3(
            sam3_proc, sam3_model, Image.fromarray(rgb), device,
            prompt=prompt, score_threshold=score_threshold, mask_threshold=mask_threshold,
        )
        if len(boxes) == 0:
            log["redetections"].append({"frame": int(f), "obj_id": int(lost_oid), "outcome": "sam3_empty"})
            continue
        other_oid = 1 - lost_oid
        other_mask = masks_per_frame.get(f, {}).get(other_oid)
        best_j = None
        best_score = -1.0
        for j in range(len(boxes)):
            cand = masks[j].astype(bool)
            if other_mask is not None and other_mask.sum() > 0:
                inter = int((cand & other_mask).sum())
                cand_area = int(cand.sum())
                if cand_area > 0 and inter / cand_area > 0.4:
                    continue
            if scores[j] > best_score:
                best_score = float(scores[j])
                best_j = j
        if best_j is None:
            log["redetections"].append({"frame": int(f), "obj_id": int(lost_oid), "outcome": "no_complement"})
            continue
        trimmed = trim_mask_at_wrist(masks[best_j].astype(bool), rgb, mp_image)
        if trimmed.sum() == 0:
            trimmed = masks[best_j].astype(bool)
        new_seeds.append((int(f), int(lost_oid), trimmed))
        log["redetections"].append({
            "frame": int(f), "obj_id": int(lost_oid),
            "outcome": "reseeded", "sam3_score": best_score,
        })

    if not new_seeds:
        return log

    # Stage 2: SAM 2 add_new_mask + re-propagate (wrap in autocast)
    earliest_reseed = min(s[0] for s in new_seeds)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for f, oid, mask in new_seeds:
            sam2_video.add_new_mask(inference_state=state, frame_idx=f, obj_id=oid, mask=mask)
            masks_per_frame.setdefault(f, {})[oid] = mask

        n_overridden = 0
        for fidx, obj_ids, mask_logits in sam2_video.propagate_in_video(
            state, start_frame_idx=earliest_reseed
        ):
            if fidx in reseed_frames:
                continue
            per_obj = {}
            for oid, logit in zip(obj_ids, mask_logits):
                m = (logit > 0).cpu().numpy()
                if m.ndim == 3:
                    m = m[0]
                per_obj[int(oid)] = m.astype(bool)
            masks_per_frame[int(fidx)] = per_obj
            n_overridden += 1
    log["repropagated_frames"] = n_overridden
    return log


def resolve_mask_collapse(
    masks_per_frame: dict[int, dict[int, np.ndarray]],
    reseed_frames: set[int],
    image_size: tuple[int, int] | None = None,
) -> dict:
    """Detect runs of frames where two obj_ids' masks overlap or have near-identical
    centroids, and zero the smaller of the two until the next reseed.

    Collapse trigger: at least MASK_COLLAPSE_MIN_RUN consecutive frames where
    IoU(obj0, obj1) >= MASK_COLLAPSE_IOU OR
    centroid_distance(obj0, obj1) < MASK_COLLAPSE_CENTROID_PX_FRAC * image_diagonal.
    The second condition catches "two masks on different parts of the same hand"
    that don't quite overlap but are clearly the same anatomy.
    """
    frames = sorted(masks_per_frame.keys())
    run_start: int | None = None
    collapsed_runs: list[dict] = []
    fixes = 0
    if image_size is not None:
        W, H = image_size
        centroid_thresh = MASK_COLLAPSE_CENTROID_PX_FRAC * float(np.hypot(W, H))
    else:
        centroid_thresh = 0.0
    for fidx in frames:
        per_obj = masks_per_frame[fidx]
        if len(per_obj) < 2:
            run_start = None
            continue
        oids = sorted(per_obj.keys())
        m0, m1 = per_obj[oids[0]], per_obj[oids[1]]
        if m0 is None or m1 is None or m0.sum() == 0 or m1.sum() == 0:
            run_start = None
            continue
        inter = int((m0 & m1).sum())
        union = int((m0 | m1).sum())
        iou = inter / union if union > 0 else 0.0
        # Centroid distance
        ys0, xs0 = np.where(m0); ys1, xs1 = np.where(m1)
        c0 = (xs0.mean(), ys0.mean())
        c1 = (xs1.mean(), ys1.mean())
        cd = float(np.hypot(c0[0] - c1[0], c0[1] - c1[1]))
        collapsed = (iou >= MASK_COLLAPSE_IOU) or (centroid_thresh > 0 and cd < centroid_thresh)
        if collapsed:
            if run_start is None:
                run_start = fidx
        else:
            if run_start is not None and fidx - run_start >= MASK_COLLAPSE_MIN_RUN:
                # Finalize the previous collapse run and remove the smaller-mask obj.
                collapsed_runs.append({"start": run_start, "end": fidx - 1})
                for f in range(run_start, fidx):
                    if f in reseed_frames:
                        continue  # never touch seed frames
                    fpo = masks_per_frame.get(f, {})
                    a0 = int(fpo[oids[0]].sum()) if oids[0] in fpo and fpo[oids[0]] is not None else 0
                    a1 = int(fpo[oids[1]].sum()) if oids[1] in fpo and fpo[oids[1]] is not None else 0
                    smaller = oids[0] if a0 <= a1 else oids[1]
                    if smaller in fpo and fpo[smaller] is not None:
                        fpo[smaller] = np.zeros_like(fpo[smaller])
                        fixes += 1
            run_start = None
    # If a collapse run extends to the end, also finalize it
    if run_start is not None and frames and frames[-1] - run_start + 1 >= MASK_COLLAPSE_MIN_RUN:
        collapsed_runs.append({"start": run_start, "end": frames[-1]})
        for f in range(run_start, frames[-1] + 1):
            if f in reseed_frames:
                continue
            fpo = masks_per_frame.get(f, {})
            oids2 = sorted(fpo.keys())
            if len(oids2) < 2:
                continue
            a0 = int(fpo[oids2[0]].sum()) if fpo[oids2[0]] is not None else 0
            a1 = int(fpo[oids2[1]].sum()) if fpo[oids2[1]] is not None else 0
            smaller = oids2[0] if a0 <= a1 else oids2[1]
            if fpo.get(smaller) is not None:
                fpo[smaller] = np.zeros_like(fpo[smaller])
                fixes += 1
    return {"collapsed_runs": collapsed_runs, "frames_fixed": fixes}


def _mask_centroid_area(mask: np.ndarray | None) -> tuple[tuple[float, float] | None, int]:
    if mask is None or mask.sum() == 0:
        return None, 0
    ys, xs = np.where(mask)
    return (float(xs.mean()), float(ys.mean())), int(len(xs))


def detect_full_collapse(
    masks_per_frame: dict[int, dict[int, np.ndarray]],
    bbox_iou_thresh: float = FULL_COLLAPSE_IOU,
    mask_iou_thresh: float = FULL_COLLAPSE_IOU,
    min_frames: int = FULL_COLLAPSE_MIN_FRAMES,
) -> dict:
    """Detect frames where obj_0 and obj_1 collapsed onto the same blob.

    Trigger: at least `min_frames` frames where both bbox IoU and mask IoU
    are >= the respective thresholds. Such collapses indicate the other track
    was lost entirely and cannot be recovered by in-pass fixes, so the caller
    should re-track the whole video with a different SAM 3 prompt.
    """
    hit_frames: list[int] = []
    for fidx in sorted(masks_per_frame.keys()):
        per_obj = masks_per_frame[fidx]
        if len(per_obj) < 2:
            continue
        oids = sorted(per_obj.keys())
        m0, m1 = per_obj[oids[0]], per_obj[oids[1]]
        if m0 is None or m1 is None or m0.sum() == 0 or m1.sum() == 0:
            continue
        inter = int((m0 & m1).sum())
        union = int((m0 | m1).sum())
        mask_iou = inter / union if union > 0 else 0.0
        b0 = mask_bbox(m0)
        b1 = mask_bbox(m1)
        if b0 is None or b1 is None:
            continue
        bbox_iou = _iou(b0, b1)
        if bbox_iou >= bbox_iou_thresh and mask_iou >= mask_iou_thresh:
            hit_frames.append(int(fidx))
    return {
        "collapse_frames": hit_frames,
        "n_collapse_frames": len(hit_frames),
        "triggered": len(hit_frames) >= min_frames,
        "bbox_iou_thresh": bbox_iou_thresh,
        "mask_iou_thresh": mask_iou_thresh,
        "min_frames": min_frames,
    }


def filter_mask_spikes(
    masks_per_frame: dict[int, dict[int, np.ndarray]],
    image_size: tuple[int, int],
    reseed_frames: set[int],
    centroid_jump_frac: float = MASK_SPIKE_CENTROID_JUMP_FRAC,
    area_ratio: float = MASK_SPIKE_AREA_RATIO,
    max_run: int = MASK_SPIKE_MAX_RUN,
) -> dict:
    """Detect short-run mask anomalies (centroid jump or area inflation that
    recovers within `max_run` frames) and overwrite the anomalous frames with
    the pre-spike mask.

    For each obj_id: a spike is triggered at frame f if the centroid jumped
    > centroid_jump_frac * diag from f-1, OR the area changed by more than
    `area_ratio` from f-1 (inflated to > 2x f-1 OR shrunk to < f-1 / 2).
    Recovery checks only the dimension(s) that triggered the spike — so an
    area-only spike (mask inflates to entire arm and then back, rgb_21) does
    NOT require the centroid to return to f-1's position; the hand may have
    moved during the spike. The spike closes at the first frame f+k <= f+max_run
    that satisfies the relevant recovery condition.

    Reseed frames CAN be overwritten: a SAM 3 seed that lands far from where
    SAM 2 was tracking is itself a spike (rgb_09 frame 50). The forward-lookup
    still stops at the *next* reseed frame, since those represent independent
    anchor points.
    """
    W, H = image_size
    diag = float(np.hypot(W, H))
    jump_thresh = centroid_jump_frac * diag
    frames = sorted(masks_per_frame.keys())
    log: dict = {"obj": {}, "total_replacements": 0}
    for oid in (0, 1):
        per_frame_ca = [
            (f, *_mask_centroid_area(masks_per_frame[f].get(oid))) for f in frames
        ]
        replaced_runs: list[dict] = []
        i = 1
        while i < len(per_frame_ca):
            f, c, a = per_frame_ca[i]
            f_prev, c_prev, a_prev = per_frame_ca[i - 1]
            if c is None or c_prev is None or a_prev <= 0:
                i += 1
                continue
            jumped = float(np.hypot(c[0] - c_prev[0], c[1] - c_prev[1])) > jump_thresh
            inflated = a > a_prev * area_ratio
            shrunk = a_prev > a * area_ratio
            if not (jumped or inflated or shrunk):
                i += 1
                continue
            # If area changed, the centroid may NOT return (hand moved during
            # the spike), so don't require centroid recovery in that case.
            require_centroid_back = jumped and not (inflated or shrunk)
            require_area_back = inflated or shrunk
            recovered_at: int | None = None
            for k in range(1, max_run + 1):
                j = i + k
                if j >= len(per_frame_ca):
                    break
                fj, cj, aj = per_frame_ca[j]
                if fj in reseed_frames:
                    break
                if cj is None:
                    continue
                ok_c = (not require_centroid_back) or (
                    float(np.hypot(cj[0] - c_prev[0], cj[1] - c_prev[1])) <= jump_thresh
                )
                ok_a = (not require_area_back) or (
                    aj <= a_prev * area_ratio and a_prev <= aj * area_ratio
                )
                if ok_c and ok_a:
                    recovered_at = j
                    break
            if recovered_at is None:
                i += 1
                continue
            mask_prev = masks_per_frame[f_prev].get(oid)
            if mask_prev is None or mask_prev.sum() == 0:
                i += 1
                continue
            other_oid = 1 - oid
            mask_prev_area = int(mask_prev.sum())
            spike_frames: list[int] = []
            skipped_frames: list[int] = []
            for k in range(i, recovered_at):
                fk = per_frame_ca[k][0]
                # Collision avoidance: refuse to replace if the replacement
                # mask overlaps the other obj_id's current mask at fk by >=
                # MASK_SPIKE_COLLISION_IOU. Without this, "smoothing" obj_1
                # back to its prior-frame side can land it on top of obj_0
                # (rgb_03 ~26s).
                other = masks_per_frame.get(fk, {}).get(other_oid)
                if other is not None and other.sum() > 0:
                    inter = int((mask_prev & other).sum())
                    union = mask_prev_area + int(other.sum()) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou >= MASK_SPIKE_COLLISION_IOU:
                        skipped_frames.append(int(fk))
                        continue
                masks_per_frame.setdefault(fk, {})[oid] = mask_prev.copy()
                spike_frames.append(int(fk))
                log["total_replacements"] += 1
            if spike_frames or skipped_frames:
                entry = {}
                if spike_frames:
                    entry["start"] = spike_frames[0]
                    entry["end"] = spike_frames[-1]
                    entry["frames"] = spike_frames
                if skipped_frames:
                    entry["skipped_collision"] = skipped_frames
                replaced_runs.append(entry)
            i = recovered_at + 1
        log["obj"][str(oid)] = replaced_runs
    return log


def filter_cross_person_hands(
    masks_per_frame: dict[int, dict[int, np.ndarray]],
    image_size: tuple[int, int],
    top_frac: float = CROSS_PERSON_TOP_FRAC,
    min_run: int = CROSS_PERSON_MIN_RUN,
) -> dict:
    """Zero masks for any obj_id that sits with its centroid in the top
    `top_frac` of the frame for at least `min_run` consecutive frames.

    For ego-centric clips the wearer's hands rarely stay in the top of the
    frame (the wearer's body is below/behind the camera). A sustained mask
    in the top region is almost always SAM 2 drift onto a co-worker reaching
    across the workspace. Marking those frames empty hides the wrong bbox
    in the visualization rather than perpetuating the bad track.
    """
    W, H = image_size
    top_y_cutoff = top_frac * H
    log = {"obj": {}, "frames_zeroed": 0}
    if not masks_per_frame:
        return log
    frames = sorted(masks_per_frame.keys())
    for oid in (0, 1):
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for f in frames:
            m = masks_per_frame[f].get(oid)
            if m is None or m.sum() == 0:
                if run_start is not None:
                    runs.append((run_start, f - 1))
                    run_start = None
                continue
            ys, _ = np.where(m)
            cy = float(ys.mean())
            if cy < top_y_cutoff:
                if run_start is None:
                    run_start = f
            else:
                if run_start is not None:
                    runs.append((run_start, f - 1))
                    run_start = None
        if run_start is not None:
            runs.append((run_start, frames[-1]))
        zeroed_runs: list[dict] = []
        for start, end in runs:
            if end - start + 1 < min_run:
                continue
            for f in range(start, end + 1):
                mf = masks_per_frame.get(f, {})
                if oid in mf and mf[oid] is not None and mf[oid].sum() > 0:
                    mf[oid] = np.zeros_like(mf[oid])
                    log["frames_zeroed"] += 1
            zeroed_runs.append({"start": int(start), "end": int(end)})
        log["obj"][str(oid)] = zeroed_runs
    return log


def fix_seed_label_swaps(
    masks_per_frame: dict[int, dict[int, np.ndarray]],
    reseed_data: dict[int, list],
    margin_px: float = SEED_LABEL_SWAP_MARGIN_PX,
) -> dict:
    """At each seed frame sf, if the seed masks for obj_0/obj_1 are closer to
    the OTHER obj_id's mask at sf-1 than to their own, swap labels at sf.

    Targets the Hungarian-at-the-seed misassignment when the wearer's hands
    moved enough between seeds that the prior-seed boxes mislead the matcher
    (rgb_03 f350: prior seed at f300 had both hands on the right side; new
    seed at f350 had one LEFT and one RIGHT candidate; Hungarian matched
    obj_0 to RIGHT and obj_1 to LEFT, opposite of the actual f349 identities).

    Only swaps the seed frame itself. SAM 2 typically reverts to the prior
    identity at sf+1 via memory_attention, so the 1-frame swap is enough to
    restore continuity.
    """
    seed_frames = sorted(reseed_data.keys())
    swap_log: list[dict] = []
    for sf in seed_frames:
        prev_f = sf - 1
        if prev_f not in masks_per_frame or sf not in masks_per_frame:
            continue
        mp = masks_per_frame[prev_f]
        ms = masks_per_frame[sf]
        if 0 not in mp or 1 not in mp or 0 not in ms or 1 not in ms:
            continue
        cp0, _ = _mask_centroid_area(mp[0])
        cp1, _ = _mask_centroid_area(mp[1])
        cs0, _ = _mask_centroid_area(ms[0])
        cs1, _ = _mask_centroid_area(ms[1])
        if any(c is None for c in (cp0, cp1, cs0, cs1)):
            continue
        d_no = float(np.hypot(cs0[0]-cp0[0], cs0[1]-cp0[1])) + float(np.hypot(cs1[0]-cp1[0], cs1[1]-cp1[1]))
        d_sw = float(np.hypot(cs0[0]-cp1[0], cs0[1]-cp1[1])) + float(np.hypot(cs1[0]-cp0[0], cs1[1]-cp0[1]))
        if d_sw + margin_px < d_no:
            ms[0], ms[1] = ms[1], ms[0]
            swap_log.append({
                "frame": int(sf),
                "d_no_swap": d_no,
                "d_swap": d_sw,
            })
    return {"swapped_frames": [s["frame"] for s in swap_log], "swap_log": swap_log}


def trim_mask_at_wrist(
    mask: np.ndarray, frame_rgb: np.ndarray, mp_image: mp_vision.HandLandmarker
) -> np.ndarray:
    """Trim a hand+forearm mask to just the hand portion, using MP wrist landmark.

    Crops a square around the mask, runs MP HandLandmarker IMAGE mode. If MP finds
    a hand, computes a "palm direction" from the wrist (lm 0) to the middle-finger
    MCP (lm 9) and keeps only mask pixels on the palm side of the cut line through
    the wrist perpendicular to that direction. If MP fails or the trim would leave
    too little mask, the original mask is returned unchanged.
    """
    if mask.sum() == 0:
        return mask
    H, W = mask.shape[:2]
    # Mask bbox
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    sx1, sy1, sx2, sy2 = expand_to_square_crop_xyxy(bbox, W, H, WRIST_TRIM_CROP_EXPAND)
    if sx2 <= sx1 or sy2 <= sy1:
        return mask
    crop = np.ascontiguousarray(frame_rgb[sy1:sy2, sx1:sx2])
    if crop.size == 0:
        return mask
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = mp_image.detect(mp_img)
    if not result.hand_landmarks:
        return mask
    lms = result.hand_landmarks[0]
    crop_h, crop_w = crop.shape[:2]
    wrist = np.array([lms[0].x * crop_w + sx1, lms[0].y * crop_h + sy1], dtype=np.float64)
    mid_mcp = np.array([lms[9].x * crop_w + sx1, lms[9].y * crop_h + sy1], dtype=np.float64)
    palm_dir = mid_mcp - wrist
    norm = float(np.linalg.norm(palm_dir))
    if norm < 1e-3:
        return mask
    palm_dir /= norm
    # Cut line passes through wrist, perpendicular to palm_dir.
    # Push the cut a bit further toward the palm so we don't clip the wrist itself.
    diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    wrist_shifted = wrist + palm_dir * (WRIST_TRIM_MARGIN_FRAC * diag)
    # For each mask pixel, project onto palm_dir; keep if projection from wrist_shifted is >= 0
    rel_x = xs - wrist_shifted[0]
    rel_y = ys - wrist_shifted[1]
    proj = rel_x * palm_dir[0] + rel_y * palm_dir[1]
    keep_mask_idx = proj >= 0.0
    n_keep = int(keep_mask_idx.sum())
    n_total = int(len(xs))
    if n_total == 0 or n_keep / n_total < WRIST_TRIM_MIN_RETAIN_FRAC:
        return mask  # sanity: skip trim if it removes too much
    out = np.zeros_like(mask)
    out[ys[keep_mask_idx], xs[keep_mask_idx]] = True
    return out


@torch.no_grad()
def run_sam3(
    processor, model, image: Image.Image, device: str,
    prompt: str = SAM3_PROMPT,
    score_threshold: float = SAM3_SCORE_THRESHOLD,
    mask_threshold: float = SAM3_MASK_THRESHOLD,
):
    """Return (boxes_xyxy, scores, masks_bool_HW) for hands at this frame."""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    boxes_t = res["boxes"]
    scores_t = res["scores"]
    masks_t = res["masks"]
    if len(boxes_t) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, image.height, image.width), dtype=bool),
        )
    boxes = boxes_t.cpu().numpy() if hasattr(boxes_t, "cpu") else np.asarray(boxes_t)
    scores = scores_t.cpu().numpy() if hasattr(scores_t, "cpu") else np.asarray(scores_t)
    masks = masks_t.cpu().numpy() if hasattr(masks_t, "cpu") else np.asarray(masks_t)
    if masks.ndim == 4:
        masks = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    masks = masks.astype(bool)
    # NMS to dedupe overlapping masks SAM 3 occasionally emits per concept
    keep = nms(boxes, scores, SAM3_NMS_IOU)
    boxes = boxes[keep]
    scores = scores[keep]
    masks = masks[keep]
    # Drop anything plausibly torso-sized
    idx = indices_under_max_area(boxes, (image.width, image.height))
    boxes = boxes[idx]
    scores = scores[idx]
    masks = masks[idx]
    # Keep top-2 by score
    if len(boxes) > MAX_HANDS:
        top = np.argsort(-scores)[:MAX_HANDS]
        boxes = boxes[top]
        scores = scores[top]
        masks = masks[top]
    # Sort by x-center for stable obj_id assignment downstream
    if len(boxes) > 0:
        order = np.argsort((boxes[:, 0] + boxes[:, 2]) / 2.0)
        boxes = boxes[order]
        scores = scores[order]
        masks = masks[order]
    return boxes, scores, masks


def reseed_pair_frames(
    num_frames: int, fps: float, interval_sec: float, pair_offset_frames: int
) -> list[tuple[int, int | None]]:
    """Pairs of seed frames: (first, second) per RESEED_INTERVAL_SEC; second=None if past end."""
    step = max(1, int(round(fps * interval_sec)))
    out: list[tuple[int, int | None]] = []
    for start in range(0, num_frames, step):
        second = start + pair_offset_frames
        out.append((start, second if second < num_frames else None))
    return out


def flatten_pairs(pairs: list[tuple[int, int | None]]) -> list[int]:
    out: list[int] = []
    for a, b in pairs:
        out.append(a)
        if b is not None:
            out.append(b)
    return out


def _centroid(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def assign_obj_ids_by_match(
    new_boxes: np.ndarray,
    last_box_per_obj: dict[int, np.ndarray],
    max_hands: int = MAX_HANDS,
) -> list[tuple[int, np.ndarray]]:
    """Map each new detection box to an obj_id while preserving identity across seeds.

    First seed (last_box_per_obj is empty): sort new boxes by x-center, assign
    obj_id = 0 (leftmost), 1 (rightmost), ...
    Subsequent seeds: solve the optimal pairwise centroid-distance assignment
    (scipy linear_sum_assignment / Hungarian). This avoids identity swaps when
    both hands move a lot between seeds and a single shortest edge would
    otherwise force the other hand onto a far-away box. Unmatched new boxes get
    the lowest free obj_id (up to max_hands).
    """
    if len(new_boxes) == 0:
        return []
    if not last_box_per_obj:
        order = np.argsort([(_centroid(b)[0]) for b in new_boxes])
        out = []
        for new_oid, idx in enumerate(order[:max_hands]):
            out.append((int(new_oid), new_boxes[int(idx)]))
        return out

    existing_ids = sorted(last_box_per_obj.keys())
    existing_centroids = np.array([_centroid(last_box_per_obj[i]) for i in existing_ids])
    new_centroids = np.array([_centroid(b) for b in new_boxes])
    # pairwise distance: rows = existing obj_ids, cols = new boxes
    dist = np.linalg.norm(existing_centroids[:, None, :] - new_centroids[None, :, :], axis=2)

    # Optimal assignment via Hungarian — minimizes the SUM of paired distances,
    # so it preserves identity through symmetric motion that would fool a greedy
    # picker.
    row_ind, col_ind = linear_sum_assignment(dist)
    assignments: dict[int, int] = {}
    used_new: set[int] = set()
    for i, j in zip(row_ind, col_ind):
        assignments[existing_ids[i]] = int(j)
        used_new.add(int(j))

    # Unmatched new boxes -> next free obj_id (could happen if more new boxes than existing)
    free_ids = [i for i in range(max_hands) if i not in last_box_per_obj and i not in assignments]
    for j in range(len(new_boxes)):
        if j in used_new:
            continue
        if not free_ids:
            break
        new_oid = free_ids.pop(0)
        assignments[new_oid] = j
    return [(int(oid), new_boxes[j]) for oid, j in sorted(assignments.items())]


def extract_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_mask_rle(mask: np.ndarray) -> dict:
    """Encode a binary HxW mask as a COCO-RLE-style dict with ASCII counts."""
    fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = coco_mask.encode(fortran)
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def mask_bbox(mask: np.ndarray) -> list[float] | None:
    """Bounding box of the LARGEST connected component of `mask`.

    Using the largest CC instead of all mask pixels prevents stray pixels far
    from the main hand blob from inflating the bbox (rgb_34 had random bbox
    jumps to ~2x size because SAM 2 occasionally emits tiny disconnected blobs).
    """
    if mask.sum() == 0:
        return None
    m_u8 = mask.astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    if num <= 1:
        ys, xs = np.where(mask)
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    x, y, w, h, _ = stats[biggest]
    return [float(x), float(y), float(x + w - 1), float(y + h - 1)]


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    """Return a new mask containing only the largest connected component.

    SAM 2 occasionally emits tiny stray CCs far from the main hand blob
    (e.g., a single-pixel remnant of a SAM 3 seed that was supposed to be
    replaced). Those strays don't affect mask area materially but stretch
    the rendered bbox to enclose them (the renderer uses np.where over all
    mask pixels). Cleaning to the largest CC keeps mask/bbox/centroid
    derivations all consistent.
    """
    if mask is None or mask.sum() == 0:
        return mask
    m_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    if num <= 2:
        return mask  # single CC (num=2 = background + one CC), nothing to drop
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == biggest)


def cleanup_masks_largest_cc(masks_per_frame: dict[int, dict[int, np.ndarray]]) -> dict:
    """Replace every mask in masks_per_frame with its largest connected
    component. Returns a small log with how many frames / masks were cleaned."""
    cleaned = 0
    for fidx, per_obj in masks_per_frame.items():
        for oid, m in list(per_obj.items()):
            if m is None or m.sum() == 0:
                continue
            new_m = keep_largest_cc(m)
            if new_m is not m and int(new_m.sum()) != int(m.sum()):
                per_obj[oid] = new_m
                cleaned += 1
    return {"masks_cleaned": cleaned}


def overlay_masks(
    frame_bgr: np.ndarray,
    masks_by_obj: dict[int, np.ndarray],
    frame_idx: int | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    H, W = out.shape[:2]
    for obj_id, mask in masks_by_obj.items():
        if mask is None or mask.sum() == 0:
            continue
        color = np.array(MASK_COLORS_BGR[obj_id % len(MASK_COLORS_BGR)], dtype=np.uint8)
        out[mask] = (0.45 * out[mask] + 0.55 * color).astype(np.uint8)
        # Bbox from the LARGEST connected component, matching the JSON bbox.
        bbox = mask_bbox(mask)
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        col = tuple(int(c) for c in color)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"hand {obj_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, max(th, y1 - 4)), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    # Frame number in the top-right corner for easy identification when
    # iterating on tracker outputs.
    if frame_idx is not None:
        text = f"f{frame_idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
        x = W - tw - 12
        y = th + 12
        cv2.rectangle(out, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(out, text, (x, y), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def truncate_video_to_frames(src: Path, max_frames: int, dst: Path) -> Path:
    """Re-encode the first `max_frames` frames of `src` into `dst` via ffmpeg.

    Keeps the source's FPS / dimensions; uses high-quality libx264 (-crf 18)
    and strips audio. The caller is responsible for cleaning up `dst`.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-frames:v", str(max_frames),
        "-c:v", "libx264", "-crf", "18", "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def decode_video_to_jpegs(video_path: Path, out_dir: Path) -> int:
    """Use ffmpeg to write `<out_dir>/00000.jpg`, `00001.jpg`, ...

    SAM 2's `init_state(video_path=<folder>)` expects this exact naming
    (zero-padded frame index starting at 0). Returns the number of frames
    written. `-q:v 2` is high-quality JPEG, comparable to the model input
    after its internal resize.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-q:v", "2", "-start_number", "0",
        str(out_dir / "%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    n = len(list(out_dir.glob("*.jpg")))
    if n == 0:
        raise RuntimeError(f"ffmpeg produced no frames for {video_path}")
    return n


def pick_next_version_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = base / f"track_v{n}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        n += 1


def _run_tracking_pass(
    video_path: Path,
    sam3_proc, sam3_model,
    sam2_video: SAM2VideoPredictor,
    mp_image: mp_vision.HandLandmarker,
    fps: float, num_frames: int, width: int, height: int,
    device: str, reseed_sec: float,
    prompt: str,
    score_threshold: float = SAM3_SCORE_THRESHOLD,
    mask_threshold: float = SAM3_MASK_THRESHOLD,
) -> dict:
    """One full forward+backward+lost-track+collapse-resolve tracking pass
    using `prompt` as the SAM 3 text prompt.

    Returns a dict with masks_per_frame and all per-pass log structures. The
    SAM 2 inference state is initialized and reset inside this call.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    pairs = reseed_pair_frames(num_frames, fps, reseed_sec, PAIR_OFFSET_FRAMES)
    flat_seeds = flatten_pairs(pairs)

    reseed_data: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    reseed_log: list[dict] = []
    last_box_per_obj: dict[int, np.ndarray] = {}
    wrist_trim_log: list[dict] = []
    seed_verification_log: list[dict] = []
    for sidx in flat_seeds:
        frame_rgb = extract_frame(cap, sidx)
        if frame_rgb is None:
            continue
        boxes, scores, masks = run_sam3(
            sam3_proc, sam3_model, Image.fromarray(frame_rgb), device,
            prompt=prompt, score_threshold=score_threshold, mask_threshold=mask_threshold,
        )
        if len(boxes) > 0:
            keep_idx = []
            for j in range(len(boxes)):
                ok, reason = seed_is_hand_like(masks[j].astype(bool), boxes[j], frame_rgb, mp_image)
                seed_verification_log.append({
                    "frame": sidx, "candidate": j,
                    "score": float(scores[j]),
                    "passes": ok, "reason": reason,
                })
                if ok:
                    keep_idx.append(j)
            if len(keep_idx) < len(boxes):
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
                masks = masks[keep_idx]
        if len(boxes) == 0:
            assignments = []
        else:
            assignments = assign_obj_ids_by_match(boxes, last_box_per_obj, MAX_HANDS)
        entries: list[tuple[int, np.ndarray, np.ndarray]] = []
        box_index_for_oid: dict[int, int] = {}
        for oid, b in assignments:
            arr = np.asarray(b, dtype=np.float32)
            for j in range(len(boxes)):
                if np.allclose(boxes[j], arr):
                    box_index_for_oid[int(oid)] = j
                    break
        for oid, j in box_index_for_oid.items():
            raw_mask = masks[j].astype(bool)
            trimmed = trim_mask_at_wrist(raw_mask, frame_rgb, mp_image)
            trimmed_area = int(trimmed.sum())
            raw_area = int(raw_mask.sum())
            wrist_trim_log.append({
                "frame": sidx,
                "obj_id": int(oid),
                "raw_mask_px": raw_area,
                "trimmed_mask_px": trimmed_area,
                "trim_frac": (1.0 - trimmed_area / raw_area) if raw_area > 0 else 0.0,
            })
            if trimmed_area > 0:
                ys, xs = np.where(trimmed)
                new_bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            else:
                new_bbox = boxes[j].astype(np.float32)
            entries.append((int(oid), new_bbox, trimmed))
        if entries:
            reseed_data[sidx] = entries
            for oid, b, _ in entries:
                last_box_per_obj[oid] = b
        reseed_log.append({
            "frame": sidx,
            "n_boxes": int(len(boxes)),
            "scores": [float(s) for s in scores.tolist()],
            "boxes": [[float(x) for x in b.tolist()] for b in boxes],
            "obj_ids": [oid for oid, _, _ in entries],
        })
    cap.release()

    if not reseed_data:
        return {
            "status": "no_seeds",
            "prompt": prompt,
            "pairs": pairs,
            "reseeds": reseed_log,
            "masks_per_frame": {},
            "reseed_data": reseed_data,
            "backward_overrides": [],
            "wrist_trim": wrist_trim_log,
            "seed_verification": seed_verification_log,
            "lost_track_redetection": {"checked": False, "lost_segments": [], "redetections": []},
        }

    # All SAM 2 video ops must run under bfloat16 autocast; otherwise certain
    # videos trigger a "mat1 BFloat16 / mat2 Float" mismatch in memory_attention.
    # Long videos OOM the 4090 if frames + state both stay GPU-resident; offload
    # past length thresholds (slower per-frame, but doesn't crash). For very
    # long videos we also pre-decode to JPEG and let SAM 2 stream frames, since
    # the mp4 code path tries to pack all frames into one tensor upfront.
    offload_video = num_frames > OFFLOAD_VIDEO_FRAME_THRESHOLD
    offload_state = num_frames > OFFLOAD_STATE_FRAME_THRESHOLD
    use_jpeg_stream = num_frames > JPEG_STREAMING_FRAME_THRESHOLD
    jpeg_tmp_dir: Path | None = None
    if use_jpeg_stream:
        jpeg_tmp_dir = Path(tempfile.mkdtemp(prefix=f"sam2_{video_path.stem}_", dir="/tmp"))
        decode_video_to_jpegs(video_path, jpeg_tmp_dir)
        init_video_path = str(jpeg_tmp_dir)
    else:
        init_video_path = str(video_path)
    backward_overrides_log: list[dict] = []
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = sam2_video.init_state(
                video_path=init_video_path,
                offload_video_to_cpu=offload_video,
                offload_state_to_cpu=offload_state,
                async_loading_frames=use_jpeg_stream,
            )
            for frame_idx, entries in reseed_data.items():
                for obj_id, _box, mask in entries:
                    sam2_video.add_new_mask(
                        inference_state=state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=mask,
                    )

            # Forward propagate through the whole video.
            masks_per_frame: dict[int, dict[int, np.ndarray]] = {}
            for fidx, obj_ids, mask_logits in sam2_video.propagate_in_video(state):
                per_obj: dict[int, np.ndarray] = {}
                for oid, logit in zip(obj_ids, mask_logits):
                    m = (logit > 0).cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    per_obj[int(oid)] = m.astype(bool)
                masks_per_frame[int(fidx)] = per_obj

            # Backward-propagate to cover gaps between consecutive valid seeds,
            # including the leading gap (0 -> first valid seed).
            seed_frames = sorted(f for f in reseed_data.keys())
            backprop_pairs: list[tuple[int, int]] = []
            if seed_frames and seed_frames[0] > 0:
                backprop_pairs.append((0, seed_frames[0]))
            for i in range(len(seed_frames) - 1):
                backprop_pairs.append((seed_frames[i], seed_frames[i + 1]))
            for a, b in backprop_pairs:
                n_track = b - a + 1
                n_overridden = 0
                for fidx, obj_ids, mask_logits in sam2_video.propagate_in_video(
                    state, start_frame_idx=b, max_frame_num_to_track=n_track, reverse=True,
                ):
                    if fidx >= b or fidx < a:
                        continue
                    if fidx in reseed_data:
                        continue
                    per_obj: dict[int, np.ndarray] = {}
                    for oid, logit in zip(obj_ids, mask_logits):
                        m = (logit > 0).cpu().numpy()
                        if m.ndim == 3:
                            m = m[0]
                        per_obj[int(oid)] = m.astype(bool)
                    masks_per_frame.setdefault(int(fidx), {}).update(per_obj)
                    n_overridden += 1
                backward_overrides_log.append({"pair": [a, b], "frames_overridden": n_overridden})

        # Lost-track redetection runs SAM 3 outside autocast and SAM 2 inside.
        lost_track_report = redetect_lost_tracks(
            masks_per_frame, video_path, sam2_video, state,
            sam3_proc, sam3_model, mp_image,
            reseed_frames=set(reseed_data.keys()),
            image_size=(width, height),
            device=device,
            prompt=prompt,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )

        sam2_video.reset_state(state)
        del state
    finally:
        if jpeg_tmp_dir is not None:
            shutil.rmtree(jpeg_tmp_dir, ignore_errors=True)

    # NOTE: resolve_mask_collapse is intentionally NOT run here. It zeroes the
    # smaller of two overlapping masks, which would hide a full-screen collapse
    # from detect_full_collapse and prevent the fallback-prompt retry.
    # track_one_video runs collapse resolution only on the CHOSEN pass.
    return {
        "status": "ok",
        "prompt": prompt,
        "pairs": pairs,
        "reseeds": reseed_log,
        "masks_per_frame": masks_per_frame,
        "reseed_data": reseed_data,
        "backward_overrides": backward_overrides_log,
        "wrist_trim": wrist_trim_log,
        "seed_verification": seed_verification_log,
        "lost_track_redetection": lost_track_report,
    }


def track_one_video(
    video_path: Path,
    out_video_path: Path,
    out_meta_path: Path,
    out_frames_path: Path,
    sam3_proc,
    sam3_model,
    sam2_video: SAM2VideoPredictor,
    mp_image: mp_vision.HandLandmarker,
    device: str,
    reseed_sec: float,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Pass 1: primary prompt
    primary = _run_tracking_pass(
        video_path, sam3_proc, sam3_model, sam2_video, mp_image,
        fps, num_frames, width, height, device, reseed_sec,
        prompt=SAM3_PROMPT,
    )

    # Detect full-screen collapse (both obj_ids on the same blob) and, if
    # triggered, retry the whole video with FALLBACK_SAM3_PROMPT. Keep the
    # pass with the fewer collapse frames.
    if primary["status"] == "no_seeds":
        primary_collapse = {"triggered": False, "n_collapse_frames": 0, "collapse_frames": []}
    else:
        primary_collapse = detect_full_collapse(primary["masks_per_frame"])
    fallback: dict | None = None
    fallback_collapse: dict | None = None
    chosen = primary
    used_pass = "primary"
    if primary_collapse.get("triggered"):
        fallback = _run_tracking_pass(
            video_path, sam3_proc, sam3_model, sam2_video, mp_image,
            fps, num_frames, width, height, device, reseed_sec,
            prompt=FALLBACK_SAM3_PROMPT,
            score_threshold=FALLBACK_SAM3_SCORE_THRESHOLD,
            mask_threshold=FALLBACK_SAM3_MASK_THRESHOLD,
        )
        if fallback["status"] == "no_seeds":
            fallback_collapse = {"triggered": False, "n_collapse_frames": 0, "collapse_frames": []}
        else:
            fallback_collapse = detect_full_collapse(fallback["masks_per_frame"])
        if (fallback["status"] == "ok"
                and fallback_collapse["n_collapse_frames"] < primary_collapse["n_collapse_frames"]):
            chosen = fallback
            used_pass = "fallback"

    if chosen["status"] == "no_seeds":
        return {
            "video": video_path.name,
            "fps": fps,
            "num_frames": num_frames,
            "size": [width, height],
            "pairs": [[a, b] for a, b in chosen["pairs"]],
            "reseeds": chosen["reseeds"],
            "status": "no_seeds",
            "output_video": None,
            "sam3_prompt": chosen["prompt"],
            "used_pass": used_pass,
            "full_collapse_check": {
                "primary": primary_collapse,
                "fallback": fallback_collapse,
            },
        }

    masks_per_frame = chosen["masks_per_frame"]
    reseed_data = chosen["reseed_data"]

    # Clean masks to their largest connected component. SAM 2 occasionally
    # emits tiny stray CCs far from the main hand blob, which inflate the
    # rendered bbox even though the JSON bbox (which uses largest CC) looks
    # fine. Run this BEFORE the collapse / spike passes so they operate on
    # clean masks too.
    mask_cleanup_report = cleanup_masks_largest_cc(masks_per_frame)

    # NOTE: filter_cross_person_hands is intentionally disabled here.
    # An earlier attempt to zero masks whose centroid sat in the top 20% of
    # the frame for 5+ frames over-pruned legitimate wearer-hand frames in
    # several videos. Keep the function defined for future tuning.
    cross_person_report = {"disabled": True}

    # Fix Hungarian-induced 1-frame label swaps at seed frames before the
    # spike filter runs (so the spike filter sees a consistent trajectory).
    seed_label_swap_report = fix_seed_label_swaps(masks_per_frame, reseed_data)

    # Resolve in-pass mask collapses (smaller mask zeroed). Runs AFTER
    # detect_full_collapse + fallback retry so that full collapses still get a
    # chance to trigger the fallback path.
    collapse_report = resolve_mask_collapse(
        masks_per_frame, set(reseed_data.keys()), image_size=(width, height)
    )

    # Smooth short-run mask spikes (rgb_09 cross-screen jump etc).
    spike_report = filter_mask_spikes(
        masks_per_frame,
        image_size=(width, height),
        reseed_frames=set(reseed_data.keys()),
    )

    # Write output video with overlays + per-frame RLE masks JSON.
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
    frame_records: list[dict] = []
    fidx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        masks = masks_per_frame.get(fidx, {})
        overlay = overlay_masks(bgr, masks, frame_idx=fidx)
        if fidx in reseed_data:
            cv2.putText(overlay, "SEED", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, "SEED", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(overlay)
        hands_entry = []
        for obj_id, m in sorted(masks.items()):
            if m is None or m.sum() == 0:
                hands_entry.append({"obj_id": int(obj_id), "bbox": None, "mask_rle": None})
                continue
            hands_entry.append({
                "obj_id": int(obj_id),
                "bbox": mask_bbox(m),
                "mask_rle": encode_mask_rle(m),
            })
        frame_records.append({"frame": fidx, "hands": hands_entry})
        fidx += 1
    writer.release()
    cap.release()
    out_frames_path.write_text(json.dumps({
        "video": video_path.name,
        "size": [width, height],
        "fps": fps,
        "frames": frame_records,
    }))

    meta = {
        "video": video_path.name,
        "fps": fps,
        "num_frames": num_frames,
        "size": [width, height],
        "reseed_interval_sec": reseed_sec,
        "pair_offset_frames": PAIR_OFFSET_FRAMES,
        "pairs": [[a, b] for a, b in chosen["pairs"]],
        "sam3_prompt": chosen["prompt"],
        "used_pass": used_pass,
        "full_collapse_check": {
            "primary": primary_collapse,
            "fallback": fallback_collapse,
        },
        "reseeds": chosen["reseeds"],
        "backward_overrides": chosen["backward_overrides"],
        "wrist_trim": chosen["wrist_trim"],
        "seed_verification": chosen["seed_verification"],
        "lost_track_redetection": chosen["lost_track_redetection"],
        "mask_cleanup_largest_cc": mask_cleanup_report,
        "cross_person_filter": cross_person_report,
        "seed_label_swap_fix": seed_label_swap_report,
        "mask_collapse_resolution": collapse_report,
        "mask_spike_filter": spike_report,
        "tracked_frames": len(masks_per_frame),
        "output_video": str(out_video_path.name),
    }
    out_meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", nargs="*", default=None, help="Video paths (default: all data/rgb_*.mp4).")
    ap.add_argument("--reseed-sec", type=float, default=RESEED_INTERVAL_SEC)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output-base", default="outputs", help="Parent dir; subfolder track_v<N> auto-versioned.")
    ap.add_argument("--reverse", action="store_true", help="Process videos in reverse order.")
    ap.add_argument(
        "--max-sec", type=float, default=MAX_TRACK_SEC_DEFAULT,
        help="Truncate each video to this many seconds before tracking "
             "(0 or negative disables). SAM 2's frame loader keeps all loaded "
             "frames in RAM, so beyond ~2000 frames (~66s) the process OOMs.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = pick_next_version_dir(root / args.output_base)
    print(f"Output dir: {out_dir}")

    print(f"Loading {SAM3_MODEL_ID} + {SAM2_VIDEO_MODEL_ID} + MediaPipe HandLandmarker on {args.device} ...")
    sam3_proc, sam3_model = load_sam3(args.device)
    sam2_video = SAM2VideoPredictor.from_pretrained(SAM2_VIDEO_MODEL_ID, device=args.device)
    sam2_video.eval()
    mp_image = load_mp_image()
    print("Models loaded.")

    videos = (
        [Path(v) if Path(v).is_absolute() else (root / v) for v in args.videos]
        if args.videos
        else sorted((root / "data").glob("*.mp4"))
    )
    if args.reverse:
        videos = list(reversed(videos))
    print(f"Videos to process: {len(videos)} (order: {'reverse' if args.reverse else 'forward'})")

    max_sec = args.max_sec if args.max_sec and args.max_sec > 0 else None
    truncate_tmp_dir: Path | None = None
    if max_sec is not None:
        truncate_tmp_dir = Path(tempfile.mkdtemp(prefix="track_truncated_", dir="/tmp"))
        print(f"Per-video cap: {max_sec:.1f}s (longer videos truncated into {truncate_tmp_dir})")

    summary = []
    t0 = time.time()
    try:
        for i, vp in enumerate(videos, 1):
            if not vp.exists():
                print(f"  [{i}/{len(videos)}] {vp.name}: MISSING, skip")
                continue
            tic = time.time()
            # Pre-truncate any video over the cap into a temp mp4; track_one_video
            # always sees a clip <= max_sec seconds long.
            effective_path = vp
            if max_sec is not None:
                cap_probe = cv2.VideoCapture(str(vp))
                src_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
                src_nframes = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_probe.release()
                src_dur = src_nframes / src_fps if src_fps > 0 else 0
                if src_dur > max_sec:
                    cap_frames = int(max_sec * src_fps)
                    truncated = truncate_tmp_dir / vp.name
                    truncate_video_to_frames(vp, cap_frames, truncated)
                    effective_path = truncated
                    print(f"  [{i}/{len(videos)}] {vp.name}: truncated {src_dur:.1f}s -> {max_sec:.1f}s ({cap_frames} frames)")
            out_video = out_dir / f"{vp.stem}_track.mp4"
            out_meta = out_dir / f"{vp.stem}_track.json"
            out_frames = out_dir / f"{vp.stem}_track.frames.json"
            try:
                meta = track_one_video(
                    effective_path, out_video, out_meta, out_frames,
                    sam3_proc, sam3_model, sam2_video, mp_image, args.device, args.reseed_sec
                )
                # Stamp original-video info on the meta so callers know it was truncated.
                if effective_path != vp:
                    meta["truncated_from_video"] = vp.name
                    meta["truncated_to_sec"] = max_sec
                summary.append(meta)
                print(
                    f"  [{i}/{len(videos)}] {vp.name}: "
                    f"{len(meta.get('reseeds', []))} reseed pts, "
                    f"{meta.get('tracked_frames', 0)} tracked frames, "
                    f"{time.time()-tic:.1f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"  [{i}/{len(videos)}] {vp.name}: ERROR {e!r}", flush=True)
                summary.append({"video": vp.name, "error": repr(e)})
            finally:
                # Drop truncated mp4 immediately (avoid disk + RAM accumulation).
                if effective_path != vp:
                    effective_path.unlink(missing_ok=True)
                # Force release of any lingering inference state + cached
                # frame tensors before moving on. Without this, AsyncVideoFrameLoader
                # tensors from earlier videos can linger long enough to OOM-kill
                # the process around video ~18 of a 60s-per-video run.
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nDone. {len(summary)} videos processed in {time.time()-t0:.1f}s. Output: {out_dir}")
    finally:
        if truncate_tmp_dir is not None:
            shutil.rmtree(truncate_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

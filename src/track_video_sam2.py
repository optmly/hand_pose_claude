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
import json
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
SAM3_SCORE_THRESHOLD = 0.50
SAM3_MASK_THRESHOLD = 0.50
SAM3_NMS_IOU = 0.50  # SAM 3 occasionally emits overlapping duplicate masks for the same hand
MAX_HAND_BBOX_AREA_FRAC = 0.20
MAX_HANDS = 2  # wearer has at most two hands

RESEED_INTERVAL_SEC = 5.0
PAIR_OFFSET_FRAMES = 50          # second seed of each pair is this many frames after the first
MASK_COLORS_BGR = [(255, 80, 0), (0, 80, 255)]  # obj 0 = blue-ish, obj 1 = red-ish (BGR)

# Mask-collapse guard: when two obj_ids' masks share too much IoU OR their
# centroids are too close for too long, both are likely locked on the same hand.
# Zero out the smaller mask until the next reseed splits them apart.
MASK_COLLAPSE_IOU = 0.30                  # IoU >= this counts as overlapping
MASK_COLLAPSE_CENTROID_PX_FRAC = 0.05     # OR centroid distance < this * image diagonal
MASK_COLLAPSE_MIN_RUN = 3                 # require this many consecutive overlapping frames

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
def run_sam3(processor, model, image: Image.Image, device: str):
    """Return (boxes_xyxy, scores, masks_bool_HW) for hands at this frame."""
    inputs = processor(images=image, text=SAM3_PROMPT, return_tensors="pt").to(device)
    outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs,
        threshold=SAM3_SCORE_THRESHOLD,
        mask_threshold=SAM3_MASK_THRESHOLD,
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
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def overlay_masks(frame_bgr: np.ndarray, masks_by_obj: dict[int, np.ndarray]) -> np.ndarray:
    out = frame_bgr.copy()
    for obj_id, mask in masks_by_obj.items():
        if mask is None or mask.sum() == 0:
            continue
        color = np.array(MASK_COLORS_BGR[obj_id % len(MASK_COLORS_BGR)], dtype=np.uint8)
        out[mask] = (0.45 * out[mask] + 0.55 * color).astype(np.uint8)
        # bbox from mask + label
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        col = tuple(int(c) for c in color)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"hand {obj_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.7, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, max(th, y1 - 4)), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def pick_next_version_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        candidate = base / f"track_v{n}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        n += 1


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

    # PAIRED reseeds at (start, start+PAIR_OFFSET_FRAMES) every reseed_sec.
    pairs = reseed_pair_frames(num_frames, fps, reseed_sec, PAIR_OFFSET_FRAMES)
    flat_seeds = flatten_pairs(pairs)

    # 1) Run SAM 3 at each seed frame; keep masks alongside boxes/scores.
    # reseed_data[sidx] = [(obj_id, box, mask), ...]
    reseed_data: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    reseed_log: list[dict] = []
    last_box_per_obj: dict[int, np.ndarray] = {}
    wrist_trim_log: list[dict] = []
    seed_verification_log: list[dict] = []
    for sidx in flat_seeds:
        frame_rgb = extract_frame(cap, sidx)
        if frame_rgb is None:
            continue
        boxes, scores, masks = run_sam3(sam3_proc, sam3_model, Image.fromarray(frame_rgb), device)
        # Verify each SAM 3 candidate looks like a hand (MP or shape).
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
            # Recompute bbox from the (possibly trimmed) mask so the bbox follows the cut
            if trimmed_area > 0:
                ys, xs = np.where(trimmed)
                new_bbox = np.array(
                    [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32
                )
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
            "video": video_path.name,
            "fps": fps,
            "num_frames": num_frames,
            "size": [width, height],
            "pairs": [[a, b] for a, b in pairs],
            "reseeds": reseed_log,
            "status": "no_seeds",
            "output_video": None,
        }

    # 2) Initialize SAM 2 video state and add prompts at every seed frame.
    # All SAM 2 video ops must run under bfloat16 autocast; otherwise certain
    # videos trigger a "mat1 BFloat16 / mat2 Float" mismatch in memory_attention.
    backward_overrides_log: list[dict] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = sam2_video.init_state(
            video_path=str(video_path),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )
        for frame_idx, entries in reseed_data.items():
            for obj_id, _box, mask in entries:
                # Feed the SAM 3 mask straight into SAM 2 so propagation starts
                # anchored to the actual hand rather than re-segmenting a bbox.
                sam2_video.add_new_mask(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )

        # 3) Forward propagate through the whole video.
        masks_per_frame: dict[int, dict[int, np.ndarray]] = {}
        for fidx, obj_ids, mask_logits in sam2_video.propagate_in_video(state):
            per_obj: dict[int, np.ndarray] = {}
            for oid, logit in zip(obj_ids, mask_logits):
                m = (logit > 0).cpu().numpy()
                if m.ndim == 3:
                    m = m[0]
                per_obj[int(oid)] = m.astype(bool)
            masks_per_frame[int(fidx)] = per_obj

        # 4) Backward-propagate to cover gaps:
        #    - between every pair of consecutive valid seeds (frames where SAM 3 detected
        #      something), and
        #    - from the FIRST valid seed all the way back to frame 0 if SAM 3 missed
        #      the start of the video (e.g. rgb_06 had 0 detections at f=0 but 2 at f=50).
        # Override the masks in those gap frames so SAM 2 starts the gap anchored at the
        # later good seed rather than relying on forward propagation that may have
        # nothing to track at the very beginning.
        seed_frames = sorted(f for f in reseed_data.keys())
        backprop_pairs: list[tuple[int, int]] = []
        if seed_frames and seed_frames[0] > 0:
            backprop_pairs.append((0, seed_frames[0]))  # leading gap
        for i in range(len(seed_frames) - 1):
            backprop_pairs.append((seed_frames[i], seed_frames[i + 1]))
        for a, b in backprop_pairs:
            n_track = b - a + 1
            n_overridden = 0
            for fidx, obj_ids, mask_logits in sam2_video.propagate_in_video(
                state,
                start_frame_idx=b,
                max_frame_num_to_track=n_track,
                reverse=True,
            ):
                if fidx >= b or fidx < a:
                    continue
                if fidx in reseed_data:
                    continue  # don't override actual seed frames
                per_obj: dict[int, np.ndarray] = {}
                for oid, logit in zip(obj_ids, mask_logits):
                    m = (logit > 0).cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    per_obj[int(oid)] = m.astype(bool)
                masks_per_frame.setdefault(int(fidx), {}).update(per_obj)
                n_overridden += 1
            backward_overrides_log.append({
                "pair": [a, b],
                "frames_overridden": n_overridden,
            })

    # 5) Resolve mask collapses (both obj_ids stuck on the same hand).
    collapse_report = resolve_mask_collapse(
        masks_per_frame, set(reseed_data.keys()), image_size=(width, height)
    )

    # 4) Write output video with overlays + per-frame RLE masks JSON.
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
        overlay = overlay_masks(bgr, masks)
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

    # Free GPU memory before next video
    sam2_video.reset_state(state)
    del state

    meta = {
        "video": video_path.name,
        "fps": fps,
        "num_frames": num_frames,
        "size": [width, height],
        "reseed_interval_sec": reseed_sec,
        "pair_offset_frames": PAIR_OFFSET_FRAMES,
        "pairs": [[a, b] for a, b in pairs],
        "reseeds": reseed_log,
        "backward_overrides": backward_overrides_log,
        "wrist_trim": wrist_trim_log,
        "seed_verification": seed_verification_log,
        "mask_collapse_resolution": collapse_report,
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

    summary = []
    t0 = time.time()
    for i, vp in enumerate(videos, 1):
        if not vp.exists():
            print(f"  [{i}/{len(videos)}] {vp.name}: MISSING, skip")
            continue
        out_video = out_dir / f"{vp.stem}_track.mp4"
        out_meta = out_dir / f"{vp.stem}_track.json"
        out_frames = out_dir / f"{vp.stem}_track.frames.json"
        tic = time.time()
        try:
            meta = track_one_video(
                vp, out_video, out_meta, out_frames,
                sam3_proc, sam3_model, sam2_video, mp_image, args.device, args.reseed_sec
            )
            summary.append(meta)
            print(
                f"  [{i}/{len(videos)}] {vp.name}: "
                f"{len(meta.get('reseeds', []))} reseed pts, "
                f"{meta.get('tracked_frames', 0)} tracked frames, "
                f"{time.time()-tic:.1f}s"
            )
        except Exception as e:
            print(f"  [{i}/{len(videos)}] {vp.name}: ERROR {e!r}")
            summary.append({"video": vp.name, "error": repr(e)})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {len(summary)} videos processed in {time.time()-t0:.1f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()

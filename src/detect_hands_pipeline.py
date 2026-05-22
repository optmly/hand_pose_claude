"""GD-count-gated cascade for ego-centric hand detection, with SAM 2 masks + MP keypoints.

For each frame:
  1. Run Grounding DINO with prompt "a hand." + NMS + max-area filter (no torso).
  2. If GD returns EXACTLY 2 boxes -> use GD boxes; ask SAM 2 for masks (PRIMARY path).
  3. If GD returns <2 boxes -> SAM 3 with prompt "hand from above", top-2 by score (FALLBACK).
  4. If GD returns >=3 boxes -> SAM 3 with prompt "hand from above", top-2 by score (FILTER).

For every kept hand:
  - The bbox is expanded by 20%, padded to a square, and the crop is fed to
    MediaPipe HandLandmarker; the 21 keypoints are mapped back to image coords.
  - Heuristic wrist (mask-edge median) is kept as a fallback when MP returns nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam3Model,
    Sam3Processor,
)

GD_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM3_MODEL_ID = "facebook/sam3"
SAM2_MODEL_ID = "facebook/sam2-hiera-base-plus"

GD_PROMPT = "a hand."
GD_BOX_THRESHOLD = 0.30
GD_TEXT_THRESHOLD = 0.25
GD_NMS_IOU = 0.50  # boxes with IoU above this are deduplicated (keep higher score)

SAM3_PROMPT = "hand from above"
SAM3_SCORE_THRESHOLD = 0.50
SAM3_MASK_THRESHOLD = 0.50

MAX_HANDS = 2  # wearer has at most two hands; final result is capped at top-2 by score
MAX_HAND_BBOX_AREA_FRAC = 0.20  # bbox larger than this is not a hand (likely torso/body)
WRIST_STRIP_PX = 16  # pixel-strip width for sampling near the image-border edge of the mask
WRIST_MARKER_COLOR = (255, 255, 0)  # bright yellow, drawn last so it's visible
WRIST_MARKER_RADIUS = 14

# MediaPipe HandLandmarker
MP_MODEL_PATHS = [
    Path("/home/jingjin/.cache/mediapipe/hand_landmarker.task"),
    Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task",
]
MP_BBOX_EXPAND_FRAC = 0.20    # 20% expansion before squaring
MP_HAND_DET_CONF = 0.30       # MP min hand detection confidence inside the crop
MP_KP_COLOR = (255, 0, 255)   # magenta keypoints
MP_KP_RADIUS = 6
MP_SKELETON_COLOR = (255, 0, 255)
MP_WRIST_COLOR = (0, 255, 255)  # cyan, draws over yellow heuristic wrist when MP is present

# MediaPipe hand-skeleton connections (21 landmarks: 0=wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

PATH_GD_COLOR = (0, 220, 0)
PATH_SAM3_COLOR = (255, 128, 0)
PATH_SAM3_FILTER_COLOR = (200, 70, 200)


def _find_mp_model() -> Path:
    for p in MP_MODEL_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(
        "hand_landmarker.task not found in any known location. "
        "Download from https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )


def load_mp_landmarker() -> mp_vision.HandLandmarker:
    base = mp_python.BaseOptions(model_asset_path=str(_find_mp_model()))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,  # we crop one hand per call, so MP only needs to find one
        min_hand_detection_confidence=MP_HAND_DET_CONF,
        min_hand_presence_confidence=MP_HAND_DET_CONF,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def load_models(device: str):
    gd_proc = AutoProcessor.from_pretrained(GD_MODEL_ID)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL_ID).to(device).eval()
    sam3_proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    sam3_model = Sam3Model.from_pretrained(SAM3_MODEL_ID, device_map=device).eval()
    sam2_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL_ID, device=device)
    mp_landmarker = load_mp_landmarker()
    return gd_proc, gd_model, sam3_proc, sam3_model, sam2_predictor, mp_landmarker


def expand_to_square_crop(
    bbox: np.ndarray | tuple, image_w: int, image_h: int, expand_frac: float = MP_BBOX_EXPAND_FRAC
) -> tuple[int, int, int, int]:
    """Expand bbox by `expand_frac` on each side, square it around its center, clip to image."""
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + expand_frac)
    half = side / 2.0
    sx1 = max(0, int(round(cx - half)))
    sy1 = max(0, int(round(cy - half)))
    sx2 = min(image_w, int(round(cx + half)))
    sy2 = min(image_h, int(round(cy + half)))
    return sx1, sy1, sx2, sy2


def run_mediapipe_on_bbox(
    landmarker: mp_vision.HandLandmarker, image_np: np.ndarray, bbox: np.ndarray | tuple
) -> tuple[list[tuple[float, float]] | None, str | None, float | None]:
    """Crop image to expanded square around bbox, run MP, return landmarks in original image coords.

    Returns (landmarks_xy, handedness, score) or (None, None, None) if MP finds nothing.
    """
    H, W = image_np.shape[:2]
    sx1, sy1, sx2, sy2 = expand_to_square_crop(bbox, W, H)
    if sx2 <= sx1 or sy2 <= sy1:
        return None, None, None
    crop = image_np[sy1:sy2, sx1:sx2]
    if crop.size == 0:
        return None, None, None
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None, None, None
    landmarks = result.hand_landmarks[0]
    crop_h, crop_w = crop.shape[:2]
    pts = [(float(lm.x) * crop_w + sx1, float(lm.y) * crop_h + sy1) for lm in landmarks]
    handedness = None
    score = None
    if result.handedness:
        cat = result.handedness[0][0]
        handedness = cat.category_name
        score = float(cat.score)
    return pts, handedness, score


@torch.no_grad()
def sam2_masks_from_boxes(predictor: SAM2ImagePredictor, image_np: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Predict one binary mask per input bbox using SAM 2.

    `boxes` is (N, 4) xyxy. Returns (N, H, W) bool.
    """
    if len(boxes) == 0:
        return np.empty((0, *image_np.shape[:2]), dtype=bool)
    predictor.set_image(image_np)
    out = []
    for box in boxes:
        masks, _scores, _logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.asarray(box, dtype=np.float32),
            multimask_output=False,
        )
        # masks is (1, H, W) (numpy bool/float depending on version) -> squeeze + binarize
        m = np.asarray(masks).reshape(-1, masks.shape[-2], masks.shape[-1])[0]
        out.append(m > 0.5 if m.dtype != bool else m)
    return np.stack(out, axis=0)


def find_wrist_point(mask: np.ndarray) -> tuple[int, int]:
    """Approximate wrist pixel from a hand mask.

    Heuristic: the wrist sits on the side of the mask nearest the image border the
    forearm exits through. We find that border, take a thin strip of mask pixels
    along that side, and return the strip's median pixel.
    Returns (-1, -1) for empty masks.
    """
    if mask.sum() == 0:
        return (-1, -1)
    H, W = mask.shape
    ys, xs = np.where(mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    dists = {
        "top": y_min,
        "bottom": H - 1 - y_max,
        "left": x_min,
        "right": W - 1 - x_max,
    }
    side = min(dists, key=dists.get)
    s = WRIST_STRIP_PX
    if side == "top":
        sub = mask[y_min : min(H, y_min + s)]
        cols = np.where(sub.any(axis=0))[0]
        return (int(np.median(cols)), y_min) if len(cols) else (-1, -1)
    if side == "bottom":
        sub = mask[max(0, y_max - s + 1) : y_max + 1]
        cols = np.where(sub.any(axis=0))[0]
        return (int(np.median(cols)), y_max) if len(cols) else (-1, -1)
    if side == "left":
        sub = mask[:, x_min : min(W, x_min + s)]
        rows = np.where(sub.any(axis=1))[0]
        return (x_min, int(np.median(rows))) if len(rows) else (-1, -1)
    sub = mask[:, max(0, x_max - s + 1) : x_max + 1]
    rows = np.where(sub.any(axis=1))[0]
    return (x_max, int(np.median(rows))) if len(rows) else (-1, -1)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) < iou_thresh for j in keep):
            keep.append(int(i))
    return keep


def top_k_by_score(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k entries by descending score."""
    if len(scores) <= k:
        return np.argsort(-scores)
    return np.argsort(-scores)[:k]


def indices_under_max_area(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Return indices of boxes whose area fraction <= MAX_HAND_BBOX_AREA_FRAC.

    Drops detections so large they cannot plausibly be a single hand (e.g. torso).
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    W, H = image_size
    image_area = float(W) * float(H)
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    areas = widths * heights
    return np.where(areas / image_area <= MAX_HAND_BBOX_AREA_FRAC)[0]


@torch.no_grad()
def run_grounding_dino(processor, model, image: Image.Image, device: str) -> dict:
    inputs = processor(images=image, text=GD_PROMPT, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=GD_BOX_THRESHOLD,
        text_threshold=GD_TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    if len(boxes) > 0:
        keep = nms(boxes, scores, GD_NMS_IOU)
        boxes = boxes[keep]
        scores = scores[keep]
    return {"boxes": boxes, "scores": scores}


@torch.no_grad()
def run_sam3(processor, model, image: Image.Image, device: str) -> dict:
    inputs = processor(images=image, text=SAM3_PROMPT, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=SAM3_SCORE_THRESHOLD,
        mask_threshold=SAM3_MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results["masks"]
    boxes = results["boxes"]
    scores = results["scores"]
    return {
        "masks": masks.cpu().numpy().astype(bool) if hasattr(masks, "cpu") else np.asarray(masks).astype(bool),
        "boxes": boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes),
        "scores": scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores),
    }


def detect_hands_cascade(
    image: Image.Image,
    image_np: np.ndarray,
    gd_proc,
    gd_model,
    sam3_proc,
    sam3_model,
    sam2_predictor: SAM2ImagePredictor,
    mp_landmarker: mp_vision.HandLandmarker,
    device: str,
) -> dict:
    image_size = (image.width, image.height)
    gd = run_grounding_dino(gd_proc, gd_model, image, device)
    n_gd_raw = int(len(gd["boxes"]))
    # Drop GD boxes that are too large to be a single hand (torso/body false positives).
    gd_keep = indices_under_max_area(gd["boxes"], image_size)
    gd_boxes = gd["boxes"][gd_keep]
    gd_scores = gd["scores"][gd_keep]
    n_gd = int(len(gd_boxes))

    # GD == 2 after area filter: use those boxes; ask SAM 2 for masks.
    if n_gd == 2:
        masks = sam2_masks_from_boxes(sam2_predictor, image_np, gd_boxes)
        result = {
            "source": "grounding_dino",
            "boxes": gd_boxes,
            "scores": gd_scores,
            "masks": masks,
            "gd_raw_count": n_gd_raw,
            "gd_kept_count": n_gd,
        }
    else:
        # SAM 3 is the authoritative source on its own scores.
        sam3 = run_sam3(sam3_proc, sam3_model, image, device)
        n_sam3_raw = int(len(sam3["boxes"]))
        # Same area filter on SAM 3 detections.
        s_keep = indices_under_max_area(sam3["boxes"], image_size)
        s_boxes = sam3["boxes"][s_keep]
        s_scores = sam3["scores"][s_keep]
        s_masks = sam3["masks"][s_keep] if sam3["masks"] is not None and len(sam3["masks"]) > 0 else None
        n_sam3 = int(len(s_boxes))
        if n_sam3 == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            masks = None
        else:
            idx = top_k_by_score(s_scores, MAX_HANDS)
            boxes = s_boxes[idx]
            scores = s_scores[idx]
            masks = s_masks[idx] if s_masks is not None else None
        source = "sam3_filter" if n_gd >= 3 else "sam3_fallback"
        result = {
            "source": source,
            "boxes": boxes,
            "scores": scores,
            "masks": masks,
            "gd_raw_count": n_gd_raw,
            "gd_kept_count": n_gd,
            "sam3_raw_count": n_sam3_raw,
            "sam3_kept_count": n_sam3,
        }

    # Wrist point per kept hand (only if we have a mask for it).
    wrists: list[tuple[int, int]] = []
    if result["masks"] is not None:
        m = result["masks"]
        if m.ndim == 4:
            m = m.reshape(-1, m.shape[-2], m.shape[-1])
        for mi in m:
            wrists.append(find_wrist_point(np.asarray(mi, dtype=bool)))
    result["wrists"] = wrists

    # MediaPipe keypoints per kept hand: crop = expand(20%) + square around the GD/SAM bbox.
    mp_landmarks: list[list[tuple[float, float]] | None] = []
    mp_handedness: list[str | None] = []
    mp_scores: list[float | None] = []
    mp_crops: list[tuple[int, int, int, int]] = []
    for box in result["boxes"]:
        crop_bbox = expand_to_square_crop(box, image.width, image.height)
        mp_crops.append(crop_bbox)
        pts, hand, sc = run_mediapipe_on_bbox(mp_landmarker, image_np, box)
        mp_landmarks.append(pts)
        mp_handedness.append(hand)
        mp_scores.append(sc)
    result["mp_landmarks"] = mp_landmarks
    result["mp_handedness"] = mp_handedness
    result["mp_scores"] = mp_scores
    result["mp_crops"] = mp_crops
    return result


def _draw_box(img: np.ndarray, box, color, label: str | None = None) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, max(th + 2, y1 - 4)), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def render_result(frame_rgb: np.ndarray, result: dict) -> np.ndarray:
    img = frame_rgb.copy()
    src = result["source"]
    if src == "grounding_dino":
        color = PATH_GD_COLOR
    elif src == "sam3_filter":
        color = PATH_SAM3_FILTER_COLOR
    else:
        color = PATH_SAM3_COLOR
    masks = result.get("masks")
    if masks is not None:
        if masks.ndim == 4:
            masks = masks[0] if masks.shape[0] == 1 else masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        elif masks.ndim < 3:
            masks = masks[None]
        for i, m in enumerate(masks):
            if m.sum() == 0:
                continue
            tint = np.array([(i * 67) % 256, (i * 113 + 80) % 256, (i * 197 + 160) % 256], dtype=np.uint8)
            img[m] = (0.45 * img[m] + 0.55 * tint).astype(np.uint8)
    for box, score in zip(result["boxes"], result["scores"]):
        _draw_box(img, box, color, f"hand {float(score):.2f}")
    # MediaPipe crop bbox (the expanded square that was fed to MP) — thin dashed-ish gray outline
    for crop in result.get("mp_crops", []):
        cx1, cy1, cx2, cy2 = crop
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (150, 150, 150), 1, lineType=cv2.LINE_AA)
    # Wrist points (heuristic) — yellow disc with black ring; will be visually superseded by MP wrist when present
    for wx, wy in result.get("wrists", []):
        if wx < 0 or wy < 0:
            continue
        cv2.circle(img, (wx, wy), WRIST_MARKER_RADIUS, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, (wx, wy), WRIST_MARKER_RADIUS - 2, WRIST_MARKER_COLOR, thickness=-1, lineType=cv2.LINE_AA)
    # MediaPipe skeleton + landmarks (drawn on top so they're visible)
    mp_landmarks = result.get("mp_landmarks", [])
    mp_handedness = result.get("mp_handedness", [])
    for pts, hand in zip(mp_landmarks, mp_handedness):
        if pts is None:
            continue
        ipts = [(int(round(x)), int(round(y))) for x, y in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, ipts[a], ipts[b], MP_SKELETON_COLOR, 2, lineType=cv2.LINE_AA)
        for i, (px, py) in enumerate(ipts):
            r = 8 if i == 0 else MP_KP_RADIUS
            cv2.circle(img, (px, py), r + 2, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            col = MP_WRIST_COLOR if i == 0 else MP_KP_COLOR
            cv2.circle(img, (px, py), r, col, thickness=-1, lineType=cv2.LINE_AA)
        if hand:
            label_pt = (ipts[0][0] + 12, ipts[0][1] - 8)
            cv2.putText(img, hand, label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, hand, label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, MP_WRIST_COLOR, 2, cv2.LINE_AA)
    return img


def make_panel(frame_rgb: np.ndarray, result: dict) -> np.ndarray:
    res_img = render_result(frame_rgb, result)
    h, w = frame_rgb.shape[:2]
    bar = 60
    canvas = np.full((h + bar, w * 2 + 10, 3), 255, dtype=np.uint8)
    canvas[bar:, 0:w] = frame_rgb
    canvas[bar:, w + 10 : 2 * w + 10] = res_img

    cv2.putText(canvas, f"original  ({w}x{h})", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    src = result["source"]
    gd_raw = result.get("gd_raw_count", 0)
    if src == "grounding_dino":
        title = f"Grounding DINO  (GD=2  -> {len(result['boxes'])} hands)"
        color = PATH_GD_COLOR
    elif src == "sam3_filter":
        s_raw = result.get("sam3_raw_count", len(result["boxes"]))
        title = f"SAM 3 filter  (GD={gd_raw}>=3; SAM3 raw {s_raw} -> top-{len(result['boxes'])} by score)"
        color = PATH_SAM3_FILTER_COLOR
    else:
        s_raw = result.get("sam3_raw_count", len(result["boxes"]))
        title = f"SAM 3 fallback  (GD={gd_raw}<2; SAM3 raw {s_raw} -> top-{len(result['boxes'])} by score)"
        color = PATH_SAM3_COLOR
    cv2.putText(canvas, title, (w + 18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    return canvas


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help='Video paths (default: all data/rgb_*.mp4 when --random; else a small sample).',
    )
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 60, 150, 250])
    ap.add_argument(
        "--random",
        action="store_true",
        help="Sample ONE random frame per video instead of using --frames.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for --random sampling.")
    ap.add_argument("--output", default="outputs/detect_hands_pipeline")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _resolve_videos(args, root: Path) -> list[Path]:
    if args.videos:
        return [Path(v) if Path(v).is_absolute() else (root / v) for v in args.videos]
    if args.random:
        return sorted((root / "data").glob("*.mp4"))
    return [
        root / p
        for p in ["data/rgb_01.mp4", "data/rgb_05.mp4", "data/rgb_10.mp4", "data/rgb_20.mp4", "data/rgb_30.mp4"]
    ]


def _frames_for_video(args, vp: Path, rng: random.Random) -> list[int]:
    if not args.random:
        return list(args.frames)
    try:
        n = video_frame_count(vp)
    except RuntimeError as e:
        print(f"  {vp.stem}: {e}")
        return []
    if n <= 1:
        return [0]
    return [rng.randrange(0, n)]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    videos = _resolve_videos(args, root)
    print(f"Device: {args.device}")
    print(f"Mode: {'random-one-per-video (seed=' + str(args.seed) + ')' if args.random else 'fixed frames'}")
    print(f"Videos: {len(videos)}")
    print(f"Loading models ({GD_MODEL_ID}, {SAM3_MODEL_ID}, {SAM2_MODEL_ID}, MediaPipe HandLandmarker) ...")
    gd_proc, gd_model, sam3_proc, sam3_model, sam2_predictor, mp_landmarker = load_models(args.device)
    print("Models loaded.")

    summary: list[dict] = []
    counts = {"grounding_dino": 0, "sam3_fallback": 0, "sam3_filter": 0}

    for vp in videos:
        if not vp.exists():
            print(f"  skip (missing): {vp}")
            continue
        frames = _frames_for_video(args, vp, rng)
        for idx in frames:
            try:
                frame = extract_frame(vp, idx)
            except RuntimeError as e:
                print(f"  {vp.stem} frame {idx}: {e}")
                continue
            result = detect_hands_cascade(
                Image.fromarray(frame),
                frame,
                gd_proc,
                gd_model,
                sam3_proc,
                sam3_model,
                sam2_predictor,
                mp_landmarker,
                args.device,
            )
            counts[result["source"]] += 1
            panel = make_panel(frame, result)
            out_path = out_dir / f"{vp.stem}_f{idx:05d}_{result['source']}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            entry = {
                "video": vp.name,
                "frame": idx,
                "source": result["source"],
                "n_hands": int(len(result["boxes"])),
                "max_score": float(max(result["scores"])) if len(result["scores"]) > 0 else None,
                "gd_raw_count": result.get("gd_raw_count"),
                "gd_kept_count": result.get("gd_kept_count"),
                "sam3_raw_count": result.get("sam3_raw_count"),
                "sam3_kept_count": result.get("sam3_kept_count"),
                "wrists": [list(w) for w in result.get("wrists", [])],
                "mp_landmarks": [
                    [[float(x), float(y)] for x, y in pts] if pts else None
                    for pts in result.get("mp_landmarks", [])
                ],
                "mp_handedness": result.get("mp_handedness", []),
                "mp_scores": result.get("mp_scores", []),
                "mp_crops": [list(c) for c in result.get("mp_crops", [])],
                "panel": str(out_path.relative_to(root)),
            }
            summary.append(entry)
            print(
                f"  {vp.stem} f={idx:5d}: [{result['source']:14s}] "
                f"n_hands={entry['n_hands']} max_score={entry['max_score']}"
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    total = sum(counts.values())
    print(
        f"\nProcessed {total} frames | "
        f"GD primary (=2): {counts['grounding_dino']} | "
        f"SAM3 fallback (GD<2): {counts['sam3_fallback']} | "
        f"SAM3 filter (GD>=3): {counts['sam3_filter']}"
    )
    print(f"Panels: {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

"""Annotate L/R hand labels onto tracking videos via MediaPipe HandLandmarker.

For each video in a tracking-output dir (default outputs/track_v<N>/), this:
  1. Loads <stem>_track.frames.json (per-frame COCO-RLE masks per obj_id).
  2. Samples up to N_SAMPLE_FRAMES evenly across the video.
  3. For each sampled frame, for each obj_id with a non-empty mask, crops the
     RGB frame around the mask bbox expanded by MP_CROP_EXPAND, squared, and
     runs MP HandLandmarker (IMAGE mode) on the crop.
  4. Tallies MP handedness votes per obj_id; assigns 'left' / 'right' by
     majority. obj_ids that get no MP signal are left unlabeled.
  5. Re-renders the overlay video as <stem>_track_labeled.mp4 with:
       - the existing mask fill + bbox (same colors as the tracker)
       - convex hull of the mask's largest connected component
       - 'L' or 'R' label at the centroid of each hand
  6. Writes the handedness map back into <stem>_track.json
     (`wearer_handedness_by_obj_id`).

This is a post-process to src/track_video_sam2.py; the tracker remains
detector-only and this script adds anatomical labels via MP.
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

# MediaPipe model paths in priority order
MP_MODEL_PATHS = [
    Path("/home/jingjin/.cache/mediapipe/hand_landmarker.task"),
    Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task",
]
MP_HAND_DET_CONF = 0.20         # generous - the crop is already tight around a hand
MP_CROP_EXPAND = 0.50           # bbox expansion before squaring/cropping for MP
N_SAMPLE_FRAMES = 24            # how many frames to sample for handedness voting
MIN_VOTES_TO_LABEL = 1          # require at least this many MP detections per obj

# Visualization
MASK_COLORS_BGR = [(255, 80, 0), (0, 80, 255)]
HULL_COLORS_BGR = [(255, 220, 0), (0, 220, 255)]
LABEL_BG_COLOR = (0, 0, 0)
LABEL_FG_COLOR = (0, 255, 255)


def _find_mp_model() -> Path:
    for p in MP_MODEL_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("hand_landmarker.task not found in any known cache.")


def load_mp_image() -> mp_vision.HandLandmarker:
    base = mp_python.BaseOptions(model_asset_path=str(_find_mp_model()))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=MP_HAND_DET_CONF,
        min_hand_presence_confidence=MP_HAND_DET_CONF,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def decode_mask_rle(rle: dict) -> np.ndarray:
    raw = {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")}
    return coco_mask.decode(raw).astype(bool)


def expand_to_square_crop(
    bbox, image_w: int, image_h: int, expand_frac: float = MP_CROP_EXPAND
) -> tuple[int, int, int, int]:
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


def mp_handedness_on_crop(mp_image: mp_vision.HandLandmarker, frame_rgb: np.ndarray, bbox, W: int, H: int) -> str | None:
    sx1, sy1, sx2, sy2 = expand_to_square_crop(bbox, W, H)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    crop = np.ascontiguousarray(frame_rgb[sy1:sy2, sx1:sx2])
    if crop.size == 0:
        return None
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = mp_image.detect(img)
    if not result.handedness:
        return None
    return result.handedness[0][0].category_name  # "Left" or "Right"


def largest_cc_hull(mask: np.ndarray) -> np.ndarray | None:
    if mask.sum() == 0:
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    component = (labels == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    pts = np.vstack(contours).reshape(-1, 2)
    return cv2.convexHull(pts.astype(np.int32))


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask)
    return (int(xs.mean()), int(ys.mean()))


def determine_handedness(
    video_path: Path,
    frames_meta: dict,
    mp_image: mp_vision.HandLandmarker,
    n_samples: int = N_SAMPLE_FRAMES,
) -> tuple[dict[int, str], dict[int, dict[str, int]]]:
    """Sample frames, run MP on each obj_id's hand, return per-obj L/R + vote tallies."""
    frames = frames_meta["frames"]
    W, H = frames_meta["size"]
    n_frames = len(frames)
    if n_frames == 0:
        return {}, {}
    sample_idxs = sorted(set(np.linspace(0, n_frames - 1, min(n_samples, n_frames)).round().astype(int).tolist()))
    # Index the frames we need
    needed = set(sample_idxs)
    votes: dict[int, dict[str, int]] = {}

    cap = cv2.VideoCapture(str(video_path))
    fidx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fidx in needed:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_meta = frames[fidx]
            for h in frame_meta["hands"]:
                if h.get("bbox") is None or h.get("mask_rle") is None:
                    continue
                oid = int(h["obj_id"])
                label = mp_handedness_on_crop(mp_image, rgb, h["bbox"], W, H)
                if label in ("Left", "Right"):
                    d = votes.setdefault(oid, {"Left": 0, "Right": 0})
                    d[label] += 1
        fidx += 1
    cap.release()

    handedness: dict[int, str] = {}
    ambiguous: list[int] = []
    for oid, d in votes.items():
        if d["Left"] + d["Right"] < MIN_VOTES_TO_LABEL:
            continue
        if d["Left"] > d["Right"]:
            handedness[oid] = "left"
        elif d["Right"] > d["Left"]:
            handedness[oid] = "right"
        else:
            ambiguous.append(oid)
    # Pairwise complement: if exactly one obj is unambiguously labeled and another
    # is tied, assume the tied one is the opposite anatomical hand.
    if ambiguous and len(handedness) == 1:
        known_oid, known_label = next(iter(handedness.items()))
        opposite = "right" if known_label == "left" else "left"
        for oid in ambiguous:
            handedness[oid] = opposite
    return handedness, votes


def render_labeled_video(
    video_path: Path,
    frames_meta: dict,
    handedness: dict[int, str],
    out_path: Path,
):
    frames = frames_meta["frames"]
    W, H = frames_meta["size"]
    fps = frames_meta["fps"]
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    fidx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        overlay = bgr.copy()
        frame_meta = frames[fidx] if fidx < len(frames) else {"hands": []}
        for h in frame_meta["hands"]:
            if h.get("mask_rle") is None:
                continue
            oid = int(h["obj_id"])
            mask = decode_mask_rle(h["mask_rle"])
            if mask.sum() == 0:
                continue
            color = MASK_COLORS_BGR[oid % len(MASK_COLORS_BGR)]
            hull_color = HULL_COLORS_BGR[oid % len(HULL_COLORS_BGR)]
            # Mask fill
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(color, dtype=np.uint8)).astype(np.uint8)
            # Hull polyline
            hull = largest_cc_hull(mask)
            if hull is not None:
                cv2.polylines(overlay, [hull], isClosed=True, color=hull_color, thickness=3, lineType=cv2.LINE_AA)
            # Bbox
            if h.get("bbox") is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in h["bbox"]]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            # L/R label at mask centroid
            c = mask_centroid(mask)
            if c is not None:
                hd = handedness.get(oid)
                tag = "L" if hd == "left" else "R" if hd == "right" else "?"
                text = f"{tag}  obj{oid}"
                cv2.putText(overlay, text, (c[0] - 30, c[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, LABEL_BG_COLOR, 6, cv2.LINE_AA)
                cv2.putText(overlay, text, (c[0] - 30, c[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, LABEL_FG_COLOR, 2, cv2.LINE_AA)
        writer.write(overlay)
        fidx += 1
    cap.release()
    writer.release()


def process_one(track_dir: Path, video_stem: str, source_dir: Path, mp_image: mp_vision.HandLandmarker) -> dict:
    video_path = source_dir / f"{video_stem}.mp4"
    frames_json = track_dir / f"{video_stem}_track.frames.json"
    meta_json = track_dir / f"{video_stem}_track.json"
    out_video = track_dir / f"{video_stem}_track_labeled.mp4"
    if not video_path.exists():
        return {"video": video_stem, "error": "source mp4 missing"}
    if not frames_json.exists():
        return {"video": video_stem, "error": "frames json missing"}

    frames_meta = json.loads(frames_json.read_text())
    handedness, votes = determine_handedness(video_path, frames_meta, mp_image)
    render_labeled_video(video_path, frames_meta, handedness, out_video)

    # Update meta json
    if meta_json.exists():
        meta = json.loads(meta_json.read_text())
        meta["wearer_handedness_by_obj_id"] = {str(k): v for k, v in handedness.items()}
        meta["handedness_votes_by_obj_id"] = {str(k): v for k, v in votes.items()}
        meta_json.write_text(json.dumps(meta, indent=2))

    return {
        "video": video_stem,
        "wearer_handedness_by_obj_id": {str(k): v for k, v in handedness.items()},
        "votes": {str(k): v for k, v in votes.items()},
        "labeled_video": out_video.name,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-dir", default=None, help="Tracking dir; default: latest outputs/track_v<N>")
    ap.add_argument("--source-dir", default="data", help="Source mp4 dir")
    ap.add_argument("--videos", nargs="*", default=None, help="video stems (e.g. rgb_10); default: all in track-dir")
    return ap.parse_args()


def latest_track_dir(base: Path) -> Path:
    candidates = [p for p in base.glob("track_v*") if p.is_dir() and p.name.split("v")[-1].isdigit()]
    candidates.sort(key=lambda p: int(p.name.split("v")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No track_v* dirs in {base}")
    return candidates[-1]


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    track_dir = Path(args.track_dir).resolve() if args.track_dir else latest_track_dir(root / "outputs")
    source_dir = (Path(args.source_dir) if Path(args.source_dir).is_absolute() else (root / args.source_dir)).resolve()
    print(f"Track dir: {track_dir}")
    print(f"Source dir: {source_dir}")

    if args.videos:
        stems = args.videos
    else:
        stems = sorted(p.stem.replace("_track", "") for p in track_dir.glob("*_track.mp4")
                       if not p.stem.endswith("_track_labeled"))

    print(f"Videos to label: {len(stems)}")
    mp_image = load_mp_image()
    summary = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        tic = time.time()
        try:
            res = process_one(track_dir, stem, source_dir, mp_image)
            summary.append(res)
            h = res.get("wearer_handedness_by_obj_id", {})
            print(f"  [{i}/{len(stems)}] {stem}: {h} ({time.time()-tic:.1f}s)")
        except Exception as e:
            print(f"  [{i}/{len(stems)}] {stem}: ERROR {e!r}")
            summary.append({"video": stem, "error": repr(e)})
    (track_dir / "labeling_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""MediaPipe video-mode hand-pose estimation for the first N tracked videos.

Two-pass design:

  Pass 1 (per video): for each frame
    1. Run MP HandLandmarker in VIDEO mode on every frame (max 2 hands).
    2. Match each MP detection to a tracker hand by IoU of bboxes.
    3. Size gate: |MP_hull_area / mask-largest-CC-hull_area| in [SIZE_GATE_MIN, SIZE_GATE_MAX].
    4. If it FAILS the gate:
         - mask area < 0.5% of image -> drop the hand for this frame
         - else: rerun MP in IMAGE mode using the mask-derived bbox expanded 50% to a square.
    5. Consistency: reject poses whose per-keypoint mean displacement from the previous
       accepted pose for that hand exceeds CONSISTENCY_MAX_PX_FRAC * image-diagonal.

  Between passes:
    Classify each obj_id as 'left' or 'right' via MP's per-frame handedness
    majority vote. MP's handedness is the anatomical label of the hand itself
    and is independent of camera angle.

  Pass 2 (per video): render the overlay video using the labels from the classifier.
    - SAM 2 mask filled overlay
    - Convex hull of the mask's largest connected component (polyline)
    - MP skeleton + 21 keypoints + cyan wrist landmark
    - 'L' or 'R' label at the wrist (wearer-anatomical)

Outputs (per video):
  outputs/pose_v<N>/<stem>_pose.json   - per-frame data + 'wearer_handedness' per obj_id
  outputs/pose_v<N>/<stem>_pose.mp4    - overlay video
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch  # noqa: F401  (kept to share env with other scripts)
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pycocotools import mask as coco_mask
from scipy.spatial import ConvexHull

MP_MODEL_PATHS = [
    Path("/home/jingjin/.cache/mediapipe/hand_landmarker.task"),
    Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task",
]
MP_HAND_DET_CONF = 0.30

# Size-gate / consistency tunables.
# Note: the MP 21-keypoint convex hull is naturally a fraction of the mask's
# silhouette convex hull (keypoints are interior, the mask outlines the silhouette).
# Empirically the median ratio MP_hull/mask_hull on these videos is ~0.6.
# The asymmetric gate catches the two failure modes:
#   - MP collapsing on a few points (ratio << 0.3)
#   - MP overshooting onto another object (ratio >> 1.25)
SIZE_GATE_MIN_RATIO = 0.30
SIZE_GATE_MAX_RATIO = 1.25
DROP_IF_MASK_AREA_BELOW_PX = 2500     # absolute pixel area floor; below this we drop without MP rerun
MP_RERUN_BBOX_EXPAND = 0.50           # expand by 50% before squaring + cropping
CONSISTENCY_MAX_PX_FRAC = 0.10        # mean keypoint jump cap (fraction of image diagonal)

# Visualization
HAND_COLORS_BGR = [(255, 80, 0), (0, 80, 255)]   # obj 0 cool, obj 1 warm
HULL_COLORS_BGR = [(255, 220, 0), (0, 220, 255)] # brighter polyline outline per obj
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
        min_hand_detection_confidence=MP_HAND_DET_CONF,
        min_hand_presence_confidence=MP_HAND_DET_CONF,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def decode_mask_rle(rle: dict) -> np.ndarray:
    raw = {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")}
    return coco_mask.decode(raw).astype(bool)


def largest_cc_hull_area(mask: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Return (hull_area_px, hull_contour) for the mask's largest connected component."""
    if mask.sum() == 0:
        return 0.0, None
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return 0.0, None
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    component = (cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)[1] == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, None
    pts = np.vstack(contours).reshape(-1, 2)
    hull = cv2.convexHull(pts.astype(np.int32))
    area = float(cv2.contourArea(hull))
    return area, hull


def points_hull_area(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(points_xy).volume)  # 2D Volume == area
    except Exception:
        return 0.0


def bbox_iou(a, b) -> float:
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


def kp_bbox(points_xy: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points_xy[:, 0].min()),
        float(points_xy[:, 1].min()),
        float(points_xy[:, 0].max()),
        float(points_xy[:, 1].max()),
    )


def expand_to_square_crop(bbox, image_w, image_h, expand_frac=MP_RERUN_BBOX_EXPAND):
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


def mp_landmarks_to_image_coords(landmarks, image_w: int, image_h: int) -> np.ndarray:
    return np.array([[lm.x * image_w, lm.y * image_h] for lm in landmarks], dtype=np.float32)


def mp_landmarks_in_crop_to_image_coords(landmarks, crop) -> np.ndarray:
    sx1, sy1, sx2, sy2 = crop
    cw, ch = sx2 - sx1, sy2 - sy1
    return np.array([[sx1 + lm.x * cw, sy1 + lm.y * ch] for lm in landmarks], dtype=np.float32)


def assign_mp_to_tracker(mp_detections, tracker_hands) -> dict[int, int]:
    """Match each MP detection (idx in `mp_detections`) to a tracker hand obj_id by IoU.

    Greedy by descending IoU. Returns mp_idx -> obj_id.
    """
    if not mp_detections or not tracker_hands:
        return {}
    pairs = []
    for mi, det in enumerate(mp_detections):
        for th in tracker_hands:
            if th["bbox"] is None:
                continue
            pairs.append((bbox_iou(det["bbox"], th["bbox"]), mi, th["obj_id"]))
    pairs.sort(reverse=True)
    assigned_mp, assigned_obj, mapping = set(), set(), {}
    for iou, mi, oid in pairs:
        if iou <= 0:
            break
        if mi in assigned_mp or oid in assigned_obj:
            continue
        mapping[mi] = oid
        assigned_mp.add(mi)
        assigned_obj.add(oid)
    return mapping


def run_mp_video(landmarker, frame_rgb, ts_ms) -> list[dict]:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
    result = landmarker.detect_for_video(mp_image, int(ts_ms))
    H, W = frame_rgb.shape[:2]
    out = []
    for i, lms in enumerate(result.hand_landmarks):
        pts = mp_landmarks_to_image_coords(lms, W, H)
        handedness = None
        score = None
        if result.handedness and i < len(result.handedness) and result.handedness[i]:
            cat = result.handedness[i][0]
            handedness = cat.category_name
            score = float(cat.score)
        out.append({"keypoints": pts, "bbox": kp_bbox(pts), "handedness": handedness, "score": score})
    return out


def run_mp_image_on_crop(landmarker, frame_rgb, crop) -> dict | None:
    sx1, sy1, sx2, sy2 = crop
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    sub = np.ascontiguousarray(frame_rgb[sy1:sy2, sx1:sx2])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=sub)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    pts = mp_landmarks_in_crop_to_image_coords(result.hand_landmarks[0], crop)
    handedness = None
    score = None
    if result.handedness:
        cat = result.handedness[0][0]
        handedness = cat.category_name
        score = float(cat.score)
    return {"keypoints": pts, "bbox": kp_bbox(pts), "handedness": handedness, "score": score}


def size_gate_passes(mp_pts: np.ndarray, mask: np.ndarray) -> tuple[bool, float, float]:
    mp_area = points_hull_area(mp_pts)
    mask_hull_area, _ = largest_cc_hull_area(mask)
    if mask_hull_area <= 0 or mp_area <= 0:
        return False, mp_area, mask_hull_area
    ratio = mp_area / mask_hull_area
    return SIZE_GATE_MIN_RATIO <= ratio <= SIZE_GATE_MAX_RATIO, mp_area, mask_hull_area


def draw_skeleton(img, pts, color):
    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, ipts[a], ipts[b], SKEL_COLOR, 2, lineType=cv2.LINE_AA)
    for i, p in enumerate(ipts):
        cv2.circle(img, p, 7, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, p, 5, WRIST_COLOR if i == 0 else KP_COLOR, -1, lineType=cv2.LINE_AA)


def overlay_mask(img, mask, color_bgr):
    if mask is None or mask.sum() == 0:
        return img
    color = np.array(color_bgr, dtype=np.uint8)
    img[mask] = (0.55 * img[mask] + 0.45 * color).astype(np.uint8)
    return img


def draw_mask_hull(img, mask, color_bgr):
    """Draw the convex hull of the largest connected component of `mask`."""
    if mask is None or mask.sum() == 0:
        return
    _area, hull = largest_cc_hull_area(mask)
    if hull is None:
        return
    cv2.polylines(img, [hull.astype(np.int32)], isClosed=True,
                  color=color_bgr, thickness=3, lineType=cv2.LINE_AA)


def classify_wearer_handedness(per_frame_records: list[dict]) -> dict[int, str]:
    """Per obj_id, take MP's handedness majority vote across the video.

    MP's `handedness` field labels the hand anatomically (Left vs Right hand of
    whoever it belongs to) based on the hand's own shape; it is independent of
    camera angle or ego/third-person framing. We trust it directly.

    Returns {obj_id: 'left' | 'right'}. obj_ids with no MP votes are omitted.
    """
    per_obj: dict[int, dict[str, int]] = {}
    for f in per_frame_records:
        for h in f["hands"]:
            oid = int(h["obj_id"])
            d = per_obj.setdefault(oid, {"Left": 0, "Right": 0})
            handed = h.get("handedness")
            if handed == "Left":
                d["Left"] += 1
            elif handed == "Right":
                d["Right"] += 1
    out: dict[int, str] = {}
    for oid, d in per_obj.items():
        if d["Left"] > d["Right"]:
            out[oid] = "left"
        elif d["Right"] > d["Left"]:
            out[oid] = "right"
        # else: insufficient evidence, leave unlabeled
    return out


def pick_next_version_dir(base: Path, prefix: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        c = base / f"{prefix}_v{n}"
        if not c.exists():
            c.mkdir()
            return c
        n += 1


def _detect_pass(
    video_path: Path,
    frames_meta: dict,
    mp_video: mp_vision.HandLandmarker,
    mp_image: mp_vision.HandLandmarker,
) -> list[dict]:
    """Pass 1: per-frame pose detection with size gate + image-mode rerun + consistency filter."""
    W, H = frames_meta["size"]
    fps = frames_meta["fps"]
    image_diag = float(np.hypot(W, H))
    consistency_thresh_px = CONSISTENCY_MAX_PX_FRAC * image_diag
    image_area = float(W) * float(H)

    cap = cv2.VideoCapture(str(video_path))
    prev_pose_by_obj: dict[int, np.ndarray] = {}
    per_frame_records: list[dict] = []
    fidx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ts_ms = int(round(fidx * 1000.0 / fps))
        tracker_hands = frames_meta["frames"][fidx]["hands"] if fidx < len(frames_meta["frames"]) else []
        bboxes_by_obj: dict[int, list[float]] = {}
        for th in tracker_hands:
            if th["bbox"] is not None:
                bboxes_by_obj[int(th["obj_id"])] = th["bbox"]

        mp_dets = run_mp_video(mp_video, rgb, ts_ms)
        mapping = assign_mp_to_tracker(mp_dets, tracker_hands)

        records_this_frame: list[dict] = []
        for th in tracker_hands:
            oid = int(th["obj_id"])
            if th["mask_rle"] is None:
                continue
            mask = decode_mask_rle(th["mask_rle"])
            mask_area_px = float(mask.sum())
            mask_area_frac = mask_area_px / image_area

            entry = {
                "obj_id": oid,
                "mask_area_px": mask_area_px,
                "mask_area_frac": mask_area_frac,
                "mask_bbox": bboxes_by_obj.get(oid),
                "source": None,
                "keypoints": None,
                "handedness": None,
                "mp_score": None,
                "mp_hull_area": None,
                "mask_hull_area": None,
                "size_gate": None,
                "consistency_dist_px": None,
                "rejected_reason": None,
            }

            mp_idx = next((mi for mi, o in mapping.items() if o == oid), None)
            candidate = mp_dets[mp_idx] if mp_idx is not None else None
            if candidate is not None:
                passes, mp_area, mask_hull = size_gate_passes(candidate["keypoints"], mask)
                entry["mp_hull_area"] = mp_area
                entry["mask_hull_area"] = mask_hull
                entry["size_gate"] = bool(passes)
                if passes:
                    entry["keypoints"] = candidate["keypoints"]
                    entry["handedness"] = candidate["handedness"]
                    entry["mp_score"] = candidate["score"]
                    entry["source"] = "mp_video"

            if entry["keypoints"] is None and entry["mask_bbox"] is not None:
                if mask_area_px < DROP_IF_MASK_AREA_BELOW_PX:
                    entry["rejected_reason"] = "mask_too_small"
                else:
                    crop = expand_to_square_crop(entry["mask_bbox"], W, H)
                    rerun = run_mp_image_on_crop(mp_image, rgb, crop)
                    if rerun is not None:
                        passes2, mp_area2, mask_hull2 = size_gate_passes(rerun["keypoints"], mask)
                        entry["mp_hull_area"] = mp_area2
                        entry["mask_hull_area"] = mask_hull2
                        entry["size_gate"] = bool(passes2)
                        if passes2:
                            entry["keypoints"] = rerun["keypoints"]
                            entry["handedness"] = rerun["handedness"]
                            entry["mp_score"] = rerun["score"]
                            entry["source"] = "mp_image_rerun"
                        else:
                            entry["rejected_reason"] = "size_gate_after_rerun"
                    else:
                        entry["rejected_reason"] = "mp_no_detection"

            if entry["keypoints"] is not None and oid in prev_pose_by_obj:
                prev = prev_pose_by_obj[oid]
                jump = float(np.linalg.norm(entry["keypoints"] - prev, axis=1).mean())
                entry["consistency_dist_px"] = jump
                if jump > consistency_thresh_px:
                    entry["keypoints"] = None
                    entry["source"] = None
                    entry["rejected_reason"] = "consistency_jump"

            if entry["keypoints"] is not None:
                prev_pose_by_obj[oid] = entry["keypoints"]
            records_this_frame.append(entry)
        per_frame_records.append({"frame": fidx, "hands": records_this_frame})
        fidx += 1
    cap.release()
    return per_frame_records


def _render_pass(
    video_path: Path,
    frames_meta: dict,
    per_frame_records: list[dict],
    handedness_map: dict[int, str],
    out_video_path: Path,
) -> None:
    """Pass 2: render overlay video using detection results + handedness labels."""
    W, H = frames_meta["size"]
    fps = frames_meta["fps"]
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))
    fidx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        tracker_hands = frames_meta["frames"][fidx]["hands"] if fidx < len(frames_meta["frames"]) else []
        masks_by_obj: dict[int, np.ndarray] = {}
        for th in tracker_hands:
            if th["mask_rle"] is None:
                continue
            masks_by_obj[int(th["obj_id"])] = decode_mask_rle(th["mask_rle"])

        overlay = bgr.copy()
        # Mask fills + mask-hull polylines
        for oid, mask in masks_by_obj.items():
            color = HAND_COLORS_BGR[oid % len(HAND_COLORS_BGR)]
            overlay = overlay_mask(overlay, mask, color)
            draw_mask_hull(overlay, mask, HULL_COLORS_BGR[oid % len(HULL_COLORS_BGR)])

        # Skeletons + L/R labels for accepted poses
        rec_by_obj = {h["obj_id"]: h for h in per_frame_records[fidx]["hands"]} if fidx < len(per_frame_records) else {}
        for oid, rec in rec_by_obj.items():
            if rec.get("keypoints") is None:
                continue
            pts = rec["keypoints"]
            if isinstance(pts, list):
                pts = np.asarray(pts, dtype=np.float32)
            color = HAND_COLORS_BGR[oid % len(HAND_COLORS_BGR)]
            draw_skeleton(overlay, pts, color)
            wx, wy = int(round(pts[0][0])), int(round(pts[0][1]))
            handed = handedness_map.get(oid)
            tag = "L" if handed == "left" else "R" if handed == "right" else "?"
            label = f"{tag}  obj{oid} {rec.get('source','')}"
            cv2.putText(overlay, label, (wx + 12, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, label, (wx + 12, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, LABEL_TEXT_COLOR, 2, cv2.LINE_AA)
        writer.write(overlay)
        fidx += 1
    cap.release()
    writer.release()


def process_one_video(
    video_path: Path,
    frames_json_path: Path,
    out_dir: Path,
    mp_video: mp_vision.HandLandmarker,
    mp_image: mp_vision.HandLandmarker,
):
    frames_meta = json.loads(frames_json_path.read_text())
    out_video = out_dir / f"{video_path.stem}_pose.mp4"
    out_json = out_dir / f"{video_path.stem}_pose.json"

    per_frame_records = _detect_pass(video_path, frames_meta, mp_video, mp_image)
    handedness_map = classify_wearer_handedness(per_frame_records)
    _render_pass(video_path, frames_meta, per_frame_records, handedness_map, out_video)

    # Materialize handedness onto each frame entry and serialize.
    serial_frames = []
    for f in per_frame_records:
        hands_out = []
        for h in f["hands"]:
            jh = dict(h)
            if h.get("keypoints") is not None:
                kp = h["keypoints"]
                if isinstance(kp, np.ndarray):
                    kp = kp.tolist()
                jh["keypoints"] = [[float(x), float(y)] for x, y in kp]
            jh["wearer_handedness"] = handedness_map.get(int(h["obj_id"]))
            hands_out.append(jh)
        serial_frames.append({"frame": f["frame"], "hands": hands_out})

    out_json.write_text(json.dumps({
        "video": video_path.name,
        "size": frames_meta["size"],
        "fps": frames_meta["fps"],
        "params": {
            "size_gate_min_ratio": SIZE_GATE_MIN_RATIO,
            "size_gate_max_ratio": SIZE_GATE_MAX_RATIO,
            "drop_if_mask_area_below_px": DROP_IF_MASK_AREA_BELOW_PX,
            "mp_rerun_bbox_expand": MP_RERUN_BBOX_EXPAND,
            "consistency_max_px_frac": CONSISTENCY_MAX_PX_FRAC,
        },
        "wearer_handedness_by_obj_id": {str(k): v for k, v in handedness_map.items()},
        "frames": serial_frames,
    }))
    return out_video, out_json


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-dir", default=None, help="Tracking output dir. Default: latest outputs/track_v*.")
    ap.add_argument("--videos", nargs="*", default=None, help="Video paths (default: first 10 from data/).")
    ap.add_argument("--num", type=int, default=10, help="Number of videos to process when --videos is unset.")
    ap.add_argument("--output-base", default="outputs")
    return ap.parse_args()


def latest_track_dir(base: Path) -> Path:
    candidates = sorted(base.glob("track_v*"), key=lambda p: int(p.name.split("v")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No track_v* dirs in {base}")
    return candidates[-1]


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    track_dir = Path(args.track_dir).resolve() if args.track_dir else latest_track_dir(root / args.output_base)
    out_dir = pick_next_version_dir(root / args.output_base, "pose")
    print(f"Track dir: {track_dir}")
    print(f"Output dir: {out_dir}")

    videos = (
        [Path(v) if Path(v).is_absolute() else (root / v) for v in args.videos]
        if args.videos
        else sorted((root / "data").glob("rgb_*.mp4"))[: args.num]
    )
    print(f"Videos to process: {len(videos)}")

    # MP video mode keeps a monotonic timestamp clock across calls — re-create per video.
    mp_image = make_mp(mp_vision.RunningMode.IMAGE, num_hands=1)

    summary = []
    t0 = time.time()
    for i, vp in enumerate(videos, 1):
        frames_json = track_dir / f"{vp.stem}_track.frames.json"
        if not frames_json.exists():
            print(f"  [{i}/{len(videos)}] {vp.name}: missing {frames_json.name}, skipping")
            continue
        tic = time.time()
        try:
            mp_video = make_mp(mp_vision.RunningMode.VIDEO, num_hands=2)
            ov, oj = process_one_video(vp, frames_json, out_dir, mp_video, mp_image)
            mp_video.close()
            summary.append({"video": vp.name, "pose_video": ov.name, "pose_json": oj.name})
            print(f"  [{i}/{len(videos)}] {vp.name}: done in {time.time()-tic:.1f}s")
        except Exception as e:
            print(f"  [{i}/{len(videos)}] {vp.name}: ERROR {e!r}")
            summary.append({"video": vp.name, "error": repr(e)})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {time.time()-t0:.1f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()

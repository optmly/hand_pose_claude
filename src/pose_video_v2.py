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
# Size sanity: a candidate pose whose bbox area exceeds this multiple of
# the mask bbox area is treated as a hallucination.
POSE_BBOX_MAX_RATIO = 2.5
# Temporal jump cap (fraction of image diagonal) for per-kp mean
# displacement vs the most recently accepted pose for the same obj_id.
CONSISTENCY_MAX_PX_FRAC = 0.10
# How long a stretch of rejections may "carry forward" the previous pose
# instead of leaving the frame empty.
CONT_CARRYFWD_MAX_FRAMES = 10
# IMAGE-mode rerun crop expansion (fraction of mask bbox, square)
RERUN_BBOX_EXPAND = 0.50
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


def gate_kpts_in_hull(kpts: np.ndarray, hull: np.ndarray) -> tuple[bool, float]:
    """At least MIN_KPTS_IN_HULL_FRAC of the 21 kpts must be inside the hull."""
    if hull is None:
        return False, 0.0
    h = hull.astype(np.float32)
    inside = 0
    for (x, y) in kpts:
        if cv2.pointPolygonTest(h, (float(x), float(y)), False) >= 0:
            inside += 1
    frac = inside / max(len(kpts), 1)
    return (frac >= MIN_KPTS_IN_HULL_FRAC), frac


def gate_size_sanity(kpts: np.ndarray, mask_bbox) -> tuple[bool, float]:
    """Pose bbox area must not exceed POSE_BBOX_MAX_RATIO * mask_bbox area."""
    if mask_bbox is None:
        return False, 0.0
    pose_area = bbox_area(kp_bbox(kpts))
    mask_a = bbox_area(mask_bbox)
    if mask_a <= 0:
        return False, 0.0
    ratio = pose_area / mask_a
    return (ratio <= POSE_BBOX_MAX_RATIO), ratio


def gate_temporal_jump(kpts: np.ndarray, prev_kpts: np.ndarray | None,
                       image_diag: float) -> tuple[bool, float]:
    """Mean per-kp displacement vs prev_kpts must be <= CONSISTENCY_MAX_PX_FRAC * image_diag."""
    if prev_kpts is None:
        return True, 0.0
    jump = float(np.linalg.norm(kpts - prev_kpts, axis=1).mean())
    return (jump <= CONSISTENCY_MAX_PX_FRAC * image_diag), jump


# ──────────────────────── MP wrappers ────────────────────────

def run_mp_video_frame(landmarker, frame_rgb: np.ndarray, ts_ms: int) -> list[dict]:
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.ascontiguousarray(frame_rgb))
    res = landmarker.detect_for_video(img, int(ts_ms))
    H, W = frame_rgb.shape[:2]
    out = []
    for i, lms in enumerate(res.hand_landmarks):
        pts = mp_pts_to_image(lms, W, H)
        score = float(res.handedness[i][0].score) if (res.handedness and i < len(res.handedness)) else 0.0
        handed = res.handedness[i][0].category_name if (res.handedness and i < len(res.handedness)) else None
        out.append({"keypoints": pts, "score": score, "handedness": handed})
    return out


def run_mp_image_crop(landmarker, frame_rgb: np.ndarray, crop) -> dict | None:
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
    return {"keypoints": pts, "score": score, "handedness": handed}


# ──────────────────────── per-frame selection ────────────────────────

def gate_candidate(kpts: np.ndarray, mask_hull: np.ndarray, hull_diag: float,
                    mask_bbox) -> tuple[bool, dict]:
    """Run the wrist + kpts + size gates on a candidate pose against an obj_id's
    mask. Returns (passes, gate_diag_dict)."""
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
    spass, sratio = gate_size_sanity(kpts, mask_bbox)
    diag["pose_bbox_to_mask_bbox_ratio"] = sratio
    diag["size_sanity"] = spass
    if not spass:
        return False, diag
    return True, diag


def select_best_for_obj(mp_dets: list[dict], mask_hull, hull_diag, mask_bbox
                         ) -> tuple[dict | None, list[dict]]:
    """Return the best-scoring MP detection that passes all gates for this obj_id,
    along with per-candidate gate diagnostics."""
    diag_per_cand = []
    qualifying = []
    for i, det in enumerate(mp_dets):
        passes, gd = gate_candidate(det["keypoints"], mask_hull, hull_diag, mask_bbox)
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

def process_one_video(video_path: Path, track_dir: Path, out_dir: Path,
                       mp_video, mp_image, max_frames: int | None = None,
                       vitpose=None) -> dict:
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
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

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

        # MP VIDEO pass
        mp_dets = run_mp_video_frame(mp_video, rgb, ts_ms)

        frame_record = {"frame": fidx, "hands": []}
        accepted_pts: dict[int, np.ndarray] = {}
        for oid in (0, 1):
            entry = {
                "obj_id": oid,
                "source": None,
                "keypoints": None,
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

            best, diag_per_cand = select_best_for_obj(
                mp_dets, hulls[oid], hull_diags[oid], mask_bboxes[oid],
            )
            entry["candidates_diag"] = diag_per_cand

            if best is None:
                # IMAGE-mode rerun on mask-bbox crop
                crop = expand_to_square_crop(mask_bboxes[oid], W, H)
                rerun = run_mp_image_crop(mp_image, rgb, crop)
                if rerun is not None:
                    passes, gd = gate_candidate(
                        rerun["keypoints"], hulls[oid], hull_diags[oid], mask_bboxes[oid],
                    )
                    entry["gate_diag"] = gd
                    if passes:
                        best = rerun
                        entry["source"] = "mp_image_rerun"
                    else:
                        entry["rejected_reason"] = "image_rerun_gate_fail"
                else:
                    entry["rejected_reason"] = "image_rerun_no_detection"

                # ViTPose backup: only when both MP attempts failed AND we
                # know which anatomical side this obj_id is (the wearer
                # L / R label from the tracker labeler). Pick the matching
                # side's hand keypoints from the ViTPose pass and gate them.
                if best is None and vit_active:
                    side = handedness_by_oid.get(oid)
                    if side in ("left", "right"):
                        v_kpts, v_score = vitpose.get_for_side(video_path, fidx, side)
                        if v_kpts is not None and v_kpts.shape == (21, 2):
                            passes, gd = gate_candidate(
                                v_kpts, hulls[oid], hull_diags[oid], mask_bboxes[oid],
                            )
                            entry["vitpose_gate_diag"] = gd
                            entry["vitpose_mean_score"] = v_score
                            if passes:
                                best = {"keypoints": v_kpts, "score": v_score,
                                        "handedness": side.capitalize()}
                                entry["source"] = "vitpose_huge"
                                entry["rejected_reason"] = None
                            else:
                                # Keep the prior rejected_reason from the MP
                                # branch; tack on a vit qualifier so we can
                                # tell which path the frame went through.
                                if entry["rejected_reason"]:
                                    entry["rejected_reason"] += "_then_vit_gate_fail"
                                else:
                                    entry["rejected_reason"] = "vit_gate_fail"
            else:
                entry["source"] = "mp_video"

            if best is not None:
                # Temporal jump check
                jump_ok, jump_px = gate_temporal_jump(
                    best["keypoints"], prev_pose[oid], image_diag,
                )
                if not jump_ok:
                    # Reject and try to carry forward
                    entry["rejected_reason"] = f"temporal_jump_{jump_px:.0f}px"
                    if prev_pose[oid] is not None and prev_pose_age[oid] < CONT_CARRYFWD_MAX_FRAMES:
                        entry["source"] = "carryforward"
                        entry["keypoints"] = prev_pose[oid].copy()
                        prev_pose_age[oid] += 1
                    else:
                        entry["keypoints"] = None
                else:
                    entry["keypoints"] = best["keypoints"]
                    entry["mp_score"] = best["score"]
                    prev_pose[oid] = best["keypoints"]
                    prev_pose_age[oid] = 0
            else:
                # No accepted candidate; try carry-forward
                if prev_pose[oid] is not None and prev_pose_age[oid] < CONT_CARRYFWD_MAX_FRAMES:
                    entry["source"] = "carryforward"
                    entry["keypoints"] = prev_pose[oid].copy()
                    prev_pose_age[oid] += 1
                else:
                    prev_pose[oid] = None
                    prev_pose_age[oid] = CONT_CARRYFWD_MAX_FRAMES  # locked off

            if entry["keypoints"] is not None:
                accepted_pts[oid] = entry["keypoints"]
            frame_record["hands"].append(entry)

        per_frame.append(frame_record)

        # Render overlay
        overlay = bgr.copy()
        for oid, m in masks.items():
            overlay = overlay_mask(overlay, m, MASK_COLORS_BGR[oid % 2])
            draw_mask_hull(overlay, hulls[oid], HULL_COLORS_BGR[oid % 2])
        for oid, pts in accepted_pts.items():
            draw_skeleton(overlay, pts, MASK_COLORS_BGR[oid % 2])
            wx, wy = int(round(pts[0][0])), int(round(pts[0][1]))
            tag = handedness_by_oid.get(oid)
            tag_short = "L" if tag == "left" else "R" if tag == "right" else "?"
            src = next(h["source"] for h in frame_record["hands"] if h["obj_id"] == oid)
            text = f"{tag_short} obj{oid} {src}"
            cv2.putText(overlay, text, (wx + 14, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, text, (wx + 14, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LABEL_TEXT_COLOR, 2, cv2.LINE_AA)
        draw_frame_number(overlay, fidx)
        writer.write(overlay)

    cap.release()
    writer.release()

    # Serialize keypoints as lists
    serial = []
    for rec in per_frame:
        hands_out = []
        for h in rec["hands"]:
            hh = dict(h)
            if isinstance(hh.get("keypoints"), np.ndarray):
                hh["keypoints"] = hh["keypoints"].tolist()
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
    return ap.parse_args()


def main():
    args = parse_args()
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
            # MP VIDEO mode needs a fresh landmarker per video (monotonic ts_ms)
            mp_video = make_mp(mp_vision.RunningMode.VIDEO, num_hands=MP_NUM_HANDS_VIDEO)
            cap = cv2.VideoCapture(str(video_path))
            fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            max_frames = int(round(args.max_sec * fps_native))
            res = process_one_video(video_path, track_dir, out_dir, mp_video, mp_image,
                                     max_frames=max_frames, vitpose=vitpose)
            mp_video.close()
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

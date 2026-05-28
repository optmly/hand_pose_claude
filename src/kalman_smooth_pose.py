"""Per-keypoint Kalman (RTS) smoother for pose outputs.

For each video in an input pose dir (e.g. outputs/pose_v3/), and for each
tracker obj_id (0 / 1) on each of the 21 keypoints' x and y axes
INDEPENDENTLY (42 univariate filters per hand), runs a constant-velocity
RTS smoother. Frames with no accepted keypoints stay as predictions
(missing measurements; predict-only). Re-renders the overlay video using
the smoothed keypoints on top of the tracker mask + hull.

Inputs:
  --pose-dir    a `pose_v<N>/` directory holding `<stem>_pose.json` files
                produced by `src/pose_video_v2.py`.
  --track-dir   the matching `track_v<N>/` directory (for `<stem>_track.frames.json`,
                which carries the per-frame masks used for the background overlay).
  --source-dir  the directory containing the source mp4s (for the actual frame
                pixels). Default: data/
  --output-dir  destination dir for `<stem>_pose_smooth.{json,mp4}`. Default:
                sibling `<pose-dir>_smooth/`.

Smoother model per (obj_id, kp_idx, axis):
    state    = [position, velocity]^T
    dynamics = constant-velocity, dt = 1 frame
    process noise: white-acceleration with std sigma_a (in px/frame^2)
    measurement noise: std sigma_m (in px)
A measurement is the accepted-pose keypoint x or y at that frame. Frames
where the pose is None (miss) or the obj_id is absent contribute no
measurement.

Defaults sigma_a = 5 (allows reasonable hand motion changes) and
sigma_m = 3 (per-keypoint detector noise). Tweak via CLI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask

# Mirror the look-and-feel of pose_video_v2.py renders.
MASK_COLORS_BGR = [(255, 80, 0), (0, 80, 255)]
HULL_COLORS_BGR = [(255, 220, 0), (0, 220, 255)]
SKEL_COLOR = (255, 0, 255)
KP_COLOR = (255, 0, 255)
WRIST_COLOR = (0, 255, 255)
LABEL_TEXT_COLOR = (0, 255, 255)
# Gap-skip rule: when a stretch of frames has no fresh detection (valid==False)
# AND the mask has moved a lot between the bracketing accepted frames, the
# smoother's interpolation is unreliable -- mark those frames as missing
# instead of rendering an extrapolated pose.
SKIP_MAX_GAP_FRAMES = 5             # gap LARGER than this triggers the motion check
# Post-smooth mask-containment threshold: smoothed/interpolated poses
# whose kpts fall less than this fraction inside the actual tracker
# mask are NaN'd out (the pose has drifted off the wearer hand).
MIN_KPTS_IN_MASK_FRAC = 0.50
SKIP_MAX_MASK_MOTION_FRAC = 0.05    # mask centroid move > this * image diag -> skip
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


# ──────────────────────── RTS smoother ────────────────────────

def rts_smooth_1d(z: np.ndarray, valid: np.ndarray,
                  dt: float = 1.0, sigma_a: float = 5.0, sigma_m: float = 3.0,
                  init_P: float = 1e4,
                  confidence: np.ndarray | None = None,
                  conf_floor: float = 0.05) -> np.ndarray:
    """Constant-velocity RTS smoother on a 1D time series.

    Parameters
    ----------
    z       (N,) observations (entries where valid==False are ignored)
    valid   (N,) boolean mask
    dt      time step between samples (1.0 = one frame)
    sigma_a process noise on velocity (px / frame^2)
    sigma_m base measurement noise on position (px) when confidence == 1.0
    init_P  initial state-covariance diagonal (large = "we know nothing")
    confidence (N,) optional per-frame measurement confidence in [0, 1].
            When given, the effective measurement noise at frame t becomes
                sigma_m_eff(t) = sigma_m / max(confidence[t], conf_floor)
            so that low-confidence measurements are trusted less (the
            smoother weights the model prediction more heavily). With
            confidence==None or all-ones, behavior matches a fixed sigma_m.
    conf_floor floor for the confidence to avoid divide-by-tiny.

    Returns
    -------
    smoothed position (N,) float32. Frames before the first valid sample
    inherit the smoothed value at the first valid index.
    """
    N = len(z)
    if N == 0 or not valid.any():
        return np.full(N, np.nan, dtype=np.float32)

    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = sigma_a ** 2 * np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                                  [dt ** 3 / 2.0, dt ** 2]])
    I2 = np.eye(2)

    # Per-frame R_t (scalar); see docstring.
    if confidence is None:
        R_t = np.full(N, sigma_m ** 2, dtype=float)
    else:
        c = np.asarray(confidence, dtype=float)
        c = np.clip(c, conf_floor, 1.0)
        R_t = (sigma_m / c) ** 2

    first = int(np.argmax(valid))   # index of first True

    x_filt = np.zeros((N, 2))
    P_filt = np.zeros((N, 2, 2))
    x_pred_ahead = np.zeros((N, 2))     # x[t+1 | t], stored at index t
    P_pred_ahead = np.zeros((N, 2, 2))  # P[t+1 | t]

    x_filt[first] = np.array([z[first], 0.0])
    P_filt[first] = I2 * init_P

    for t in range(first + 1, N):
        # predict from t-1 to t
        x_pred = F @ x_filt[t - 1]
        P_pred = F @ P_filt[t - 1] @ F.T + Q
        x_pred_ahead[t - 1] = x_pred
        P_pred_ahead[t - 1] = P_pred

        if valid[t]:
            y = z[t] - (H @ x_pred)[0]
            S = (H @ P_pred @ H.T)[0, 0] + R_t[t]
            K = (P_pred @ H.T) / S
            x_filt[t] = x_pred + K.ravel() * y
            P_filt[t] = (I2 - K @ H) @ P_pred
        else:
            x_filt[t] = x_pred
            P_filt[t] = P_pred

    # Backward pass (RTS)
    x_smooth = x_filt.copy()
    for t in range(N - 2, first - 1, -1):
        # P_pred_ahead[t] is the prediction at t+1 from the filtered state at t.
        # Invert robustly via pseudo-inverse to avoid singular matrices.
        try:
            A = P_filt[t] @ F.T @ np.linalg.inv(P_pred_ahead[t])
        except np.linalg.LinAlgError:
            A = P_filt[t] @ F.T @ np.linalg.pinv(P_pred_ahead[t])
        x_smooth[t] = x_filt[t] + A @ (x_smooth[t + 1] - x_pred_ahead[t])

    # Frames before the first valid sample: inherit
    for t in range(first):
        x_smooth[t] = x_smooth[first]

    return x_smooth[:, 0].astype(np.float32)


def smooth_keypoints_per_obj(per_frame_kpts: list[np.ndarray | None],
                              dt: float, sigma_a: float, sigma_m: float,
                              per_frame_kp_conf: list[np.ndarray | None] | None = None,
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a per-frame sequence of (21, 2) keypoints with one independent
    RTS smoother per (kp_idx, axis). If per_frame_kp_conf is provided, the
    matching (21,) confidence array per frame is used to scale the per-
    measurement noise (see rts_smooth_1d).

    Returns
        (N, 21, 2) smoothed kpts,
        (N,) bool mask True where the frame had a measurement.
    """
    N = len(per_frame_kpts)
    obs = np.full((N, 21, 2), np.nan, dtype=np.float32)
    conf = np.ones((N, 21), dtype=np.float32)   # default = full trust
    valid_per_frame = np.zeros(N, dtype=bool)
    for t, kp in enumerate(per_frame_kpts):
        if kp is None:
            continue
        a = np.asarray(kp, dtype=np.float32)
        if a.shape != (21, 2):
            continue
        obs[t] = a
        valid_per_frame[t] = True
        if per_frame_kp_conf is not None and per_frame_kp_conf[t] is not None:
            c = np.asarray(per_frame_kp_conf[t], dtype=np.float32)
            if c.shape == (21,):
                conf[t] = c

    smoothed = np.zeros((N, 21, 2), dtype=np.float32)
    for k in range(21):
        for axis in range(2):
            z = obs[:, k, axis]
            v = valid_per_frame & ~np.isnan(z)
            smoothed[:, k, axis] = rts_smooth_1d(
                z, v, dt=dt, sigma_a=sigma_a, sigma_m=sigma_m,
                confidence=conf[:, k],
            )
    return smoothed, valid_per_frame


# ──────────────────────── IO helpers ────────────────────────

def compute_skip_intervals(valid: np.ndarray,
                            mask_centroids: list[tuple[float, float] | None],
                            image_diag: float,
                            max_gap: int = SKIP_MAX_GAP_FRAMES,
                            max_motion_frac: float = SKIP_MAX_MASK_MOTION_FRAC
                            ) -> np.ndarray:
    """For each gap of consecutive False entries in `valid` whose length is
    greater than `max_gap`, check the L2 displacement of the mask centroid
    between the bracketing valid frames. If the gap is long AND the hand
    moved a lot, mark every frame in the gap as 'skip' (return True). The
    smoother's interpolation across that gap is unreliable; the caller
    should NaN out the smoothed pose for those frames.
    """
    N = len(valid)
    skip = np.zeros(N, dtype=bool)
    valid_idxs = np.where(valid)[0]
    if len(valid_idxs) < 2:
        return skip
    thresh_px = max_motion_frac * image_diag
    for i in range(len(valid_idxs) - 1):
        a = int(valid_idxs[i])
        b = int(valid_idxs[i + 1])
        gap_len = b - a - 1
        if gap_len <= max_gap:
            continue
        ca = mask_centroids[a] if a < len(mask_centroids) else None
        cb = mask_centroids[b] if b < len(mask_centroids) else None
        if ca is None or cb is None:
            continue
        motion = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]))
        if motion > thresh_px:
            skip[a + 1 : b] = True
    return skip


def decode_mask_rle(rle: dict) -> np.ndarray:
    raw = {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")}
    return coco_mask.decode(raw).astype(bool)


def largest_cc_hull(mask: np.ndarray):
    if mask is None or mask.sum() == 0:
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return None
    biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    comp = (labels == biggest).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    pts = np.vstack(contours).reshape(-1, 2).astype(np.int32)
    return cv2.convexHull(pts)


def overlay_mask(img, mask, color_bgr, alpha: float = 0.45):
    if mask is None or mask.sum() == 0:
        return img
    color = np.array(color_bgr, dtype=np.uint8)
    img[mask] = ((1.0 - alpha) * img[mask] + alpha * color).astype(np.uint8)
    return img


def draw_mask_hull(img, hull, color_bgr):
    if hull is None:
        return
    cv2.polylines(img, [hull.astype(np.int32)], isClosed=True,
                  color=color_bgr, thickness=3, lineType=cv2.LINE_AA)


def draw_skeleton(img, pts):
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


# ──────────────────────── per-video processing ────────────────────────

def process_one_video(pose_json_path: Path, track_dir: Path, source_dir: Path,
                       out_dir: Path, sigma_a: float, sigma_m: float,
                       carryforward_kept: bool = True) -> dict:
    """Smooth and re-render one pose JSON's keypoints.

    carryforward_kept: if True, frames whose source was 'carryforward' are
        treated as valid measurements (the pipeline already accepted them);
        if False, they're treated as missing so the smoother predicts.
    """
    stem = pose_json_path.stem.replace("_pose", "")
    pose = json.loads(pose_json_path.read_text())
    W, H = pose["size"]
    fps = pose["fps"]
    handedness_by_oid = {int(k): v for k, v in pose.get("wearer_handedness_by_obj_id", {}).items()}
    n_frames = len(pose["frames"])

    # Build per-obj_id keypoint sequences (with None for misses).
    per_obj_seq: dict[int, list] = {0: [None] * n_frames, 1: [None] * n_frames}
    per_obj_conf: dict[int, list] = {0: [None] * n_frames, 1: [None] * n_frames}
    per_obj_source: dict[int, list] = {0: [None] * n_frames, 1: [None] * n_frames}
    for fi, fr in enumerate(pose["frames"]):
        for h in fr["hands"]:
            oid = int(h["obj_id"])
            if oid not in per_obj_seq:
                continue
            src = h.get("source")
            kp = h.get("keypoints")
            if kp is None:
                continue
            if not carryforward_kept and src == "carryforward":
                continue
            per_obj_seq[oid][fi] = kp
            per_obj_conf[oid][fi] = h.get("kp_confidences")
            per_obj_source[oid][fi] = src

    # Smooth each obj_id independently.
    smoothed_seq: dict[int, np.ndarray] = {}
    valid_in: dict[int, np.ndarray] = {}
    for oid in (0, 1):
        sm, val = smooth_keypoints_per_obj(
            per_obj_seq[oid], dt=1.0, sigma_a=sigma_a, sigma_m=sigma_m,
            per_frame_kp_conf=per_obj_conf[oid],
        )
        smoothed_seq[oid] = sm
        valid_in[oid] = val

    # ── Re-render ──
    src_video = source_dir / f"{stem}.mp4"
    track_frames_json = track_dir / f"{stem}_track.frames.json"
    if not src_video.exists():
        return {"video": stem, "error": "source mp4 missing"}
    if not track_frames_json.exists():
        return {"video": stem, "error": "tracker frames.json missing"}
    track_frames = json.loads(track_frames_json.read_text())

    # Per-frame mask centroids + raw masks per obj. Centroids feed the
    # gap-skip rule; raw masks feed the post-smooth mask-containment
    # check (smoothed kpts must be >= MIN_KPTS_IN_MASK_FRAC inside the
    # actual mask, else NaN'd out).
    mask_centroids: dict[int, list[tuple[float, float] | None]] = {
        0: [None] * n_frames, 1: [None] * n_frames}
    mask_arrays: dict[int, list[np.ndarray | None]] = {
        0: [None] * n_frames, 1: [None] * n_frames}
    for fi, fr in enumerate(track_frames["frames"]):
        if fi >= n_frames:
            break
        for h in fr["hands"]:
            if h.get("mask_rle") is None:
                continue
            oid = int(h["obj_id"])
            if oid not in mask_centroids:
                continue
            m = decode_mask_rle(h["mask_rle"])
            if m.sum() == 0:
                continue
            ys, xs = np.where(m)
            mask_centroids[oid][fi] = (float(xs.mean()), float(ys.mean()))
            mask_arrays[oid][fi] = m

    # Per-frame "small_edge_mask" flag from the pose stage. These frames
    # had the partial-hand skip; we NaN out the smoother output too so
    # it doesn't extrapolate across an unreliable region.
    small_edge: dict[int, list[bool]] = {
        0: [False] * n_frames, 1: [False] * n_frames}
    for fi, fr in enumerate(pose["frames"]):
        if fi >= n_frames:
            break
        for h in fr["hands"]:
            oid = int(h["obj_id"])
            if oid in small_edge and h.get("rejected_reason") == "small_edge_mask":
                small_edge[oid][fi] = True

    # Gap-skip: when the smoother has interpolated across a long no-detection
    # gap during which the hand moved a lot, NaN out the smoothed pose for
    # those frames so downstream consumers see the gap as missing.
    image_diag = float(np.hypot(W, H))
    skip_stats: dict[int, dict] = {0: {}, 1: {}}
    for oid in (0, 1):
        skip = compute_skip_intervals(
            valid_in[oid], mask_centroids[oid], image_diag,
        )
        if skip.any():
            smoothed_seq[oid][skip] = np.nan
        # Small-edge-mask frames: NaN out (pose stage already skipped
        # detection; we suppress smoother extrapolation too).
        for fi, is_se in enumerate(small_edge[oid]):
            if is_se:
                smoothed_seq[oid][fi] = np.nan
        # Mask-containment: smoothed pose must have at least
        # MIN_KPTS_IN_MASK_FRAC of keypoints inside the actual mask.
        # If not, NaN out the smoothed pose for that frame (the pose
        # has drifted too far outside the wearer hand).
        mask_skip_count = 0
        for fi in range(n_frames):
            pts = smoothed_seq[oid][fi]
            if np.isnan(pts).any():
                continue
            m = mask_arrays[oid][fi]
            if m is None:
                continue
            Hm, Wm = m.shape[:2]
            inside = 0
            for (x, y) in pts:
                xi = int(round(float(x))); yi = int(round(float(y)))
                if 0 <= xi < Wm and 0 <= yi < Hm and bool(m[yi, xi]):
                    inside += 1
            frac = inside / max(len(pts), 1)
            if frac < MIN_KPTS_IN_MASK_FRAC:
                smoothed_seq[oid][fi] = np.nan
                mask_skip_count += 1
        skip_stats[oid] = {
            "skipped_frames": int(skip.sum()),
            "small_edge_frames": int(sum(small_edge[oid])),
            "mask_containment_skipped": mask_skip_count,
        }

    out_mp4 = out_dir / f"{stem}_pose_smooth.mp4"
    out_mp4_clean = out_dir / f"{stem}_pose_smooth_clean.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    writer_clean = cv2.VideoWriter(str(out_mp4_clean), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap = cv2.VideoCapture(str(src_video))

    for fi in range(n_frames):
        ok, bgr = cap.read()
        if not ok:
            break
        overlay = bgr.copy()
        overlay_clean = bgr.copy()
        # Tracker mask + (full only) hull background; clean version uses a
        # lighter mask alpha (0.20 vs 0.45) and skips the convex-hull outline.
        if fi < len(track_frames["frames"]):
            for h in track_frames["frames"][fi]["hands"]:
                if h.get("mask_rle") is None:
                    continue
                m = decode_mask_rle(h["mask_rle"])
                if m.sum() == 0:
                    continue
                oid = int(h["obj_id"])
                overlay = overlay_mask(overlay, m, MASK_COLORS_BGR[oid % 2])
                draw_mask_hull(overlay, largest_cc_hull(m), HULL_COLORS_BGR[oid % 2])
                overlay_clean = overlay_mask(overlay_clean, m,
                                              MASK_COLORS_BGR[oid % 2], alpha=0.20)
        # Smoothed skeletons (both videos get the same skeleton)
        for oid in (0, 1):
            if not valid_in[oid].any():
                continue
            # Only draw on frames within or after the first valid measurement
            # (don't extrapolate backwards beyond the input data).
            first_valid = int(np.argmax(valid_in[oid]))
            if fi < first_valid:
                continue
            pts = smoothed_seq[oid][fi]
            if np.isnan(pts).any():
                continue
            draw_skeleton(overlay, pts)
            draw_skeleton(overlay_clean, pts)
            wx, wy = int(round(pts[0, 0])), int(round(pts[0, 1]))
            handed = handedness_by_oid.get(oid)
            tag_short = "L" if handed == "left" else "R" if handed == "right" else "?"
            label = f"{tag_short} obj{oid} kalman"
            cv2.putText(overlay, label, (wx + 14, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, label, (wx + 14, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LABEL_TEXT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(overlay_clean, label, (wx + 14, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay_clean, label, (wx + 14, wy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LABEL_TEXT_COLOR, 2, cv2.LINE_AA)
        draw_frame_number(overlay, fi)
        draw_frame_number(overlay_clean, fi)
        writer.write(overlay)
        writer_clean.write(overlay_clean)

    cap.release()
    writer.release()
    writer_clean.release()

    # Smoothed JSON dump
    out_json = out_dir / f"{stem}_pose_smooth.json"
    serial_frames = []
    for fi in range(n_frames):
        hands_out = []
        for oid in (0, 1):
            pts = smoothed_seq[oid][fi]
            entry = {
                "obj_id": oid,
                "wearer_handedness": handedness_by_oid.get(oid),
                "source": per_obj_source[oid][fi] if valid_in[oid][fi] else "kalman_predict",
                "keypoints": pts.tolist() if not np.isnan(pts).any() else None,
                "had_measurement": bool(valid_in[oid][fi]),
            }
            hands_out.append(entry)
        serial_frames.append({"frame": fi, "hands": hands_out})

    out_json.write_text(json.dumps({
        "video": pose["video"],
        "size": [W, H],
        "fps": fps,
        "wearer_handedness_by_obj_id": handedness_by_oid,
        "params": {
            "smoother": "RTS_constant_velocity_per_kp_per_axis",
            "sigma_a": sigma_a,
            "sigma_m": sigma_m,
            "carryforward_kept": carryforward_kept,
        },
        "frames": serial_frames,
    }))
    return {"video": stem, "pose_video": out_mp4.name, "pose_json": out_json.name,
            "gap_skip_per_obj": skip_stats}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-dir", required=True, help="Input pose_v<N>/ directory.")
    ap.add_argument("--track-dir", required=True, help="Matching track_v<N>/ for mask backgrounds.")
    ap.add_argument("--source-dir", required=True, help="Directory containing source mp4s.")
    ap.add_argument("--output-dir", default=None,
                    help="Destination dir (default: sibling <pose-dir>_smooth/).")
    ap.add_argument("--sigma-a", type=float, default=5.0,
                    help="Process noise (acceleration std, px/frame^2). Higher = more responsive.")
    ap.add_argument("--sigma-m", type=float, default=3.0,
                    help="Measurement noise (per-keypoint std, px). Higher = trust model more, smoother.")
    ap.add_argument("--drop-carryforward", action="store_true",
                    help="Treat frames whose source was 'carryforward' as missing (predict-only).")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Optional list of stems to process (default: all *_pose.json in --pose-dir).")
    return ap.parse_args()


def main():
    args = parse_args()
    pose_dir = Path(args.pose_dir).resolve()
    track_dir = Path(args.track_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = pose_dir.parent / (pose_dir.name + "_smooth")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pose dir:   {pose_dir}")
    print(f"Track dir:  {track_dir}")
    print(f"Source dir: {source_dir}")
    print(f"Output dir: {out_dir}")
    print(f"sigma_a={args.sigma_a}  sigma_m={args.sigma_m}  drop_carry={args.drop_carryforward}")

    if args.videos:
        pose_jsons = [pose_dir / f"{s}_pose.json" for s in args.videos]
        pose_jsons = [p for p in pose_jsons if p.exists()]
    else:
        pose_jsons = sorted(pose_dir.glob("*_pose.json"))
    print(f"Videos to smooth: {len(pose_jsons)}")

    t0 = time.time()
    summary = []
    for i, pj in enumerate(pose_jsons, 1):
        tic = time.time()
        try:
            res = process_one_video(
                pj, track_dir, source_dir, out_dir,
                sigma_a=args.sigma_a, sigma_m=args.sigma_m,
                carryforward_kept=not args.drop_carryforward,
            )
            summary.append(res)
            print(f"  [{i}/{len(pose_jsons)}] {pj.stem}: done in {time.time()-tic:.1f}s")
        except Exception as e:
            import traceback
            print(f"  [{i}/{len(pose_jsons)}] {pj.stem}: ERROR {e!r}")
            traceback.print_exc()
            summary.append({"video": pj.stem, "error": repr(e)})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone in {time.time()-t0:.1f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()

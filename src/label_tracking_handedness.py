"""Annotate L/R hand labels onto tracking videos using SAM 3 with explicit
"wearer left/right hand" text prompts.

For each video in a tracking-output dir (default outputs/track_v<N>/), this:
  1. Loads <stem>_track.frames.json (per-frame COCO-RLE masks per obj_id).
  2. Samples up to N_SAMPLE_FRAMES evenly across the video.
  3. At each sampled frame, runs SAM 3 with the prompts SAM3_LEFT_PROMPT and
     SAM3_RIGHT_PROMPT. For each prompt, the top-scoring detection (above
     SAM3_SCORE_THRESHOLD) is associated with the tracker's nearest obj_id by
     bbox-centroid distance. Confidence scores are summed into a per-(obj_id,
     L/R) cell.
  4. Final assignment uses joint-max on the confidence sums: for the two-
     obj_id case (the wearer's two hands are anatomically opposite), the
     (Left, Right) assignment between obj_0 and obj_1 that maximizes the
     joint confidence sum is picked.
  5. Re-renders the overlay video as <stem>_track_labeled.mp4 with mask fill,
     bbox, convex hull, and 'L'/'R' label at the centroid of each hand.
  6. Writes the handedness map back into <stem>_track.json
     (`wearer_handedness_by_obj_id`).

Replaces an earlier MP HandLandmarker-based labeler. MP was unreliable on
ego-centric views (frequently mirrored or unable to detect at all when the
hand is gloved); SAM 3 with text prompts directly identifies the wearer's
anatomical L/R regardless of where the hand appears on screen.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from transformers import Sam3Model, Sam3Processor

SAM3_MODEL_ID = "facebook/sam3"
SAM3_LEFT_PROMPT = "wearer left hand"
SAM3_RIGHT_PROMPT = "wearer right hand"
SAM3_SCORE_THRESHOLD = 0.30      # accept SAM 3 detections at >= this score
SAM3_MASK_THRESHOLD = 0.30
SAM3_MATCH_MAX_DIST_FRAC = 0.20  # SAM 3 detection -> obj_id match must be within this * diag
N_SAMPLE_FRAMES = 24             # how many frames to sample for handedness voting
MIN_VOTES_TO_LABEL = 1           # require at least this many votes per obj_id

# Visualization
MASK_COLORS_BGR = [(255, 80, 0), (0, 80, 255)]
HULL_COLORS_BGR = [(255, 220, 0), (0, 220, 255)]
LABEL_BG_COLOR = (0, 0, 0)
LABEL_FG_COLOR = (0, 255, 255)


def load_sam3(device: str):
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, device_map=device).eval()
    return proc, model


def decode_mask_rle(rle: dict) -> np.ndarray:
    raw = {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii")}
    return coco_mask.decode(raw).astype(bool)


@torch.no_grad()
def sam3_detect(processor, model, image: Image.Image, device: str, prompt: str):
    """Return (top_box_xyxy, top_score) for the highest-confidence SAM 3 hit
    above SAM3_SCORE_THRESHOLD, or None."""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs,
        threshold=SAM3_SCORE_THRESHOLD,
        mask_threshold=SAM3_MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    boxes_t = res["boxes"]
    scores_t = res["scores"]
    if len(boxes_t) == 0:
        return None
    boxes = boxes_t.cpu().numpy() if hasattr(boxes_t, "cpu") else np.asarray(boxes_t)
    scores = scores_t.cpu().numpy() if hasattr(scores_t, "cpu") else np.asarray(scores_t)
    top = int(np.argmax(scores))
    return boxes[top], float(scores[top])


def match_to_obj_id(
    detection_bbox: np.ndarray,
    obj_bboxes: dict[int, list[float]],
    image_diag: float,
) -> int | None:
    """Return the obj_id whose tracker bbox centroid is nearest to the SAM 3
    detection bbox centroid, provided the distance is within
    SAM3_MATCH_MAX_DIST_FRAC * image_diag. Returns None if no obj_id is close
    enough."""
    if not obj_bboxes:
        return None
    dcx = (detection_bbox[0] + detection_bbox[2]) / 2.0
    dcy = (detection_bbox[1] + detection_bbox[3]) / 2.0
    best_oid = None
    best_d = float("inf")
    for oid, bb in obj_bboxes.items():
        if bb is None:
            continue
        ocx = (bb[0] + bb[2]) / 2.0
        ocy = (bb[1] + bb[3]) / 2.0
        d = float(np.hypot(dcx - ocx, dcy - ocy))
        if d < best_d:
            best_d = d
            best_oid = oid
    if best_oid is None:
        return None
    if best_d > SAM3_MATCH_MAX_DIST_FRAC * image_diag:
        return None
    return best_oid


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
    sam3_proc, sam3_model,
    device: str,
    n_samples: int = N_SAMPLE_FRAMES,
    sam3_video_path: Path | None = None,
) -> tuple[dict[int, str], dict[int, dict[str, float]], dict[int, dict[str, int]]]:
    """Sample frames; at each, run SAM 3 with `SAM3_LEFT_PROMPT` and
    `SAM3_RIGHT_PROMPT` and match each detection to the nearest tracker
    obj_id by centroid distance. Returns the final L/R assignment plus
    confidence-weighted vote sums and raw counts.

    If `sam3_video_path` is given and differs from `video_path`, SAM 3 reads
    its inputs from sam3_video_path (typically source resolution) while
    obj_id matching stays in the downsampled coordinate frame of
    `video_path` (where the tracker bboxes live). SAM 3 detection bboxes are
    scaled from source-res to downsampled-res before matching.

    Final assignment uses joint-max on confidence sums for the two-obj_id
    case (the wearer's two hands are anatomically opposite, so the (Left,
    Right) pair must be different).
    """
    frames = frames_meta["frames"]
    W, H = frames_meta["size"]
    image_diag = float(np.hypot(W, H))
    n_frames = len(frames)
    if n_frames == 0:
        return {}, {}, {}
    sample_idxs = sorted(set(np.linspace(0, n_frames - 1, min(n_samples, n_frames)).round().astype(int).tolist()))
    needed = set(sample_idxs)
    score_sums: dict[int, dict[str, float]] = {}
    counts: dict[int, dict[str, int]] = {}

    use_dual = sam3_video_path is not None and sam3_video_path != video_path
    cap = cv2.VideoCapture(str(video_path))
    cap_sam3 = cv2.VideoCapture(str(sam3_video_path)) if use_dual else None
    sam3_W, sam3_H = W, H
    if cap_sam3 is not None:
        sam3_W = int(cap_sam3.get(cv2.CAP_PROP_FRAME_WIDTH))
        sam3_H = int(cap_sam3.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fidx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        bgr_sam3 = bgr
        if cap_sam3 is not None:
            ok_s, bgr_s = cap_sam3.read()
            if ok_s:
                bgr_sam3 = bgr_s
        if fidx in needed:
            rgb_sam3 = cv2.cvtColor(bgr_sam3, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_sam3)
            frame_meta = frames[fidx]
            obj_bboxes: dict[int, list[float]] = {}
            for h in frame_meta["hands"]:
                if h.get("bbox") is None or h.get("mask_rle") is None:
                    continue
                obj_bboxes[int(h["obj_id"])] = h["bbox"]
            if obj_bboxes:
                for prompt, label in ((SAM3_LEFT_PROMPT, "Left"), (SAM3_RIGHT_PROMPT, "Right")):
                    det = sam3_detect(sam3_proc, sam3_model, img, device, prompt)
                    if det is None:
                        continue
                    det_bbox, det_score = det
                    if use_dual:
                        sx = W / float(sam3_W)
                        sy = H / float(sam3_H)
                        det_bbox = np.array([
                            det_bbox[0] * sx, det_bbox[1] * sy,
                            det_bbox[2] * sx, det_bbox[3] * sy,
                        ], dtype=np.float32)
                    matched_oid = match_to_obj_id(det_bbox, obj_bboxes, image_diag)
                    if matched_oid is None:
                        continue
                    sd = score_sums.setdefault(matched_oid, {"Left": 0.0, "Right": 0.0})
                    cd = counts.setdefault(matched_oid, {"Left": 0, "Right": 0})
                    sd[label] += det_score
                    cd[label] += 1
        fidx += 1
    cap.release()
    if cap_sam3 is not None:
        cap_sam3.release()

    handedness: dict[int, str] = {}
    eligible = [oid for oid, d in counts.items() if d["Left"] + d["Right"] >= MIN_VOTES_TO_LABEL]
    if len(eligible) == 2:
        a, b = sorted(eligible)
        sa, sb = score_sums[a], score_sums[b]
        score_AB = sa["Left"] + sb["Right"]
        score_BA = sa["Right"] + sb["Left"]
        if score_AB >= score_BA:
            handedness[a] = "left"
            handedness[b] = "right"
        else:
            handedness[a] = "right"
            handedness[b] = "left"
    else:
        for oid in eligible:
            d = score_sums[oid]
            if d["Left"] > d["Right"]:
                handedness[oid] = "left"
            elif d["Right"] > d["Left"]:
                handedness[oid] = "right"
    return handedness, score_sums, counts


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


def process_one(track_dir: Path, video_stem: str, source_dir: Path, sam3_proc, sam3_model, device: str) -> dict:
    video_path = source_dir / f"{video_stem}.mp4"
    frames_json = track_dir / f"{video_stem}_track.frames.json"
    meta_json = track_dir / f"{video_stem}_track.json"
    out_video = track_dir / f"{video_stem}_track_labeled.mp4"
    if not video_path.exists():
        return {"video": video_stem, "error": "source mp4 missing"}
    if not frames_json.exists():
        return {"video": video_stem, "error": "frames json missing"}
    # Dual-resolution SAM 3: if src_<stem>.mp4 lives next to <stem>.mp4 (the
    # tracker persists it when --max-short-edge is set), feed it to SAM 3 for
    # better text-prompt recall while keeping the rest of the pipeline in the
    # downsampled coordinate frame.
    sam3_video_path = source_dir / f"src_{video_stem}.mp4"
    if not sam3_video_path.exists():
        sam3_video_path = None

    frames_meta = json.loads(frames_json.read_text())
    handedness, score_sums, counts = determine_handedness(
        video_path, frames_meta, sam3_proc, sam3_model, device,
        sam3_video_path=sam3_video_path,
    )
    render_labeled_video(video_path, frames_meta, handedness, out_video)

    # Update meta json
    if meta_json.exists():
        meta = json.loads(meta_json.read_text())
        meta["wearer_handedness_by_obj_id"] = {str(k): v for k, v in handedness.items()}
        meta["handedness_votes_by_obj_id"] = {str(k): v for k, v in counts.items()}
        meta["handedness_score_sums_by_obj_id"] = {
            str(k): {kk: round(vv, 3) for kk, vv in v.items()} for k, v in score_sums.items()
        }
        meta_json.write_text(json.dumps(meta, indent=2))

    return {
        "video": video_stem,
        "wearer_handedness_by_obj_id": {str(k): v for k, v in handedness.items()},
        "votes": {str(k): v for k, v in counts.items()},
        "score_sums": {str(k): {kk: round(vv, 3) for kk, vv in v.items()} for k, v in score_sums.items()},
        "labeled_video": out_video.name,
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-dir", default=None, help="Tracking dir; default: latest outputs/track_v<N>")
    ap.add_argument("--source-dir", default="data", help="Source mp4 dir")
    ap.add_argument("--videos", nargs="*", default=None, help="video stems (e.g. rgb_10); default: all in track-dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"Loading {SAM3_MODEL_ID} on {args.device} ...")
    sam3_proc, sam3_model = load_sam3(args.device)
    print("SAM 3 loaded.")
    summary = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        tic = time.time()
        try:
            res = process_one(track_dir, stem, source_dir, sam3_proc, sam3_model, args.device)
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

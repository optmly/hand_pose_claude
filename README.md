# Hand Tracking and Hand Pose Estimation

Ego-centric hand detection, segmentation, tracking, and pose estimation.

Pipeline stages, in order:

1. **Tracking** ([src/track_video_sam2.py](src/track_video_sam2.py)):
   per-video hand mask tracking. SAM 3 seeds at paired frames every 5 s,
   SAM 2 propagates forward and backward, with identity preserved via
   Hungarian matching, wrist-based mask trimming, seed verification, and
   a mask-collapse guard. The seed-verification gate scores the top-3
   substantial CCs (each ≥20% of total) and accepts if any one is
   hand-shaped, so multi-piece masks (hand + arm separated by a
   wristband; clean hand + stray pixels) still pass. The post-tracking
   mask cleanup retains the largest CC plus any substantial proximal CC
   (area ≥5% of total AND bbox-gap ≤10% of the largest CC's diagonal) —
   tool/utensil-separated fingers stay in the mask instead of being
   discarded. SAM 3 runs on the source-resolution frame and SAM 2 on a
   downsampled copy when `--max-short-edge` is set. Use `--output-dir`
   to write a per-video subprocess loop into one dir (avoids SAM 2's
   frame-loader OOM on long batches).
2. **L/R labeling** ([src/label_tracking_handedness.py](src/label_tracking_handedness.py)):
   samples frames, queries SAM 3 with the explicit prompts `"wearer
   left hand"` / `"wearer right hand"`, matches each top detection to
   the nearest tracker obj_id, and picks the (L, R) assignment that
   maximizes the joint confidence sum. Reads `src_<stem>.mp4` next to
   `<stem>.mp4` for source-res SAM 3 when present.
3. **Pose** ([src/pose_video_v2.py](src/pose_video_v2.py)): five-stage
   cascade per (frame, obj_id). Per-frame pre-check: if the mask is
   missing / below `MIN_MASK_AREA_PX`, skip with reason
   `mask_too_small_or_missing`. If the mask is small
   (area < 0.5% of screen) AND touches a frame border, skip with
   reason `small_edge_mask` (partial hand entering/exiting view).
   Otherwise run:
   1. **MP video VIDEO mode** at 1024 short-edge, gated by hull
      checks (wrist near 20%-expanded hull, ≥50% kpts inside hull,
      ≥50% kpts inside the actual mask, pose-hull / mask-hull area
      ∈ [0.30, 2.5]). If the area ratio is < 0.45
      (`MP_VIDEO_MIN_AREA_RATIO`) the VIDEO-tracked pose is
      considered compressed (tucked fingers held over from prior
      frame) and the cascade falls through.
   2. **MP video IMAGE mode** (no tracking) on the same frame; same
      0.45 hull-area gate.
   3. **MP image rerun** on the tight mask-zeroed crop.
   4. **MP image rerun wide** — 75%-padded square crop, no background
      zeroing. Gives MP enough context for closed-fist / dorsal-view
      cases.
   5. **ViTPose-Huge wholebody** (`--vitpose`) for the matching L/R
      side, gated against the same hull and mask checks, plus a mean
      heatmap-peak floor (`VITPOSE_MIN_MEAN_SCORE`). Runs lazily — only
      on frames that reach this stage (memoized per frame), never
      eagerly over the whole clip — and requires a wearer L/R label.
   Every detector candidate must pass all hull/mask gates AND
   `gate_temporal_jump` (`base × √(1 + prev_age)`); first to pass
   wins. Carryforward (≤10 frames) snaps the held pose to the current
   mask hull centroid + diagonal. After the loop, a post-pass
   linearly interpolates compressed-MP-video runs of length ≤7
   bracketed by uncompressed mp_video accepts, overriding the
   cascade output for those frames.
4. **Smoothing** ([src/kalman_smooth_pose.py](src/kalman_smooth_pose.py)):
   per-(keypoint, axis) RTS Kalman smoother (42 univariate filters per
   hand). Carryforward / vitpose frames carry a per-kp confidence that
   weights measurement noise so the smoother trusts the model more on
   uncertain frames. Post-smooth, three conditions NaN out the
   smoothed pose so the rendered overlay and JSON show no pose:
   (a) gaps > 5 frames where the mask centroid moved > 5% of image
   diagonal, (b) frames the pose stage flagged `small_edge_mask`,
   (c) frames where the smoothed kpts fall < 50% inside the actual
   tracker mask. Writes two videos per source:
   `<stem>_pose_smooth.mp4` (full overlay) and
   `<stem>_pose_smooth_clean.mp4` (no hull, lighter mask α=0.20).

## Requirements

- Python 3.11 (the project's conda env)
- CUDA-capable GPU (RTX 4090 in dev)
- HuggingFace account with approved access to `facebook/sam3` (gated)
- `mediapipe>=0.10.35`, `transformers>=5.9`, `sam2`, `pycocotools`,
  `scipy`, `opencv-python`, `decord`, `torch>=2.7+cu12.x`
- MediaPipe hand_landmarker.task at one of:
  - `~/.cache/mediapipe/hand_landmarker.task`
  - `models/hand_landmarker.task`
- ViTPose-Huge wholebody checkpoint (only when `--vitpose` is passed
  at pose time):

      ~/.cache/huggingface/hub/models--JunkyByte--easy_ViTPose/snapshots/*/torch/wholebody/vitpose-h-wholebody.pth

  Override with `--vitpose-ckpt /path/to/vitpose-h-wholebody.pth`.

## Setup

```bash
git clone git@github.com:optmly/hand_pose_claude.git
cd hand_pose_claude

# place .mp4 inputs in data/ (data/ is git-ignored)
ls data/rgb_*.mp4 | head

# log in to HuggingFace (needs approval for facebook/sam3)
huggingface-cli login
```

## Run

```bash
# 1) Track — per-video subprocess loop so SAM 2's frame loader is
#    freed between videos. --max-short-edge persists both the
#    downsampled and source-resolution truncated mp4s under inputs/.
OUT=outputs/release_50_v3/track_v1
mkdir -p "$OUT"
for v in /mnt/data/ws/nv-data/full/rgb_*.mp4; do
    STEM=$(basename "$v" .mp4)
    [ -f "$OUT/${STEM}_track.json" ] && continue
    python src/track_video_sam2.py \
        --videos "$v" --max-sec 30 --max-short-edge 1024 --output-dir "$OUT"
done

# 2) Label tracks with wearer L / R. Auto-detects src_<stem>.mp4 in
#    source-dir for higher-recall SAM 3.
python src/label_tracking_handedness.py \
    --track-dir outputs/release_50_v3/track_v1 \
    --source-dir outputs/release_50_v3/track_v1/inputs

# 3) Pose: full cascade with ViTPose backup.
#    NOTE: --source-dir MUST be the tracker's <track_dir>/inputs (the
#    downsampled videos), NOT the full-res dataset — the pose + smoother
#    stages now fail fast if the source resolution doesn't match the
#    tracker masks.
python src/pose_video_v2.py \
    --track-dir outputs/release_50_v3/track_v1 \
    --source-dir outputs/release_50_v3/track_v1/inputs \
    --num 50 --max-sec 30 \
    --output-base outputs/release_50_v3 \
    --vitpose

# 4) Kalman smoother (writes regular + clean smoothed mp4s).
python src/kalman_smooth_pose.py \
    --pose-dir outputs/release_50_v3/pose_v1 \
    --track-dir outputs/release_50_v3/track_v1 \
    --source-dir outputs/release_50_v3/track_v1/inputs
```

Each tracker run writes, per video:

- `<stem>_track.mp4` — overlay video (mask + bbox per obj_id, frame
  numbers, "SEED" markers on reseed frames)
- `<stem>_track.json` — prompt + pass used, full-collapse check,
  reseed log, pair structure, backward overrides, wrist-trim stats,
  seed-verification log (per-candidate solidity / aspect),
  mask-collapse resolution, mask-spike filter log, mask-cleanup
  multi-CC retention stats
- `<stem>_track.frames.json` — per-frame masks (COCO RLE) + bboxes
  per obj_id
- `<stem>_track_labeled.mp4` — labeler overlay with L / R tags
- `inputs/<stem>.mp4`, `inputs/src_<stem>.mp4` — downsampled +
  source-res truncated inputs (when `--max-short-edge` is set)

Each pose run writes:

- `<stem>_pose.mp4` — overlay with mask, hull polyline, skeleton,
  L / R label, source tag, frame number
- `<stem>_pose.json` — per-frame keypoints + flags (`source`,
  `rejected_reason`, `mp_score`, `kp_confidences`,
  `wearer_handedness`, `gate_diag`, `wide_gate_diag`,
  `video_image_gate_diag`, `vitpose_gate_diag`,
  `vitpose_mean_score`, `mp_video_compressed_ratio` when applicable)

Each smoother run writes (per video):

- `<stem>_pose_smooth.mp4` — full overlay with mask, hull, smoothed
  skeleton, source tag
- `<stem>_pose_smooth_clean.mp4` — no convex hull, lighter mask
  (α=0.20), skeleton only
- `<stem>_pose_smooth.json` — per-frame smoothed keypoints +
  `had_measurement` flag

## Version

Current release identifier: [VERSION](VERSION). Full history:
[CHANGELOG.md](CHANGELOG.md).

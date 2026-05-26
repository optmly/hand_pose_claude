# Hand Tracking and Hand Pose Estimation

Ego-centric hand detection, segmentation, tracking, and pose estimation.

Pipeline stages, in order:

1. **Tracking** ([src/track_video_sam2.py](src/track_video_sam2.py)):
   per-video hand mask tracking. SAM 3 seeds at paired frames every 5 s,
   SAM 2 propagates the masks forward and backward, with identity preserved
   via Hungarian matching, wrist-based mask trimming, seed verification,
   and a mask-collapse guard. The seed-verification gate scores the top
   connected components (>= 20% of the candidate mask, up to 3) and accepts
   if any one looks hand-like, so multi-piece masks (e.g. arm + hand
   separated by a wristband, or a clean hand + a few stray pixels) still
   pass. SAM 3 runs on the source-resolution frame and SAM 2 on a
   downsampled copy when `--max-short-edge` is set.
2. **L/R labeling** ([src/label_tracking_handedness.py](src/label_tracking_handedness.py)):
   samples frames, queries SAM 3 with the explicit prompts `"wearer left
   hand"` / `"wearer right hand"`, matches each top detection to the
   nearest tracker obj_id, and picks the (L, R) assignment that maximizes
   the joint confidence sum. Reads `src_<stem>.mp4` next to `<stem>.mp4`
   for source-res SAM 3 when present.
3. **Pose** ([src/pose_video_v2.py](src/pose_video_v2.py)): MediaPipe
   HandLandmarker in VIDEO mode at a single 1024-short-edge frame, gated
   against the tracker mask convex hulls (expanded by 20% for the
   kpts-in-hull check). Two MP image fallbacks on failure: tight
   mask-zeroed crop, then a 75%-expanded square crop with no zeroing.
   ViTPose-Huge wholebody (`--vitpose`) is the final backup, used only
   when MP exhausts and a wearer L/R label is known. Carryforward holds
   the last accepted pose for up to 10 frames, snapped to the current
   mask hull centroid + diagonal.
4. **Smoothing** ([src/kalman_smooth_pose.py](src/kalman_smooth_pose.py)):
   per-(keypoint, axis) RTS Kalman smoother (42 univariate filters per
   hand). Carryforward / vitpose frames carry a per-kp confidence that
   weights measurement noise so the smoother trusts the model more on
   uncertain frames.

## Requirements

- Python 3.11 (the project's conda env)
- CUDA-capable GPU (RTX 4090 in dev)
- HuggingFace account with approved access to `facebook/sam3` (gated)
- `mediapipe>=0.10.35`, `transformers>=5.9`, `sam2`, `pycocotools`, `scipy`,
  `opencv-python`, `decord`, `torch>=2.7+cu12.x`
- MediaPipe hand_landmarker.task at one of:
  - `~/.cache/mediapipe/hand_landmarker.task`
  - `models/hand_landmarker.task`
- ViTPose-Huge wholebody checkpoint (only required when `--vitpose` is
  passed at pose time):

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
# 1) Track: seeds + SAM 2 propagation for all videos in data/.
#    Auto-versions to outputs/track_v<N>/. --max-short-edge persists both
#    the downsampled and source-resolution truncated mp4s under inputs/.
python src/track_video_sam2.py \
    --videos /mnt/data/ws/nv-data/full/rgb_01.mp4 /mnt/data/ws/nv-data/full/rgb_18.mp4 \
    --max-sec 30 \
    --max-short-edge 1024

# Process all 50 source clips, 30 s each:
python src/track_video_sam2.py \
    --videos /mnt/data/ws/nv-data/full/rgb_*.mp4 \
    --max-sec 30 \
    --max-short-edge 1024

# 2) Label tracks with wearer L / R. Auto-detects src_<stem>.mp4 in
#    source-dir for higher-recall SAM 3.
python src/label_tracking_handedness.py \
    --track-dir outputs/track_v<N> \
    --source-dir outputs/track_v<N>/inputs

# 3) Pose: MP video + tight MP image rerun + wide MP image rerun, then
#    ViTPose backup. Writes <stem>_pose.{mp4,json} to outputs/pose_v<N>/.
python src/pose_video_v2.py \
    --track-dir outputs/track_v<N> \
    --source-dir outputs/track_v<N>/inputs \
    --num 20 \
    --max-sec 30 \
    --vitpose

# 4) Kalman smoothing on the pose JSONs. Writes <stem>_pose_smooth.{mp4,json}
#    to outputs/pose_v<N>_smooth/.
python src/kalman_smooth_pose.py \
    --pose-dir outputs/pose_v<N> \
    --track-dir outputs/track_v<N> \
    --source-dir outputs/track_v<N>/inputs
```

Each tracker run writes, per video:

- `<stem>_track.mp4` -- overlay video (one mask + bbox per obj_id, with
  frame numbers)
- `<stem>_track.json` -- which prompt + pass was used, full-collapse
  check, reseed log, pair structure, backward overrides, wrist-trim
  stats, seed-verification log (with per-candidate solidity / aspect),
  mask-collapse resolution, mask-spike filter log
- `<stem>_track.frames.json` -- per-frame masks (COCO RLE) + bboxes per
  obj_id
- `<stem>_track_labeled.mp4` -- post-label overlay with L / R tags
  (after `label_tracking_handedness.py`)
- `inputs/<stem>.mp4` -- downsampled truncated input (when
  `--max-short-edge` is set; downstream coord frame)
- `inputs/src_<stem>.mp4` -- source-res truncated input (when
  `--max-short-edge` is set; for SAM 3 consumers)

Each pose run writes:

- `<stem>_pose.mp4` -- overlay with mask, hull polyline, MP / ViTPose
  skeleton, L / R label, source tag, frame number
- `<stem>_pose.json` -- per-frame keypoints + flags (`source`,
  `rejected_reason`, `mp_score`, `kp_confidences`, `wearer_handedness`,
  `gate_diag`, `wide_gate_diag`, `vitpose_gate_diag`,
  `vitpose_mean_score`)

Each smoother run writes:

- `<stem>_pose_smooth.mp4` -- overlay with the smoothed skeleton
- `<stem>_pose_smooth.json` -- per-frame smoothed keypoints + the
  `had_measurement` flag

## Version

Current release identifier: [VERSION](VERSION). Full history:
[CHANGELOG.md](CHANGELOG.md).

# hand_pose_claude

Ego-centric hand detection, segmentation, tracking, and pose estimation.

The pipeline runs in two stages:

1. **Tracking** ([src/track_video_sam2.py](src/track_video_sam2.py)):
   per-video hand mask tracking. SAM 3 seeds at paired frames every 5 s,
   SAM 2 propagates the masks forward and backward, with identity preserved
   via Hungarian matching, wrist-based mask trimming, seed verification,
   and a mask-collapse guard. After the pass finishes, a full-screen-collapse
   detector (both obj_ids on the same blob, bbox + mask IoU >= 0.95) can
   trigger an automatic retry with the alternate SAM 3 prompt
   `"egocentric first person's hands"`. Finally a mask-spike filter
   replaces short-run mask anomalies (centroid jump or >2x area change that
   recovers within ~20 frames) with the previous frame's mask.
2. **Pose** ([src/pose_video_mp.py](src/pose_video_mp.py)): MediaPipe
   HandLandmarker in VIDEO mode, with a hull-area size gate against the
   tracker masks, a 50%-expanded-square image-mode rerun for failures,
   and a frame-to-frame consistency filter.

A standalone post-process ([src/label_tracking_handedness.py](src/label_tracking_handedness.py))
adds L/R wearer-anatomical labels to an existing tracking output by sampling
frames and majority-voting MP handedness per obj_id.

## Requirements

- Python 3.11 (the project's conda env)
- CUDA-capable GPU (RTX 4090 in dev)
- HuggingFace account with approved access to `facebook/sam3` (gated)
- `mediapipe>=0.10.35`, `transformers>=5.9`, `sam2`, `pycocotools`, `scipy`,
  `opencv-python`, `decord`, `torch>=2.7+cu12.x`
- MediaPipe hand_landmarker.task at one of:
  - `~/.cache/mediapipe/hand_landmarker.task`
  - `models/hand_landmarker.task`

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
#    Auto-versions to outputs/track_v<N>/.
python src/track_video_sam2.py

# Optional: just a single video, reverse order, etc.
python src/track_video_sam2.py --videos data/rgb_01.mp4 data/rgb_18.mp4
python src/track_video_sam2.py --reverse

# 2) Label tracks with L / R wearer-anatomical labels (writes <stem>_track_labeled.mp4)
python src/label_tracking_handedness.py --track-dir outputs/track_v<N>

# 3) Pose estimation on the first N videos using the tracker masks
python src/pose_video_mp.py --num 10 --track-dir outputs/track_v<N> --output-base outputs
```

Each tracker run writes, per video:

- `<stem>_track.mp4` -- overlay video (one mask + bbox per obj_id)
- `<stem>_track.json` -- reseed log, pair structure, backward overrides,
  wrist-trim stats, seed-verification log, mask-collapse resolution
- `<stem>_track.frames.json` -- per-frame masks (COCO RLE) + bboxes per obj_id

Each pose run writes:

- `<stem>_pose.mp4` -- overlay with mask, hull polyline, MP skeleton, L/R label
- `<stem>_pose.json` -- per-frame keypoints + flags (`source`,
  `rejected_reason`, `size_gate`, `mp_score`, `wearer_handedness`)

## Version

Current release identifier: [VERSION](VERSION). Full history: [CHANGELOG.md](CHANGELOG.md).

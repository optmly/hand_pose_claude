# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions use a zero-padded four-digit scheme starting from `0001`.

## [Unreleased]

## [0006] - 2026-05-22

### Labeler -- SAM 3 with "wearer left/right hand" prompts
- `src/label_tracking_handedness.py` rewritten to use SAM 3 with explicit
  text prompts `"wearer left hand"` and `"wearer right hand"` instead of
  MediaPipe HandLandmarker. At each sampled frame, the top SAM 3 detection
  for each prompt is matched to the nearest tracker obj_id by bbox-centroid
  distance (within 20% of image diagonal). Confidence-weighted joint-max
  assignment picks the (Left, Right) pair between obj_0 / obj_1 that
  maximizes the total SAM 3 confidence sum.
- Fixes prior MP-based labeler failures:
  - rgb_10: MP voted majority-Right on both obj_ids, producing two right
    hands. SAM 3 directly assigns the correct {0: left, 1: right}.
  - rgb_14: MP returned no detections (black gloves -- no skin tones for
    HandLandmarker). SAM 3 succeeds.
  - rgb_38, rgb_47: MP's per-frame handedness output was noisy and the
    wrong-direction votes summed higher; SAM 3's explicit L/R prompts
    return consistent labels regardless of on-screen position (so a
    wearer's left hand reaching to the right side of the screen is still
    labeled left).

## [0005] - 2026-05-22

### Tracker robustness
- Full-screen collapse detection + fallback-prompt retry
  (`detect_full_collapse`): after the tracking pipeline finishes, the tracker
  scans every frame for cases where obj_0 and obj_1 have both bbox IoU and
  mask IoU >= 0.95. If at least 3 frames trigger, the entire video is
  re-tracked with the prompt `"egocentric first person's hands"` at lower SAM
  3 thresholds (0.20 vs 0.50). The pass with fewer collapse frames is kept.
  Replaces the rgb_33-style in-pass salvage with a cleaner end-to-end retry
  (rgb_33).
- Mask-spike filter (`filter_mask_spikes`): detects short-run (<= 20-frame)
  mask anomalies and replaces them with the previous frame's mask. Spike
  triggers on either centroid jump (> 10% of image diagonal) or area change
  > 2x in either direction. Recovery checks only the dimension(s) that
  triggered: area-only spikes (rgb_21 obj_0 f37-47, where the mask inflated
  to the entire arm) do NOT require the centroid to return to the pre-spike
  position, since the hand may have moved during the spike. Seed frames are
  eligible for replacement, catching cases where SAM 3 re-seeded onto the
  wrong location (rgb_09 f50 cross-screen jump).
- Refactor: extracted `_run_tracking_pass(prompt, score_threshold,
  mask_threshold)` from `track_one_video` so the same tracking sequence can be
  invoked twice (primary + fallback). `resolve_mask_collapse` was moved out of
  the per-pass body into the outer orchestrator so `detect_full_collapse` sees
  the raw obj_0/obj_1 overlap (rather than the already-cleaned masks where
  the smaller of the two was zeroed).

## [0004] - 2026-05-22

### Tracker robustness
- `mask_bbox` now returns the bounding box of the largest connected component
  rather than the bbox of all mask pixels. Eliminates the random bbox jumps
  caused by tiny stray pixels SAM 2 occasionally emits far from the main hand
  blob (rgb_34).
- Mid-stream lost-track redetection
  (`find_lost_track_segments` + `redetect_lost_tracks`): after forward and
  backward propagation, the tracker samples every 15 frames and runs MP on
  each obj_id's mask region. When MP cannot place a wrist or palm-base
  landmark inside the mask for at least 2 consecutive sampled frames, the
  obj_id is marked lost. SAM 3 is run on the start-of-run frame; a SAM 3
  candidate that does not overlap the *other* obj_id's mask is wrist-trimmed
  and fed to SAM 2 via `add_new_mask`. SAM 2 is then re-propagated forward
  from the earliest redetection. Recovers cases where SAM 2 drifted onto a
  non-hand object between scheduled reseeds (rgb_33).
- `redetect_lost_tracks` carefully splits SAM 3 calls (outside the bfloat16
  autocast) from SAM 2 mask injection + re-propagation (inside autocast), so
  SAM 3's bfloat16 tensor outputs don't hit `.cpu().numpy()` directly.

## [0003] - 2026-05-22

### Added
- `src/detect_hands_demo.py`: per-frame side-by-side comparison of Grounding DINO
  vs SAM 3 hand detection on sample frames.
- `src/detect_hands_pipeline.py`: GD-count cascade (GD == 2 -> use GD + SAM 2
  masks; else SAM 3 fallback + top-2 by score). Per-hand MediaPipe HandLandmarker
  keypoints, mask convex-hull overlay, wearer-anatomical L/R labels via MP
  majority vote.
- `src/track_video_sam2.py`: video tracking pipeline. SAM 3 seeds at paired
  frames every 5 s ((0, 50), (150, 200), ...), masks fed to SAM 2 video
  predictor via `add_new_mask`. Forward + backward propagation, identity
  preserved across seeds via Hungarian (scipy `linear_sum_assignment`) centroid
  matching. Per-frame COCO-RLE mask output JSON. Reverse-order processing flag.
- `src/pose_video_mp.py`: two-pass per-video pose estimator. Pass 1 runs MP
  HandLandmarker VIDEO mode with size gate (asymmetric `MP_hull / mask_hull` in
  [0.30, 1.25]), image-mode rerun for failures (50% expanded square crop),
  consistency filter (10% image-diagonal jump cap). Pass 2 renders overlay
  video with L/R labels (MP majority vote per obj_id), mask convex-hull
  polyline, and 21-keypoint skeleton.
- `src/label_tracking_handedness.py`: post-process that adds L/R wearer-
  anatomical labels to a tracking-only run via MP image-mode on sampled
  frames, with pairwise-complement tie-break.

### Tracker robustness
- Hungarian matcher replaces greedy in `assign_obj_ids_by_match` -- fixes
  identity swaps when both hands undergo large symmetric motion (rgb_05).
- Wrist-based mask trim at seed time: MP HandLandmarker finds the wrist and
  palm direction; mask is cut perpendicular to palm direction at the wrist
  (shifted slightly toward forearm to keep a small buffer). Long-glove cases
  no longer include the gloved forearm in the mask (rgb_18).
- Extended backward propagation: backward pass runs between every pair of
  consecutive valid SAM 3 seeds, including the leading gap (frame 0 to first
  valid seed) when SAM 3 misses the early frames (rgb_06, rgb_16).
- Seed verification: each SAM 3 candidate must either have MP land its wrist
  or a palm-base landmark inside the candidate's mask, or pass a shape sanity
  check (solidity, aspect ratio). Filters obvious non-hand seeds.
- Mask-collapse guard: when two obj_ids' masks have IoU >= 0.30 OR centroid
  distance < 5% of the image diagonal for at least three consecutive frames,
  the smaller-area mask is zeroed until the next reseed. Catches "two masks
  on the same hand" after one obj_id loses its actual hand (rgb_23, rgb_33).
- BFloat16/Float dtype crash in SAM 2 `memory_attention` worked around by
  wrapping all SAM 2 video calls in `torch.autocast(cuda, bfloat16)`.

### Data
- `data/`: 50 ego-centric RGB videos (`rgb_01.mp4` through `rgb_50.mp4`,
  1920x1440 @ 30 fps, 4-10 s each). Not committed to git.

## [0002] - 2026-05-21

### Added
- Project-scoped Claude Code hook in `.claude/settings.json`. Fires before
  `git commit` and `git push` and reminds the agent to bump `VERSION`,
  prepend a `CHANGELOG.md` entry, and rewrite `README.md` to contain only
  the current setup/run instructions.

### Changed
- `README.md` rewritten to hold only the latest setup and run information.
  Project history and scaffolding details now live exclusively in
  `CHANGELOG.md`.

## [0001] - 2026-05-21

### Added
- Initial project scaffolding.
- `README.md` describing the project.
- `CHANGELOG.md` for tracking versioned changes.
- `VERSION` file recording the current release identifier.
- `.gitignore` covering common Python, editor, and OS artifacts.

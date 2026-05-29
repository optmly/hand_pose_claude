# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions use a zero-padded four-digit scheme starting from `0001`.

## [Unreleased]

## [0016] - 2026-05-29

Code-review hardening pass (no coverage change; correctness + safety +
perf). Verified on 50 videos × 30 s: accepted 67,089 / effective
(96.40%), missing 577 (0.83%) — statistically identical to 0015's run,
with cleaner masks and faster ViTPose.

### Fixed
- Tracker `redetect_lost_tracks` mask clobber
  ([src/track_video_sam2.py](src/track_video_sam2.py)): after re-seeding
  only the lost obj_id, the forward re-propagation did
  `masks_per_frame[fidx] = per_obj`, a full per-frame REPLACE. Since SAM 2
  `propagate_in_video` yields ALL tracked obj_ids, this overwrote the
  non-lost hand's already-merged forward+backward masks with plain
  forward-only propagation. Now writes back only the lost obj_id's masks
  (`.update()` with lost-oids only). On 20×30 s this preserved 1,219
  non-lost-hand frames across 9 videos (redetect fires on 17/20); both
  hands stay tracked everywhere, and the changed masks are tighter
  (less forearm bleed).

### Added
- Fail-fast source-resolution guards in the pose and smoother stages:
  if the source video's WxH doesn't match the tracker masks
  (`frames_meta["size"]`) — the classic "--source-dir points at full-res
  data instead of <track_dir>/inputs" mistake — raise/return a clear
  error instead of silently running every gate on a mismatched
  coordinate frame.
- Lazy ViTPose ([src/vitpose_runner.py](src/vitpose_runner.py)):
  ViTPose-Huge now runs on demand per (video, frame) and memoizes,
  instead of an eager full-clip forward + DARK decode. It is the
  cascade's last-resort backup (consulted only on frames where every MP
  path fails), so MP-dominated clips drop from N ViT forwards to a
  handful (e.g. rgb_01: 1 vs 384). The pose loop passes the
  already-decoded frame so there is no re-read. Output verified
  byte-identical to the eager path (0 source mismatches, max kpt diff
  0.000000 px over 1,517 ViTPose frames).
- ViTPose confidence floor (`VITPOSE_MIN_MEAN_SCORE = 0.05` in
  [src/pose_video_v2.py](src/pose_video_v2.py)): ViTPose never abstains,
  so a near-zero mean heatmap-peak score means no real hand — reject
  rather than feed the gates. (Replaces the old run_video docstring that
  claimed a floor the code never applied.)
- Handedness tie-break + empty-handedness warning
  ([src/label_tracking_handedness.py](src/label_tracking_handedness.py)):
  when exactly two hands are tracked but only one gets a confident L/R
  label, assign the other the opposite side (guarded on `seen_oids==2`
  so genuine single-hand clips are untouched); and the pose stage warns
  loudly if `--vitpose` is set with no handedness labels (backup would
  be silently inert).
- `cv2.VideoWriter.isOpened()` + positive-(W,H,fps) guards in all four
  render stages (tracker, labeler, pose, smoother): a failed codec /
  degenerate size silently dropped every frame while the stage reported
  success.

### Removed
- Dead code in [src/pose_video_v2.py](src/pose_video_v2.py): `bbox_area`,
  `kp_bbox`, `expand_to_square_crop`, `run_mp_image_crop`,
  `RERUN_BBOX_EXPAND` (all unreferenced after the hull-area gate switch).
- Legacy modules superseded since 0003 and referenced nowhere in the live
  pipeline: `src/detect_hands_demo.py`, `src/detect_hands_pipeline.py`,
  `src/pose_video_mp.py`.

### Deferred (tracked for a later pass; need isolated evaluation)
- Shared module + central config to de-dup ~50 thresholds and the
  decode/hull/render primitives copy-pasted across stages.
- One calibrated [0,1] per-keypoint confidence channel across detectors
  (an algorithm change that would alter Kalman weighting).
- Caching decoded masks/hulls in the render passes (perf; naive caching
  risks a multi-GB RAM regression).

## [0015] - 2026-05-28

### Added
- Small-edge-mask skip in
  [src/pose_video_v2.py](src/pose_video_v2.py): when the tracker mask
  is small (area < `SMALL_EDGE_MASK_AREA_FRAC` = 0.005 of the screen)
  AND touches any frame border, the frame is partial hand-in-view and
  pose estimation is skipped (`rejected_reason = "small_edge_mask"`,
  `keypoints = None`). New helper `is_small_edge_mask`.
- `gate_kpts_in_mask` enabled by default with
  `MASK_KPTS_MIN_FRAC = 0.50`: candidates (from any detector path)
  must have >= 50% of keypoints lying inside the actual tracker mask
  (not just the convex hull). Catches coworker-hand or otherwise
  outside-mask poses that geometrically fall inside the hull but
  don't overlap the wearer-mask pixels.
- Kalman post-smooth mask-containment NaN in
  [src/kalman_smooth_pose.py](src/kalman_smooth_pose.py)
  (`MIN_KPTS_IN_MASK_FRAC` = 0.50): after smoothing, any frame whose
  smoothed/interpolated keypoints fall less than 50% inside the
  actual tracker mask is NaN'd out so the rendered overlay and JSON
  show no pose for that frame.
- Kalman small_edge propagation: frames flagged
  `small_edge_mask` at the pose stage are also NaN'd out in the
  smoother so the rendered video doesn't extrapolate across them.

### Result (50 videos × 30 s, release_50_v3 re-run)
- Pose: 67,097 / 69,594 accepted (96.41%); 576 missing (0.83%);
  255 new `small_edge_mask` frames; 1,921 carryforward.
- Source mix: mp_video 84.6%, vitpose 5.9%, interpolated 3.6%,
  mp_image_rerun 2.6%, mp_video_image 2.1%, mp_image_rerun_wide
  1.1%.
- Kalman smoothed mp4s now show no pose on 2,081 frames where the
  pose either was a small-edge skip, hit the post-smooth
  mask-containment threshold, or fell in a gap-skip interval.

## [0014] - 2026-05-28

### Added
- Tracker: multi-CC mask preservation in
  [src/track_video_sam2.py](src/track_video_sam2.py). New
  `keep_substantial_proximal_ccs` (≥5% of total area AND within 10% of
  the largest CC's bbox diagonal) replaces the aggressive
  `keep_largest_cc` cleanup. Tool/utensil-separated fingers are now
  retained instead of being discarded. `mask_bbox` updated to use the
  same rule so the rendered bbox covers all kept CCs.
- Tracker `--output-dir` CLI flag for fixed output paths (per-video
  subprocess loops can now all write to one dir, avoiding SAM 2's
  frame-loader OOM on long-running invocations).
- Pose cascade refactor in
  [src/pose_video_v2.py](src/pose_video_v2.py): every detector path
  (mp_video → mp_video_image → mp_image_rerun → mp_image_rerun_wide →
  vitpose_huge) must pass both the hull gates AND temporal_jump; first
  to pass wins. Allows mp_image / wide / vitpose to rescue frames where
  the MP video pick failed the jump check.
- Gap-scaled temporal_jump threshold: base × √(1 + prev_age).
  Accommodates real hand motion across short stretches with no fresh
  detection without spurious jump rejections.
- MP image wide-crop fallback (`run_mp_image_wide`): square crop
  expanded by 75% around the mask, no background zeroing. Gives MP
  enough context for closed-fist / dorsal-view cases where the tight
  mask-zeroed crop is context-starved.
- 20%-expanded convex-hull check on `gate_kpts_in_hull` so fingertips
  extending a few px past the SAM mask hull pass without loosening the
  50%-in-hull threshold itself.
- Snap-to-mask carryforward: held pose is translated + scaled to align
  with the current mask hull centroid + diagonal. Rendered skeleton
  stays anchored to the actual hand instead of drifting at the stale
  frame's coordinates.
- Pose-size floor `POSE_BBOX_MIN_RATIO=0.30` (lower bound on
  pose-hull / mask-hull area ratio) rejects degenerate MP image
  detections that collapse all 21 kpts into a tiny cluster.
- MP video VIDEO mode "compression gate"
  (`MP_VIDEO_MIN_AREA_RATIO=0.45`): when the VIDEO-mode tracked pose
  has pose-hull / mask-hull area ratio < 0.45, the frame falls through
  the cascade. Catches the tucked-finger pose held from a prior
  grasping frame.
- MP video IMAGE mode (`run_mp_video_frame_image`) as the first
  cascade backup with the same 0.45 hull-area threshold. Provides a
  no-tracking re-detection for compressed VIDEO-mode outputs.
- Post-pass interpolation
  (`interpolate_short_compressed_runs`): runs of consecutive
  compressed MP-video frames ≤ `COMPRESSED_INTERP_MAX_LEN`=7 that are
  bracketed by uncompressed mp_video accepts get linearly interpolated
  from the bracketing keypoints, overriding the backup-cascade
  decision. Longer runs keep the cascade output.
- Hull-area helpers (`hull_area`, `mask_hull_area`) so size checks are
  scale-invariant against hand orientation (bbox-area was biased by
  axis alignment).
- `gate_kpts_in_mask` + `--mask-kpts-min-frac` CLI flag (disabled by
  default) for stricter coworker-hand filtering when needed.
- Kalman smoother
  [src/kalman_smooth_pose.py](src/kalman_smooth_pose.py): gap-skip
  rule — when a gap between accepted measurements is > 5 frames AND
  the mask centroid moved > 5% of image diagonal, the smoother's
  interpolation across the gap is NaN'd out (the gap is treated as
  missing instead of extrapolating across an unreliable stretch).
- Second smoothed video per source: `<stem>_pose_smooth_clean.mp4` —
  no convex hull, lighter mask alpha (0.20 vs 0.45), skeleton only.

### Fixed
- Pose: temporal_jump prev_pose reset when carryforward window
  exhausted. Previously prev_pose stayed pinned forever, causing
  every subsequent MP-video detection to be rejected as a jump
  against a many-frames-old anchor.

### Result (50 videos × 30 s, release_50_v3)
- Tracker: 50/50 succeeded, 50/50 both hands labeled.
- Pose: 68,263 / 69,849 accepted (97.73%); 1,268 carryforward
  (1.82%); 318 missing (0.46%). 29/50 videos have zero misses.
- Source mix: mp_video 84.4%, vitpose 6.4%, interpolated 3.6%,
  mp_image_rerun 2.5%, mp_video_image 2.0%, mp_image_rerun_wide
  1.0%. The new interpolation + IMAGE-mode paths contribute 3,810
  accepted poses (5.6% of total).

## [0013] - 2026-05-26

### Added
- Dual-resolution SAM 3 in [src/track_video_sam2.py](src/track_video_sam2.py):
  feed SAM 3 the source-resolution frame for seeding and lost-track
  redetection, while SAM 2 / MP / overlays run on a downsampled
  copy. Wired via `--max-short-edge` (also CLI) and the new
  `sam3_video_path` plumbing through `_run_tracking_pass`,
  `redetect_lost_tracks`, `track_one_video`, and `main`. Persists
  both `inputs/<stem>.mp4` (downsampled, downstream coord frame) and
  `inputs/src_<stem>.mp4` (source-res, for any subsequent SAM 3
  consumer).
- Dual-resolution SAM 3 in [src/label_tracking_handedness.py](src/label_tracking_handedness.py):
  auto-detects `src_<stem>.mp4` next to `<stem>.mp4` in the source
  dir and feeds it to SAM 3 while keeping obj_id matching in the
  downsampled coordinate frame (detection bboxes scaled back).
- Top-3-CC seed_is_hand_like gate in [src/track_video_sam2.py](src/track_video_sam2.py):
  for each SAM 3 candidate, evaluate the top 3 connected components
  whose area is at least `SEED_VERIFY_CC_AREA_FRAC` (=0.20) of the
  full mask, score each CC's solidity + aspect, and accept if any
  one is hand-shaped. Handles both noisy multi-CC masks (rgb_16:
  99% in 1 CC + 7 strays drop sol 0.86 -> 0.48) and SAM 3 fusing
  arm + hand into one box separated by a wristband.
- Wide-crop MP image fallback in [src/pose_video_v2.py](src/pose_video_v2.py):
  after the existing tight + mask-zeroed `run_mp_image_masked` step
  fails, `run_mp_image_wide` retries on a square crop expanded by
  `MP_WIDE_CROP_EXPAND_FRAC` (=0.75) of the mask bbox without
  background zeroing. Gives MP enough surrounding context for
  closed-fist / dorsal-view / small-mask cases where the tight
  zeroed crop has no recognizable hand structure. New source label
  `mp_image_rerun_wide` and corresponding `wide_gate_fail` /
  `wide_no_detection` reason chains.
- 20% expanded convex-hull check in
  [src/pose_video_v2.py](src/pose_video_v2.py):
  `gate_kpts_in_hull` now scales the mask convex hull about its
  centroid by `1 + KPTS_HULL_EXPAND_FRAC` (=0.20) before the
  point-in-polygon test, letting MP video detections whose
  fingertips extend a few px past the SAM-mask hull pass without
  having to loosen the 50%-in-hull threshold itself.
- Snap-to-mask carryforward in
  [src/pose_video_v2.py](src/pose_video_v2.py): when a frame falls
  through to carryforward, the held pose is translated so its
  centroid aligns with the current mask hull centroid and scaled so
  its bbox diagonal matches the hull diagonal. Keeps the rendered
  skeleton visually anchored to the actual hand instead of drawing
  at the stale frame's coordinates.
- Per-keypoint Kalman smoother [src/kalman_smooth_pose.py](src/kalman_smooth_pose.py):
  RTS (Rauch-Tung-Striebel) constant-velocity smoother run
  independently per (keypoint, axis) -- 42 univariate filters per
  hand. Optional per-frame keypoint confidence weights the
  measurement noise (`R_t = (sigma_m / max(conf, conf_floor))^2`),
  so carryforward / low-confidence frames get smoothed over more
  aggressively. Writes `<stem>_pose_smooth.{mp4,json}` to a fresh
  `outputs/pose_v<N>_smooth/`.

### Fixed
- Temporal-jump prev_pose reset in
  [src/pose_video_v2.py](src/pose_video_v2.py): when a fresh MP
  detection fails the temporal-jump gate AND the carryforward window
  is exhausted, also reset `prev_pose[oid] = None` (matching the
  no-detection branch). Without this, prev_pose stayed pinned to a
  many-frames-old pose forever and every subsequent MP-video
  detection was rejected as a jump against that stale anchor. On
  rgb_02 this single fix collapsed missing from 85 -> 4 frames
  (+76 accepted, +12.7 pp).

### Result (rgb_02 alone, 10 s @ 1024p, 598 effective hand-slots)
- baseline 0012: accepted 488 (81.6%), missing 85 (14.2%), of which
  temporal_jump=74.
- 0013: accepted 564 (94.3%), missing 4 (0.7%); only 9 temporal-jump
  rejections survive (median 279 px, all real moderate jumps).

## [0012] - 2026-05-24

### Added
- `src/vitpose_huge_wholebody.py`: pure-PyTorch ViTPose-Huge wholebody
  model definition (~240 lines, 133 keypoints; uses indices 91-111 for
  left hand and 112-132 for right hand).
- `src/vitpose_runner.py`: per-video ViTPose pass that decodes the
  48x64 heatmaps with UDP / DARK sub-pixel refinement (~10x more
  accurate than argmax+sign at this resolution), caches the result
  on the model, and exposes a `get_for_side(video, fidx, side)`
  lookup used by the pose pipeline.
- `pose_video_v2.py --vitpose`: enable ViTPose as the third source
  (after MP VIDEO and MP IMAGE rerun). Picked only when both MP
  attempts failed AND the obj_id has a known wearer L/R label; the
  ViTPose hand keypoints for that side then go through the same
  mask-hull / size-sanity gate suite. Runs lazily, once per video,
  and frees its frame cache after each video to keep working-set
  memory bounded.

### Result (20 clips at 10 s each, ~6000 frames per side)
- Left: mp_video 4716 (79%) + mp_image 94 + vitpose 631 (10.5%) +
  carryforward 100 -> accepted 5541 (92.4%), miss 147 (2.5%).
- Right: mp_video 4909 (82%) + mp_image 37 + vitpose 539 (9%) +
  carryforward 90 -> accepted 5575 (93%), miss 113 (1.9%).
- Black-gloved clips rgb_14 / rgb_15 went from 100% miss in 0011 to
  ~99% / ~90% L+R covered, entirely via ViTPose.

## [0011] - 2026-05-24

### Added
- `src/pose_video_v2.py`: new hand-pose pipeline that runs MediaPipe
  HandLandmarker (VIDEO mode, `num_hands=4`, conf 0.20) over each video
  and, per tracker `obj_id`, picks the best candidate that passes a
  gate suite anchored to the tracker mask convex hull (already computed
  by `track_video_sam2.py`). Gates:
  - **wrist-near-hull**: wrist landmark (lm 0) must be inside the mask
    hull OR within 25% of the hull's bbox diagonal of its border. The
    tracker masks are wrist-trimmed, so the wrist often sits just
    outside; we don't require wrist-IN-mask.
  - **kpts-in-hull**: >= 50% of the 21 keypoints must lie inside the
    mask hull.
  - **size sanity**: pose bbox area <= 2.5 * mask bbox area (catches
    blow-up skeletons when a hand goes partially off-screen).
  - **temporal jump**: per-kp mean displacement vs the most recently
    accepted pose for this obj_id <= 10% of the image diagonal.
- MP IMAGE-mode rerun on a 50%-expanded square crop around the mask
  bbox when MP VIDEO produces no qualifying candidate; same gate suite
  applied. Carry-forward of the previous accepted pose for up to 10
  frames covers brief rejections.
- L / R label per `obj_id` is read from the tracker's
  `wearer_handedness_by_obj_id` (SAM 3 labeler from 0006); MP's
  per-frame handedness is not used.

### Initial result (20 clips at 10 s each, ~6000 frames per side)
- Left: 4710 mp_video + 86 mp_image_rerun + 159 carryforward = 82%
  accepted, 12% miss.
- Right: 4909 + 37 + 82 = 84% accepted, 11% miss.
- rgb_14 / rgb_15 (black gloves) are at 100% miss -- MP can't see
  gloved hands. These need the ViTPose backup that will land in a
  follow-up release.

### Output
- `outputs/pose_v<N>/<stem>_pose.json` per-frame keypoints + gate
  diagnostics + source label (mp_video / mp_image_rerun / carryforward).
- `outputs/pose_v<N>/<stem>_pose.mp4` overlay video: tracker mask +
  mask hull polyline + 21-kp skeleton + L / R obj label + frame number.

## [0010] - 2026-05-24

### Docs
- README: marked the hand-tracking + L / R labeling stages as complete at
  the current release, and described the 0008-era tracker post-passes
  (largest-CC bbox in the renderer, mask cleanup, frame-number overlay,
  spike-filter collision avoidance, seed-frame label-swap fix). No code
  changes -- this is purely a follow-up bump to keep VERSION /
  CHANGELOG aligned with the documentation update that landed on top of
  0009.

## [0009] - 2026-05-24

### Hand tracking + L/R labeling FINAL

This is the version we consider done for hand tracking on the
`/mnt/data/ws/nv-data/full/` dataset (50 clips, capped at 30 s each):
per-video hand mask tracking via SAM 3 seeds + SAM 2 propagation, plus
SAM 3-based wearer L / R labeling. Future work on pose estimation will
build on this output.

### Tracker
- Disabled `filter_cross_person_hands` in `track_one_video` (over-pruned
  legitimate wearer-hand frames on multiple clips when triggered by a
  high-in-frame mask centroid that turned out to still be a wearer hand).
  The function is still defined so it can be re-enabled and tuned later.
  All other 0008 fixes remain: largest-CC bbox in the renderer,
  `cleanup_masks_largest_cc`, frame-number overlay, spike-filter collision
  avoidance, and `fix_seed_label_swaps`.

### Run on all 50 full-length clips (truncated to 30 s)
- 50 / 50 tracked, 50 / 50 labeled cleanly.
- 0 hand flips detected (centroid swap relative to f-1).
- 0 empty or two-same L / R label assignments.
- Per-video subprocess loop (one python invocation per input mp4) for
  memory isolation; tracker outputs in `outputs/track_v1/` (run via
  `bash /tmp/run_all_v8.sh`-style loop, see README for the recipe).

## [0008] - 2026-05-23

### Tracker -- visualization + post-pass robustness
- Render-bbox correctness: `overlay_masks` now draws bboxes from the largest
  connected component (matching `mask_bbox` / the JSON bbox) instead of
  `np.where(mask)` over all mask pixels. SAM 2 occasionally leaks a 1-3
  pixel stray CC far from the main hand blob (e.g., rgb_09 f58 had a 3-px
  stray at the bottom of the frame, left over from a SAM 3 seed there);
  the renderer was stretching the bbox to enclose it and showing a tall
  full-frame bbox even though the JSON was clean.
- `cleanup_masks_largest_cc` post-pass: drops stray CCs from each frame's
  mask before downstream collapse / spike / collision logic runs.
- Frame number overlay (`f<N>`) in the top-right of every rendered frame so
  it's easy to identify which frame a glitch is on.
- Spike-filter collision avoidance: refuse to replace obj_X's spike-frame
  mask with the prior-frame mask if doing so would create IoU >= 0.30 with
  the OTHER obj_id's mask at that frame. Without this, the spike filter's
  "smoothing" was creating 5-frame collapses at seed frames where Hungarian
  had swapped obj_ids (rgb_03 ~26s).
- Seed-frame label-swap fix: at each SAM 3 seed frame, if the new seed
  masks for obj_0 / obj_1 are closer to the OTHER obj_id's f-1 mask than
  to their own, swap labels at the seed frame only (1-frame correction;
  SAM 2 typically reverts via memory_attention at sf+1 so the local swap
  is enough). Resolves the rgb_03 "hand flip" at f350 / f450 / f500 /
  f600 / f650 / f750 / f800 caused by Hungarian misassignment after the
  wearer's hands crossed between seeds.
- Cross-person hand filter: zero masks whose centroid sits in the top
  20% of the frame for 5+ consecutive frames. In ego-centric video the
  wearer's hands rarely stay near the top of the frame; sustained top-
  of-frame masks are almost always SAM 2 drift onto a co-worker reaching
  across the workspace (rgb_17 obj_0 from f680+).

## [0007] - 2026-05-23

### Tracker -- handle longer / full-length videos
- Added `--max-sec` CLI flag (default 60). Any input mp4 longer than the cap
  is re-encoded via ffmpeg (`truncate_video_to_frames`) into a temp mp4 with
  exactly `--max-sec * fps` frames before being handed to the tracking
  pipeline. Used 30 s for the first pass on the full-length dataset at
  `/mnt/data/ws/nv-data/full/`.
- Adaptive SAM 2 memory handling for longer clips
  (`OFFLOAD_VIDEO_FRAME_THRESHOLD=600`, `OFFLOAD_STATE_FRAME_THRESHOLD=2000`,
  `JPEG_STREAMING_FRAME_THRESHOLD=1000`). Videos past the streaming threshold
  are pre-decoded to a `<tmp>/sam2_<stem>_*` JPEG folder so `init_state` can
  use `async_loading_frames=True`. The temp folder is cleaned up in a
  `try/finally` around the pass.
- `decode_video_to_jpegs` helper writes `00000.jpg` ... `<n>.jpg` (the naming
  SAM 2's `init_state(video_path=<dir>)` expects).
- Aggressive cleanup between videos: each `track_one_video` call is followed
  by `gc.collect()` + `torch.cuda.empty_cache()` to release any lingering
  `AsyncVideoFrameLoader` tensors, and truncated mp4s are deleted as soon as
  tracking finishes. Without this the process accumulates host-RAM use across
  videos and the kernel OOM-kills it around the 18th 60 s-cap video.
- For very long videos (e.g., rgb_43 at 243 s), even with the above the
  current SAM 2 frame-loader cache strategy still pushes host RAM past
  comfort. Running the tracker as a per-video subprocess loop (one python
  invocation per input mp4) plus a 30 s cap was the reliable way to clear
  all 50 full-length videos.

### Outputs
- `outputs/track_v<N>/` now contains 50 / 50 tracked + L/R-labeled clips for
  the full-length dataset (each truncated to 30 s).

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

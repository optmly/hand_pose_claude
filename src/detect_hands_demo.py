"""Compare Grounding DINO and SAM 3 for ego-centric hand detection.

For each sampled frame:
  - run Grounding DINO with prompt "a hand." -> boxes + scores
  - run SAM 3 with prompt "hand" -> masks + boxes + scores
  - save a 3-panel PNG: original | Grounding DINO overlay | SAM 3 overlay
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam3Model,
    Sam3Processor,
)

GD_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM3_MODEL_ID = "facebook/sam3"

GD_PROMPT = "a hand."
GD_BOX_THRESHOLD = 0.30
GD_TEXT_THRESHOLD = 0.25

SAM3_PROMPT = "hand"
SAM3_SCORE_THRESHOLD = 0.50
SAM3_MASK_THRESHOLD = 0.50

MASK_COLOR = np.array([0, 200, 255], dtype=np.uint8)  # cyan-ish
BOX_COLOR = (0, 255, 0)  # green for GD
SAM3_BOX_COLOR = (255, 128, 0)  # orange for SAM3


def load_models(device: str):
    gd_proc = AutoProcessor.from_pretrained(GD_MODEL_ID)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(GD_MODEL_ID).to(device).eval()
    sam3_proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    sam3_model = Sam3Model.from_pretrained(SAM3_MODEL_ID, device_map=device).eval()
    return gd_proc, gd_model, sam3_proc, sam3_model


@torch.no_grad()
def run_grounding_dino(processor, model, image: Image.Image, device: str):
    inputs = processor(images=image, text=GD_PROMPT, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=GD_BOX_THRESHOLD,
        text_threshold=GD_TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]
    return {
        "boxes": results["boxes"].cpu().numpy(),
        "scores": results["scores"].cpu().numpy(),
        "labels": list(results.get("text_labels", results.get("labels", []))),
    }


@torch.no_grad()
def run_sam3(processor, model, image: Image.Image, device: str):
    inputs = processor(images=image, text=SAM3_PROMPT, return_tensors="pt").to(device)
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=SAM3_SCORE_THRESHOLD,
        mask_threshold=SAM3_MASK_THRESHOLD,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results["masks"]
    boxes = results["boxes"]
    scores = results["scores"]
    return {
        "masks": masks.cpu().numpy().astype(bool) if hasattr(masks, "cpu") else np.asarray(masks).astype(bool),
        "boxes": boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes),
        "scores": scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores),
    }


def _draw_box(img: np.ndarray, box, color, label: str | None = None) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, max(th + 2, y1 - 4)), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def overlay_gd(frame_rgb: np.ndarray, gd: dict) -> np.ndarray:
    img = frame_rgb.copy()
    for box, score in zip(gd["boxes"], gd["scores"]):
        _draw_box(img, box, BOX_COLOR, f"hand {score:.2f}")
    return img


def overlay_sam3(frame_rgb: np.ndarray, sam3: dict) -> np.ndarray:
    img = frame_rgb.copy()
    masks = sam3["masks"]
    if masks.ndim == 4:
        masks = masks[0] if masks.shape[0] == 1 else masks.reshape(-1, masks.shape[-2], masks.shape[-1])
    elif masks.ndim == 3:
        pass
    else:
        masks = masks[None]
    overlay = img.copy()
    for i, m in enumerate(masks):
        if m.sum() == 0:
            continue
        color = np.array([(i * 67) % 256, (i * 113 + 80) % 256, (i * 197 + 160) % 256], dtype=np.uint8)
        overlay[m] = (0.45 * overlay[m] + 0.55 * color).astype(np.uint8)
    img = overlay
    for box, score in zip(sam3["boxes"], sam3["scores"]):
        _draw_box(img, box, SAM3_BOX_COLOR, f"hand {float(score):.2f}")
    return img


def make_panel(frame_rgb: np.ndarray, gd: dict, sam3: dict) -> np.ndarray:
    orig = frame_rgb.copy()
    gd_img = overlay_gd(frame_rgb, gd)
    sam3_img = overlay_sam3(frame_rgb, sam3)
    h, w = frame_rgb.shape[:2]
    bar = 60
    canvas = np.full((h + bar, w * 3 + 20, 3), 255, dtype=np.uint8)
    canvas[bar:, 0:w] = orig
    canvas[bar:, w + 10 : 2 * w + 10] = gd_img
    canvas[bar:, 2 * w + 20 : 3 * w + 20] = sam3_img

    def _title(text, x):
        cv2.putText(canvas, text, (x + 8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)

    _title(f"original  ({w}x{h})", 0)
    _title(f"Grounding DINO  ({len(gd['boxes'])} dets, thr {GD_BOX_THRESHOLD})", w + 10)
    n_masks = len(sam3["masks"]) if sam3["masks"].ndim >= 3 else 0
    _title(f"SAM 3  ({n_masks} masks, thr {SAM3_SCORE_THRESHOLD})", 2 * w + 20)
    return canvas


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--videos",
        nargs="+",
        default=["data/rgb_01.mp4", "data/rgb_05.mp4", "data/rgb_10.mp4"],
        help="Paths to mp4 files (relative to project root or absolute).",
    )
    ap.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[0, 60, 150, 250],
        help="Frame indices to sample from each video.",
    )
    ap.add_argument("--output", default="outputs/detect_hands_demo", help="Output directory.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Loading models ({GD_MODEL_ID}, {SAM3_MODEL_ID}) ...")
    gd_proc, gd_model, sam3_proc, sam3_model = load_models(args.device)
    print("Models loaded.")

    for vid in args.videos:
        vp = Path(vid) if Path(vid).is_absolute() else (root / vid)
        if not vp.exists():
            print(f"  skip (missing): {vp}")
            continue
        stem = vp.stem
        for idx in args.frames:
            try:
                frame = extract_frame(vp, idx)
            except RuntimeError as e:
                print(f"  {stem} frame {idx}: {e}")
                continue
            pil = Image.fromarray(frame)
            gd = run_grounding_dino(gd_proc, gd_model, pil, args.device)
            sam3 = run_sam3(sam3_proc, sam3_model, pil, args.device)
            panel = make_panel(frame, gd, sam3)
            out_path = out_dir / f"{stem}_f{idx:05d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(
                f"  {stem} f={idx:5d}: GD={len(gd['boxes'])} dets "
                f"(max={gd['scores'].max():.2f} if gd) | "
                f"SAM3={len(sam3['boxes'])} masks "
                f"(max={float(sam3['scores'].max()):.2f} if sam3) -> {out_path.name}"
                if (len(gd["boxes"]) > 0 and len(sam3["boxes"]) > 0)
                else f"  {stem} f={idx:5d}: GD={len(gd['boxes'])} | SAM3={len(sam3['boxes'])} -> {out_path.name}"
            )

    print(f"\nDone. Panels in: {out_dir}")


if __name__ == "__main__":
    main()

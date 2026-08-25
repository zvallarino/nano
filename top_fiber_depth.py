#!/usr/bin/env python3
"""
top_fiber_depth.py — highlight the fiber(s) closest to the viewer with a
pretrained monocular-depth model (Depth Anything V2). No training, CPU-friendly.

Idea: the model predicts relative depth for every pixel. The fibers nearest the
camera are the "top" ones. We keep the nearest depth band (inside the fiber
mask) and highlight it. Brightness is never used, so a dark top fiber is fine.

Runs on CPU. Small model ~1-3s/image on a laptop; Base ~10-20s; Large slower.
First run downloads weights from Hugging Face (~100MB Small / ~400MB Base).

Usage
-----
    python top_fiber_depth.py input.png
    python top_fiber_depth.py input.png --model base
    python top_fiber_depth.py input.png --nearest 15      # keep nearest 15% depth
    python top_fiber_depth.py input.png --invert          # if near/far look flipped
    python top_fiber_depth.py input.png --save-depth      # also dump the depth map

Dependencies:
    pip install torch transformers pillow numpy
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# ======================================================================
#  EASY-TO-EDIT SETTINGS
# ======================================================================

MODEL = "small"            # "small" | "base" | "large"
NEAREST_PERCENT = 20       # keep the nearest this-% of depth as "top" fibers
MIN_BLOB = 1500            # ignore highlighted blobs smaller than this (px)
INVERT_DEPTH = False       # flip if the model's near/far comes out reversed
HIGHLIGHT_COLOR = (60, 220, 90)
OVERLAY_ALPHA = 0.55

# background red mask (these images have red = non-fiber); tune if red drifts
RED_R_MIN, RED_G_MAX, RED_B_MAX = 140, 90, 90

DEFAULT_OUTPUT_DIR = r"C:\Users\zvcoding\code\clement\output\depth_output"

_HF_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
# ======================================================================


def fiber_mask(rgb):
    """Fiber = everything that isn't the red background. If there's no red
    (a raw grayscale SEM), treat the whole frame as fiber."""
    r = rgb[..., 0].astype(int); g = rgb[..., 1].astype(int); b = rgb[..., 2].astype(int)
    red = (r > RED_R_MIN) & (g < RED_G_MAX) & (b < RED_B_MAX)
    if red.mean() < 0.02:          # basically no red -> not a masked image
        return np.ones(red.shape, bool)
    return ~red


_PIPE = {}


def get_pipe(model_key):
    """Load the model once and reuse it across a whole folder."""
    if model_key not in _PIPE:
        from transformers import pipeline
        _PIPE[model_key] = pipeline("depth-estimation",
                                    model=_HF_IDS[model_key], device=-1)  # -1 = CPU
    return _PIPE[model_key]


def run_depth(pil_img, model_key):
    pipe = get_pipe(model_key)
    out = pipe(pil_img)
    depth = np.asarray(out["predicted_depth"], dtype=np.float32)
    # pipeline returns depth at model resolution; resize to image size
    if depth.shape != (pil_img.height, pil_img.width):
        depth = np.asarray(
            Image.fromarray(depth).resize((pil_img.width, pil_img.height), Image.BILINEAR),
            dtype=np.float32)
    return depth


def main():
    p = argparse.ArgumentParser(description="Highlight closest fibers via Depth Anything V2.")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--model", choices=list(_HF_IDS), default=MODEL)
    p.add_argument("--nearest", type=float, default=NEAREST_PERCENT,
                   help="keep the nearest this %% of depth (default 20)")
    p.add_argument("--min-blob", type=int, default=MIN_BLOB)
    p.add_argument("--invert", action="store_true", default=INVERT_DEPTH)
    p.add_argument("--save-depth", action="store_true",
                   help="(deprecated) the depth map is always saved now")
    p.add_argument("--overlay", action="store_true",
                   help="also save the green nearest-%% overlay. Off by default: "
                        "it uses a cruder selection than fiber_diameter.py and can "
                        "highlight different fibers than the ones measured.")
    p.add_argument("--threads", type=int, default=12,
                   help="CPU threads for torch (default 12; you have 12 physical cores)")
    p.add_argument("--ext", default=".tif,.tiff,.png,.jpg,.jpeg",
                   help="extensions to process when input is a folder")
    args = p.parse_args()

    try:
        import torch
        torch.set_num_threads(args.threads)
    except Exception:
        pass

    if not os.path.exists(args.input):
        sys.exit(f"error: not found: {args.input}")

    # ---- folder mode ---------------------------------------------------
    if os.path.isdir(args.input):
        exts = tuple(e.strip().lower() for e in args.ext.split(","))
        files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                 if f.lower().endswith(exts)
                 and not f.lower().endswith(("_depth.png", "_topdepth.png"))]
        if not files:
            sys.exit(f"no images ({args.ext}) in {args.input}")
        print(f"batch: {len(files)} images, model={args.model} "
              f"(loading model once)\n")
        ok = fail = 0
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {os.path.basename(f)}")
            try:
                process_one(f, None, args)
                ok += 1
            except Exception as e:
                print(f"   FAILED: {e}")
                fail += 1
        print(f"\ndone: {ok} succeeded, {fail} failed")
        return

    # ---- single file ---------------------------------------------------
    try:
        process_one(args.input, args.output, args)
    except FileNotFoundError:
        sys.exit(f"error: file not found: {args.input}")
    except Exception as e:
        sys.exit(f"error: {e}")


def process_one(in_path, out_path, args):
    pil = Image.open(in_path)
    pil.seek(0)
    pil = pil.convert("RGB")
    rgb = np.asarray(pil)

    depth = run_depth(pil, args.model)

    # Depth Anything: larger value = closer. Invert if it comes out flipped.
    if args.invert:
        depth = -depth

    if out_path is None:
        stem = os.path.splitext(os.path.basename(in_path))[0]
        outp = os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_topdepth.png")
    else:
        outp = out_path
    od = os.path.dirname(outp)
    if od:
        os.makedirs(od, exist_ok=True)

    # the depth map is the real product — always written
    dnorm = depth - depth.min()
    dnorm = (dnorm / (dnorm.max() + 1e-9) * 255).astype(np.uint8)
    dp = outp.replace(".png", "_depth.png")
    Image.fromarray(dnorm).save(dp)
    print(f"   depth   -> {dp}")

    # optional green overlay (cruder selection; off by default)
    if args.overlay:
        mask = fiber_mask(rgb)
        vals = depth[mask]
        if vals.size == 0:
            raise RuntimeError("no fiber pixels found (check the red mask thresholds)")
        cutoff = np.percentile(vals, 100 - args.nearest)
        top = mask & (depth >= cutoff)
        try:
            from scipy import ndimage
            lbl, n = ndimage.label(top)
            sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
            top = np.isin(lbl, [i + 1 for i, s in enumerate(sizes) if s >= args.min_blob])
        except ImportError:
            pass
        out = rgb.copy()
        out[top] = (OVERLAY_ALPHA * np.array(HIGHLIGHT_COLOR)
                    + (1 - OVERLAY_ALPHA) * out[top]).astype(np.uint8)
        Image.fromarray(out).save(outp)
        print(f"   overlay -> {outp}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
r"""
crop_banner.py — remove the SEM info banner (text + scale bar) from micrographs.

Works on a single file or a whole folder, and keeps the format (TIFF stays TIFF,
no quality loss). By default it AUTO-DETECTS the banner: SEM banners are a solid
dark strip along the bottom, so it scans up from the bottom and cuts at the first
row where real image content resumes. You can also force a fixed pixel height.

Usage
-----
    python crop_banner.py input.tif                 # one file, auto-detect
    python crop_banner.py pictures\TIFF             # whole folder
    python crop_banner.py pictures\TIFF -o pictures\TIFF_cropped
    python crop_banner.py input.tif --bottom 60     # force 60 px off the bottom
    python crop_banner.py pictures\TIFF --preview    # print what it WOULD cut, no writing

Dependencies: numpy, pillow
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# ======================================================================
#  EASY-TO-EDIT SETTINGS
# ======================================================================

# Force a fixed crop height in pixels. None = auto-detect the banner.
FORCE_BOTTOM = None

# Auto-detect tuning. A banner row is "dark and quiet"; content resumes when a
# row's mean brightness jumps above this fraction of the image's overall mean.
CONTENT_MEAN_RATIO = 0.55
# Never cut more than this fraction of the image height (safety clamp).
MAX_CROP_FRACTION = 0.20

# File extensions treated as images when a folder is given.
EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# Suffix added to output filenames.
SUFFIX = "_crop"

# Where cropped files go when -o isn't given. Set to None to instead write
# alongside the input (folder -> <input>_cropped).
DEFAULT_OUTPUT_DIR = r"C:\Users\zvcoding\code\clement\pictures\TIFF_cropped"
# ======================================================================


def detect_banner(gray):
    """Return how many rows to cut off the bottom by finding the strongest
    content->banner transition. Uses medians so bright TEXT rows inside the
    banner don't fool it."""
    h, w = gray.shape
    overall = gray.mean()
    threshold = overall * CONTENT_MEAN_RATIO
    max_cut = int(h * MAX_CROP_FRACTION)

    means = gray.mean(axis=1)
    best_r, best_score = h, 0.0
    # scan the bottom strip for the sharpest step: bright content above (median
    # > threshold) dropping to dark banner below (median < threshold)
    for r in range(h - max_cut, h - 3):
        above = np.median(means[max(0, r - 8):r])
        below = np.median(means[r:min(h, r + 8)])
        if above > threshold and below < threshold:
            score = above - below
            if score > best_score:
                best_score, best_r = score, r
    return h - best_r if best_score > 0 else 0


def write_visual_preview(path, gray_shape, bottom, out_png, max_width=1200):
    """Save a downscaled copy with a red line at the cut and the region below
    it dimmed, so you can eyeball whether the cut is right."""
    from PIL import ImageDraw
    im = Image.open(path)
    im.seek(0)
    im = im.convert("RGB")
    w, h = im.size

    # dim the part that will be discarded
    disc = Image.new("RGB", (w, bottom), (200, 0, 0)) if bottom > 0 else None
    if disc is not None:
        region = im.crop((0, h - bottom, w, h))
        im.paste(Image.blend(region, disc, 0.45), (0, h - bottom))

    d = ImageDraw.Draw(im)
    lw = max(2, h // 400)
    if bottom > 0:
        d.line([(0, h - bottom), (w, h - bottom)], fill=(255, 0, 0), width=lw)

    if w > max_width:
        scale = max_width / w
        im = im.resize((max_width, int(h * scale)), Image.LANCZOS)

    od = os.path.dirname(out_png)
    if od:
        os.makedirs(od, exist_ok=True)
    im.save(out_png)
    print(f"  visual preview -> {out_png}  (red line = cut, red tint = discarded)")


def process(path, out_path, force_bottom, preview, show=False):
    im = Image.open(path)
    im.seek(0)
    gray = np.asarray(im.convert("L"), dtype=np.float64)
    h, w = gray.shape

    bottom = force_bottom if force_bottom is not None else detect_banner(gray)
    bottom = max(0, min(bottom, int(h * MAX_CROP_FRACTION)))

    if preview:
        print(f"{os.path.basename(path)}: {w}x{h} -> would cut {bottom}px "
              f"(keep {h - bottom}px)")
        if show:
            base = os.path.splitext(os.path.basename(path))[0]
            target = out_path if out_path.lower().endswith(".png") else \
                os.path.join(os.path.dirname(out_path), f"{base}_cutpreview.png")
            write_visual_preview(path, gray.shape, bottom, target)
        return

    cropped = im.crop((0, 0, w, h - bottom))
    # preserve format; TIFF stays TIFF (lossless)
    save_kwargs = {}
    if im.format == "TIFF":
        save_kwargs["compression"] = "tiff_lzw"  # lossless
    cropped.save(out_path, **save_kwargs)
    print(f"{os.path.basename(path)}: cut {bottom}px -> {out_path}")


def main():
    p = argparse.ArgumentParser(description="Crop the SEM info banner off image(s).")
    p.add_argument("input", help="an image file OR a folder of images")
    p.add_argument("-o", "--output", help="output file or folder "
                   "(default: alongside input with a _crop suffix)")
    p.add_argument("--bottom", type=int, default=FORCE_BOTTOM,
                   help="force N px off the bottom instead of auto-detecting")
    p.add_argument("--preview", action="store_true",
                   help="print what would be cut without writing anything")
    p.add_argument("--show", action="store_true",
                   help="with --preview, also save a downscaled image with the "
                        "cut line drawn so you can see it")
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"error: not found: {args.input}")

    # gather files
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                 if f.lower().endswith(EXTS)]
        if not files:
            sys.exit(f"no images ({', '.join(EXTS)}) in {args.input}")
        out_dir = args.output or DEFAULT_OUTPUT_DIR or (args.input.rstrip("\\/") + "_cropped")
        if not args.preview:
            os.makedirs(out_dir, exist_ok=True)
        for f in files:
            base, ext = os.path.splitext(os.path.basename(f))
            process(f, os.path.join(out_dir, f"{base}{SUFFIX}{ext}"),
                    args.bottom, args.preview, args.show)
    else:
        if args.output:
            out = args.output
        else:
            base, ext = os.path.splitext(os.path.basename(args.input))
            fname = f"{base}{SUFFIX}{ext}"
            out = os.path.join(DEFAULT_OUTPUT_DIR, fname) if DEFAULT_OUTPUT_DIR else \
                os.path.join(os.path.dirname(args.input), fname)
        od = os.path.dirname(out)
        if od and not args.preview:
            os.makedirs(od, exist_ok=True)
        process(args.input, out, args.bottom, args.preview, args.show)


if __name__ == "__main__":
    main()
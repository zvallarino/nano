#!/usr/bin/env python3
"""
fiber_diameter.py — measure the diameter of the fiber(s) closest to the viewer.

Takes the DEPTH MAP produced by top_fiber_depth.py (the "..._depth.png" file)
and:
  1. splits it into individual fiber instances (watershed on depth edges)
  2. ranks those instances by depth  -> the brightest/nearest are the top fibers
  3. measures each one's diameter with perpendicular chords along its axis
  4. draws the measurement lines so you can SEE that it's right
  5. prints the numbers and writes a CSV

Why instances instead of a brightness threshold: depth changes gradually ALONG
a fiber as it recedes, so a global "nearest 20%" cut clips fibers into lens
shapes. Segmenting first, then ranking whole instances, keeps fibers intact.

Diameter is reported in PIXELS (no scale calibration).

Usage
-----
    python fiber_diameter.py depth.png
    python fiber_diameter.py depth.png --top 3
    python fiber_diameter.py depth.png --original original_crop.tif
    python fiber_diameter.py depth.png --background sem.png   # draw on the SEM

Dependencies: numpy, pillow, scipy, scikit-image
"""

import argparse
import csv
import os
import sys
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

warnings.filterwarnings("ignore")

# ======================================================================
#  EASY-TO-EDIT SETTINGS
# ======================================================================

# How many of the closest fibers to measure.
TOP_N = 3

# Segmentation runs at 1/DOWNSCALE resolution for speed; measurement is scaled
# back to full-resolution pixels. 2 is a good balance on 20 MP images.
DOWNSCALE = 2

# A pixel belongs to a "flat" fiber interior when the depth gradient is below
# this. Raise if fibers get split into pieces; lower if fibers merge together.
FLAT_GRADIENT_MAX = 12

# Instance filters (in downscaled pixels).
MIN_AREA = 2000          # ignore small fragments
MIN_ELONGATION = 4.0     # a fiber is long and thin; this rejects merged blobs

# Measurement sampling along the fiber axis.
N_SAMPLES = 25           # how many chords to measure per fiber
END_TRIM = 0.10          # skip this fraction at each tapered end

# Drawing
LINE_COLOR = (255, 60, 60)      # measurement ticks
AXIS_COLOR = (60, 200, 255)     # fiber centerline
LABEL_COLOR = (255, 255, 255)
TICK_EVERY = 4                  # draw every Nth measured chord (keeps it readable)
PREVIEW_MAX_WIDTH = 2000        # downscale the saved overlay to this width

DEFAULT_OUTPUT_DIR = r"C:\Users\zvcoding\code\clement\output\diameter_output"

# ======================================================================


def load_gray(path):
    im = Image.open(path)
    im.seek(0)
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(im).astype(np.float64)
        lo, hi = np.percentile(arr, [0.5, 99.5])
        if hi <= lo:
            lo, hi = arr.min(), arr.max()
        arr = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1) * 255.0
        return arr.astype(np.uint8)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def segment_instances(depth_small, flat_max):
    """Watershed on depth-gradient: fiber boundaries are sharp depth steps,
    fiber interiors are smooth."""
    sm = ndimage.gaussian_filter(depth_small.astype(float), 1.0)
    grad = np.hypot(ndimage.sobel(sm, 1), ndimage.sobel(sm, 0))
    markers = label(grad < flat_max)
    return watershed(grad, markers)


def rank_fibers(ws, depth_small, min_area, min_elong):
    """Keep elongated instances, sorted nearest-first (highest median depth)."""
    out = []
    for p in regionprops(ws, intensity_image=depth_small):
        if p.area < min_area:
            continue
        elong = p.axis_major_length / (p.axis_minor_length + 1e-6)
        if elong < min_elong:
            continue
        p._median_depth = float(np.median(p.image_intensity[p.image]))
        p._elong = elong
        out.append(p)
    out.sort(key=lambda p: -p._median_depth)
    return out


def measure_fiber(mask, n_samples=N_SAMPLES, end_trim=END_TRIM):
    """Perpendicular-chord widths along the fiber's principal axis.
    Returns (widths, chord_endpoints, axis_endpoints)."""
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return np.array([]), [], None
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]
    perp = np.array([-axis[1], axis[0]])

    t = (pts - c) @ axis
    tmin, tmax = t.min(), t.max()
    H, W = mask.shape
    max_r = int(max(H, W))

    widths, chords = [], []
    for frac in np.linspace(end_trim, 1.0 - end_trim, n_samples):
        tv = tmin + frac * (tmax - tmin)
        center = c + axis * tv
        ends = []
        n = 0
        for sgn in (1, -1):
            k = 0
            while k < max_r:
                q = center + perp * sgn * k
                xi, yi = int(round(q[0])), int(round(q[1]))
                if not (0 <= xi < W and 0 <= yi < H) or not mask[yi, xi]:
                    break
                k += 1
            n += k
            ends.append(center + perp * sgn * max(k - 1, 0))
        if n > 2 and len(ends) == 2:
            widths.append(n)
            chords.append((ends[0], ends[1]))

    axis_pts = (c + axis * tmin, c + axis * tmax)
    return np.array(widths, dtype=float), chords, axis_pts


def main():
    ap = argparse.ArgumentParser(
        description="Measure diameter of the closest fiber(s) from a depth map.")
    ap.add_argument("depth", help="the ..._depth.png from top_fiber_depth.py")
    ap.add_argument("-o", "--output", help="overlay image path")
    ap.add_argument("--background", help="image to draw on (default: the depth map). "
                                         "Pass the SEM image for a nicer overlay.")
    ap.add_argument("--original", help="original SEM image; used to sharpen the "
                                       "fiber mask (depth edges are ~7px soft)")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--downscale", type=int, default=DOWNSCALE)
    ap.add_argument("--flat", type=float, default=FLAT_GRADIENT_MAX)
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--min-elong", type=float, default=MIN_ELONGATION)
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    ap.add_argument("--csv", action="store_true", help="also write a CSV of every chord")
    args = ap.parse_args()

    if not os.path.exists(args.depth):
        sys.exit(f"error: not found: {args.depth}")

    if not args.depth.lower().endswith("_depth.png"):
        print("WARNING: expected the '..._topdepth_depth.png' (grayscale depth map).\n"
              "         '..._topdepth.png' is the green overlay and will NOT measure\n"
              "         correctly. Continuing anyway.\n")

    depth_full = load_gray(args.depth)
    d = max(1, args.downscale)
    depth_small = depth_full[::d, ::d]

    ws = segment_instances(depth_small, args.flat)
    fibers = rank_fibers(ws, depth_small, args.min_area, args.min_elong)
    if not fibers:
        sys.exit("no fiber-like instances found — try lowering --min-elong or --min-area")

    keep = fibers[:args.top]

    # optional: sharpen masks using the original image's crisp edges
    orig_small = None
    if args.original:
        og = load_gray(args.original)
        if og.shape != depth_full.shape:
            og = np.asarray(Image.fromarray(og).resize(
                (depth_full.shape[1], depth_full.shape[0]), Image.BILINEAR))
        orig_small = og[::d, ::d]
        thr = np.percentile(orig_small, 45)   # fiber vs dark background

    # ---- measure -------------------------------------------------------
    results = []
    for i, p in enumerate(keep):
        mask = (ws == p.label)
        if orig_small is not None:
            sharpened = mask & (orig_small > thr)
            if sharpened.sum() > 0.3 * mask.sum():
                mask = ndimage.binary_closing(sharpened, np.ones((3, 3)))
        widths, chords, axis_pts = measure_fiber(mask, args.samples)
        if widths.size == 0:
            continue
        results.append({
            "rank": i + 1,
            "median_px": float(np.median(widths)) * d,
            "mean_px": float(widths.mean()) * d,
            "sd_px": float(widths.std()) * d,
            "min_px": float(widths.min()) * d,
            "max_px": float(widths.max()) * d,
            "n": int(widths.size),
            "length_px": float(np.hypot(*(axis_pts[1] - axis_pts[0]))) * d,
            "depth": p._median_depth,
            "chords": chords,
            "axis": axis_pts,
            "widths_px": (widths * d).tolist(),
        })

    if not results:
        sys.exit("found fibers but could not measure them — try --samples 15")

    # ---- report --------------------------------------------------------
    print(f"\nmeasured {len(results)} fiber(s), closest first   [pixels, full-res]\n")
    print(f"{'rank':>4} {'diameter':>9} {'sd':>7} {'range':>15} {'length':>8} {'depth':>6}")
    print("-" * 56)
    for r in results:
        print(f"{r['rank']:>4} {r['median_px']:>9.1f} {r['sd_px']:>7.1f} "
              f"{r['min_px']:>6.0f}-{r['max_px']:<8.0f} {r['length_px']:>8.0f} "
              f"{r['depth']:>6.0f}")
    print()

    # ---- draw ----------------------------------------------------------
    bg_path = args.background or args.original or args.depth
    bg = load_gray(bg_path)
    if bg.shape != depth_full.shape:
        bg = np.asarray(Image.fromarray(bg).resize(
            (depth_full.shape[1], depth_full.shape[0]), Image.BILINEAR))
    canvas = Image.fromarray(np.stack([bg] * 3, -1).astype(np.uint8))
    dr = ImageDraw.Draw(canvas)

    fsize = max(20, canvas.width // 60)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()

    lw = max(2, canvas.width // 900)
    for r in results:
        a0, a1 = r["axis"]
        dr.line([tuple(a0 * d), tuple(a1 * d)], fill=AXIS_COLOR, width=lw)
        for j, (e0, e1) in enumerate(r["chords"]):
            if j % TICK_EVERY:
                continue
            dr.line([tuple(e0 * d), tuple(e1 * d)], fill=LINE_COLOR, width=lw * 2)
        mid = ((a0 + a1) / 2) * d
        txt = f"#{r['rank']}  {r['median_px']:.0f} px"
        tb = dr.textbbox((0, 0), txt, font=font)
        pad = fsize // 4
        dr.rectangle([mid[0] - pad, mid[1] - pad,
                      mid[0] + (tb[2] - tb[0]) + pad, mid[1] + (tb[3] - tb[1]) + 2 * pad],
                     fill=(0, 0, 0))
        dr.text((mid[0], mid[1]), txt, fill=LABEL_COLOR, font=font)

    if canvas.width > PREVIEW_MAX_WIDTH:
        sc = PREVIEW_MAX_WIDTH / canvas.width
        canvas = canvas.resize((PREVIEW_MAX_WIDTH, int(canvas.height * sc)), Image.LANCZOS)

    stem = os.path.splitext(os.path.basename(args.depth))[0].replace("_depth", "")
    out = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_diameter.png")
    od = os.path.dirname(out)
    if od:
        os.makedirs(od, exist_ok=True)
    canvas.save(out)
    print(f"overlay -> {out}")

    if args.csv:
        cpath = os.path.splitext(out)[0] + ".csv"
        with open(cpath, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "chord_index", "width_px", "median_px", "sd_px", "length_px"])
            for r in results:
                for k, wd in enumerate(r["widths_px"]):
                    w.writerow([r["rank"], k, f"{wd:.1f}",
                                f"{r['median_px']:.1f}", f"{r['sd_px']:.1f}",
                                f"{r['length_px']:.0f}"])
        print(f"csv     -> {cpath}")


if __name__ == "__main__":
    main()
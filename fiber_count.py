#!/usr/bin/env python3
"""
fiber_count.py — count the fibers visible in a depth map.

Input is the "..._topdepth_depth.png" produced by top_fiber_depth.py.

How it works
------------
1. WATERSHED on the depth gradient splits the image into flat-depth regions.
   A fiber that passes BEHIND another gets cut into several fragments here,
   so the raw region count badly over-counts (45 regions for ~20 fibers).

2. FILTER out things that aren't fiber surface:
      - the far background seen through the gaps (depth below --bg-cutoff)
      - blobs that aren't elongated
      - fragments far off the typical fiber width

3. MERGE fragments back into whole fibers. These fibers are straight ribbons,
   so every fragment of one fiber lies on the SAME LINE. Each fragment is
   described by its axis angle and perpendicular offset -- Hough-style
   (theta, rho) -- and fragments that share a line, have similar width, and
   run at the same angle are united into one fiber.

4. COUNT the merged groups, and draw a numbered overlay so the count can be
   checked by eye.

The count is sensitive to --bg-cutoff: raising it counts only fibers near the
surface, lowering it includes fibers buried deeper in the mat. The script
prints a sensitivity table so you can see this rather than trusting one number.

Usage
-----
    python fiber_count.py depth.png
    python fiber_count.py depth.png --bg-cutoff 50
    python fiber_count.py depth.png --sweep          # show count vs cutoff
    python fiber_count.py depth_folder --csv         # whole folder

Dependencies: numpy pillow scipy scikit-image
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

# Depth below this is treated as far background (seen through the gaps),
# not fiber. LOWER = counts more deeply-buried fibers. This is the main knob.
BG_CUTOFF = 45

# Segmentation runs at 1/DOWNSCALE resolution for speed.
DOWNSCALE = 2

# Watershed: a pixel is "flat fiber interior" when the depth gradient is
# below this. Raise if fibers fragment too much; lower if they merge.
FLAT_GRADIENT_MAX = 12

# Fragment filters (downscaled pixels).
MIN_AREA = 800
MIN_ELONGATION = 2.0
# keep fragments whose width is within this factor of the median fiber width
WIDTH_LO, WIDTH_HI = 0.35, 2.2

# Merging: two fragments are the same fiber if their axes agree in angle,
# they sit on the same line, and their widths are comparable.
MERGE_ANGLE_DEG = 7.0
MERGE_OFFSET_FRAC = 0.6      # x max(width)
MERGE_WIDTH_RATIO = 2.2

DEFAULT_OUTPUT_DIR = r"C:\Users\zvcoding\code\clement\output\count_output"

PALETTE = [
    (255, 80, 80), (80, 170, 255), (90, 230, 130), (255, 200, 60),
    (200, 110, 255), (255, 140, 40), (70, 230, 230), (255, 110, 180),
    (150, 255, 90), (120, 140, 255), (255, 235, 120), (0, 200, 160),
]

# ======================================================================


def load_gray(path):
    im = Image.open(path)
    im.seek(0)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def fragments(depth_small, bg_cutoff, flat_max, min_area, min_elong):
    """Watershed the depth map and keep fragments that look like fiber surface."""
    sm = ndimage.gaussian_filter(depth_small.astype(float), 1.0)
    grad = np.hypot(ndimage.sobel(sm, 1), ndimage.sobel(sm, 0))
    ws = watershed(grad, label(grad < flat_max))

    cand = []
    for p in regionprops(ws, intensity_image=depth_small):
        if p.area < min_area:
            continue
        med = float(np.median(p.image_intensity[p.image]))
        if med < bg_cutoff:                      # far background, not fiber
            continue
        elong = p.axis_major_length / (p.axis_minor_length + 1e-6)
        if elong < min_elong:
            continue
        cand.append((p, med, elong))

    if not cand:
        return ws, []

    med_w = float(np.median([p.axis_minor_length for p, _, _ in cand]))
    out = []
    for p, med, elong in cand:
        w = p.axis_minor_length
        if not (WIDTH_LO * med_w <= w <= WIDTH_HI * med_w):
            continue
        ys, xs = np.where(ws == p.label)
        pts = np.stack([xs, ys], 1).astype(float)
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        ax = vt[0]
        theta = float(np.arctan2(ax[1], ax[0]) % np.pi)
        normal = np.array([-np.sin(theta), np.cos(theta)])
        out.append({
            "label": p.label, "theta": theta, "rho": float(normal @ c),
            "w": float(w), "len": float(p.axis_major_length),
            "area": int(p.area), "depth": med, "centroid": c,
        })
    return ws, out


def merge_collinear(frags, angle_deg=MERGE_ANGLE_DEG,
                    offset_frac=MERGE_OFFSET_FRAC, width_ratio=MERGE_WIDTH_RATIO):
    """Union fragments that lie on the same line -> whole fibers."""
    n = len(frags)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    dmax = np.deg2rad(angle_deg)
    for i in range(n):
        for j in range(i + 1, n):
            f, g = frags[i], frags[j]
            dt = abs(f["theta"] - g["theta"])
            dt = min(dt, np.pi - dt)
            if dt > dmax:
                continue
            if abs(f["rho"] - g["rho"]) > offset_frac * max(f["w"], g["w"]):
                continue
            lo, hi = min(f["w"], g["w"]), max(f["w"], g["w"])
            if hi / max(lo, 1e-6) > width_ratio:
                continue
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b

    groups = {}
    for i, f in enumerate(frags):
        groups.setdefault(find(i), []).append(f)
    return list(groups.values())


def count_fibers(depth_small, bg_cutoff, flat_max, min_area, min_elong):
    ws, frags = fragments(depth_small, bg_cutoff, flat_max, min_area, min_elong)
    if not frags:
        return ws, [], []
    return ws, frags, merge_collinear(frags)


def draw(depth_full, ws, groups, downscale, out_png):
    canvas = Image.fromarray(np.stack([depth_full] * 3, -1).astype(np.uint8))
    tint = canvas.copy()
    dr = ImageDraw.Draw(tint)

    # colour each fiber's fragments
    lut = {}
    for gi, g in enumerate(groups):
        for f in g:
            lut[f["label"]] = PALETTE[gi % len(PALETTE)]
    if lut:
        small_rgb = np.zeros(ws.shape + (3,), np.uint8)
        hit = np.zeros(ws.shape, bool)
        for lab, col in lut.items():
            m = ws == lab
            small_rgb[m] = col
            hit |= m
        big = np.asarray(Image.fromarray(small_rgb).resize(
            (depth_full.shape[1], depth_full.shape[0]), Image.NEAREST))
        bigmask = np.asarray(Image.fromarray(hit.astype(np.uint8) * 255).resize(
            (depth_full.shape[1], depth_full.shape[0]), Image.NEAREST)) > 127
        arr = np.asarray(tint).copy()
        arr[bigmask] = (0.45 * big[bigmask] + 0.55 * arr[bigmask]).astype(np.uint8)
        tint = Image.fromarray(arr)
        dr = ImageDraw.Draw(tint)

    fsize = max(24, tint.width // 55)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()

    for gi, g in enumerate(groups):
        big = max(g, key=lambda f: f["area"])
        cx, cy = big["centroid"] * downscale
        txt = str(gi + 1)
        tb = dr.textbbox((0, 0), txt, font=font)
        pad = fsize // 3
        dr.rectangle([cx - pad, cy - pad,
                      cx + (tb[2] - tb[0]) + pad, cy + (tb[3] - tb[1]) + 2 * pad],
                     fill=(0, 0, 0))
        dr.text((cx, cy), txt, fill=PALETTE[gi % len(PALETTE)], font=font)

    if tint.width > 2000:
        sc = 2000 / tint.width
        tint = tint.resize((2000, int(tint.height * sc)), Image.LANCZOS)
    od = os.path.dirname(out_png)
    if od:
        os.makedirs(od, exist_ok=True)
    tint.save(out_png)


def process(path, args):
    depth_full = load_gray(path)
    d = max(1, args.downscale)
    small = depth_full[::d, ::d]

    ws, frags, groups = count_fibers(small, args.bg_cutoff, args.flat,
                                     args.min_area, args.min_elong)
    name = os.path.basename(path)
    print(f"\n{name}")
    print(f"  fragments kept : {len(frags)}")
    print(f"  FIBERS COUNTED : {len(groups)}   (bg-cutoff {args.bg_cutoff})")

    if groups:
        widths = [np.median([f['w'] for f in g]) * d for g in groups]
        multi = sum(1 for g in groups if len(g) > 1)
        print(f"  median width   : {np.median(widths):.0f} px")
        print(f"  {multi} fiber(s) were reassembled from multiple fragments")

    if args.sweep:
        print("\n  sensitivity to --bg-cutoff:")
        for cut in (30, 40, 45, 50, 60, 70, 80, 100):
            _, _, gg = count_fibers(small, cut, args.flat,
                                    args.min_area, args.min_elong)
            bar = "#" * len(gg)
            print(f"    cutoff {cut:>3} -> {len(gg):>2} fibers  {bar}")

    stem = os.path.splitext(name)[0].replace("_topdepth_depth", "")
    out_png = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"{stem}_count.png")
    draw(depth_full, ws, groups, d, out_png)
    print(f"  overlay -> {out_png}")
    return {"image": name, "fiber_count": len(groups),
            "fragments": len(frags), "bg_cutoff": args.bg_cutoff,
            "median_width_px": (f"{np.median([np.median([f['w'] for f in g]) * d for g in groups]):.0f}"
                                if groups else "")}


def main():
    ap = argparse.ArgumentParser(description="Count fibers in a depth map.")
    ap.add_argument("input", help="a ..._depth.png file, or a folder of them")
    ap.add_argument("-o", "--output", help="overlay path (single-file mode)")
    ap.add_argument("--bg-cutoff", type=float, default=BG_CUTOFF,
                    help=f"depth below this is background (default {BG_CUTOFF}); "
                         "lower counts more buried fibers")
    ap.add_argument("--downscale", type=int, default=DOWNSCALE)
    ap.add_argument("--flat", type=float, default=FLAT_GRADIENT_MAX)
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--min-elong", type=float, default=MIN_ELONGATION)
    ap.add_argument("--sweep", action="store_true",
                    help="print how the count varies with the cutoff")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"error: not found: {args.input}")

    rows = []
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
                 if f.lower().endswith("_depth.png")]
        if not files:
            sys.exit(f"no '*_depth.png' files in {args.input}")
        args.output = None
        for f in files:
            rows.append(process(f, args))
    else:
        rows.append(process(args.input, args))

    if args.csv and rows:
        out_csv = os.path.join(DEFAULT_OUTPUT_DIR, "fiber_counts.csv")
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\ncsv -> {out_csv}")
    print()


if __name__ == "__main__":
    main()
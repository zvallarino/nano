#!/usr/bin/env python3
"""
run_pipeline.py — full nanofiber pipeline, original TIFF -> real diameters.

For every TIFF in a folder it runs, in order:

    1. SCALE     read JEOL metadata -> nm/px            (sem_scale.py)
    2. CROP      strip the SEM info banner              (crop_banner.py)
    3. DEPTH     Depth Anything V2 -> depth map         (top_fiber_depth.py)
    4. DIAMETER  segment + measure closest fiber(s)     (fiber_diameter.py)
    5. REPORT    per-image CSV in nanometres + combined summary

Cropping removes bottom rows only, so image WIDTH is unchanged and the
nm/px from the original file stays valid for the cropped one.

All four helper scripts must sit in the same folder as this one.

Usage
-----
    python run_pipeline.py pictures\TIFF
    python run_pipeline.py pictures\TIFF --model large --top 3
    python run_pipeline.py pictures\TIFF --skip-crop --skip-depth   # re-measure only
    python run_pipeline.py pictures\TIFF --scale-from output\scale_output\summary.txt

Dependencies: numpy pillow scipy scikit-image torch transformers tifffile
"""

import argparse
import csv
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# ======================================================================
#  EASY-TO-EDIT SETTINGS
# ======================================================================

BASE = r"C:\Users\zvcoding\code\clement"

SCALE_DIR = os.path.join(BASE, "output", "scale_output")
CROP_DIR = os.path.join(BASE, "pictures", "TIFF_cropped")
DEPTH_DIR = os.path.join(BASE, "output", "depth_output")
DIAM_DIR = os.path.join(BASE, "output", "diameter_output")

MODEL = "large"        # small | base | large
TOP_N = 3              # how many of the closest fibers to measure per image
DOWNSCALE = 2          # segmentation downscale (measurement is scaled back up)
THREADS = 12

EXTS = (".tif", ".tiff")

# ======================================================================

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import crop_banner            # noqa: E402
import fiber_diameter as fd   # noqa: E402
import top_fiber_depth as tfd  # noqa: E402

try:
    import sem_scale
    HAVE_SEM_SCALE = True
except ImportError:
    HAVE_SEM_SCALE = False


# ------------------------------------------------------------ scale
def parse_summary(path):
    """Parse sem_scale's summary.txt -> {filename: nm_per_px}."""
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 3 or parts[0].lower() == "file":
                continue
            m = re.search(r"([\d.]+)\s*nm/px", parts[2])
            if m:
                out[parts[0]] = float(m.group(1))
    return out


def build_scale_table(files, scale_from):
    """nm/px per original filename, from summary.txt or from the TIFFs."""
    if scale_from:
        if not os.path.exists(scale_from):
            sys.exit(f"error: scale summary not found: {scale_from}")
        table = parse_summary(scale_from)
        print(f"scale: read {len(table)} entries from {scale_from}")
        return table

    if not HAVE_SEM_SCALE:
        print("scale: sem_scale.py not found -> diameters will be PIXELS only")
        return {}

    os.makedirs(SCALE_DIR, exist_ok=True)
    table, rows = {}, []
    for f in files:
        try:
            r = sem_scale.analyze_tiff(f)
        except Exception as e:
            print(f"   scale FAILED {os.path.basename(f)}: {e}")
            continue
        if r.get("pixel_size_nm"):
            table[os.path.basename(f)] = r["pixel_size_nm"]
            rows.append(r)
    if rows:
        with open(os.path.join(SCALE_DIR, "summary.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(sem_scale.format_summary(rows) + "\n")
        print(f"scale: computed {len(table)} entries -> "
              f"{os.path.join(SCALE_DIR, 'summary.txt')}")
    return table


# ------------------------------------------------------------ crop
def do_crop(src, dst):
    im = Image.open(src)
    im.seek(0)
    gray = np.asarray(im.convert("L"), dtype=np.float64)
    h, w = gray.shape
    bottom = crop_banner.detect_banner(gray)
    cropped = im.crop((0, 0, w, h - bottom))
    kw = {"compression": "tiff_lzw"} if (im.format or "").upper() == "TIFF" else {}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cropped.save(dst, **kw)
    return bottom, cropped.size


# ------------------------------------------------------------ depth
def do_depth(src, dst_png, model):
    pil = Image.open(src)
    pil.seek(0)
    pil = pil.convert("RGB")
    depth = tfd.run_depth(pil, model)
    d = depth - depth.min()
    d = (d / (d.max() + 1e-9) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(dst_png), exist_ok=True)
    Image.fromarray(d).save(dst_png)
    return d


# ------------------------------------------------------------ measure
def do_measure(depth_full, top_n, downscale):
    """Return list of per-fiber dicts with pixel measurements."""
    small = depth_full[::downscale, ::downscale]
    ws = fd.segment_instances(small, fd.FLAT_GRADIENT_MAX)
    fibers = fd.rank_fibers(ws, small, fd.MIN_AREA, fd.MIN_ELONGATION)
    results = []
    for i, p in enumerate(fibers[:top_n]):
        mask = (ws == p.label)
        widths, chords, axis_pts = fd.measure_fiber(mask)
        if widths.size == 0:
            continue
        results.append({
            "rank": i + 1,
            "median_px": float(np.median(widths)) * downscale,
            "mean_px": float(widths.mean()) * downscale,
            "sd_px": float(widths.std()) * downscale,
            "min_px": float(widths.min()) * downscale,
            "max_px": float(widths.max()) * downscale,
            "n_chords": int(widths.size),
            "length_px": float(np.hypot(*(axis_pts[1] - axis_pts[0]))) * downscale,
            "depth": p._median_depth,
            "chords": chords,
            "axis": axis_pts,
            "widths_px": (widths * downscale).tolist(),
        })
    return results


def draw_overlay(depth_full, results, downscale, out_png, nm_per_px):
    from PIL import ImageDraw, ImageFont
    canvas = Image.fromarray(np.stack([depth_full] * 3, -1).astype(np.uint8))
    dr = ImageDraw.Draw(canvas)
    fsize = max(20, canvas.width // 60)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()
    lw = max(2, canvas.width // 900)
    d = downscale
    for r in results:
        a0, a1 = r["axis"]
        dr.line([tuple(a0 * d), tuple(a1 * d)], fill=fd.AXIS_COLOR, width=lw)
        for j, (e0, e1) in enumerate(r["chords"]):
            if j % fd.TICK_EVERY:
                continue
            dr.line([tuple(e0 * d), tuple(e1 * d)], fill=fd.LINE_COLOR, width=lw * 2)
        mid = ((a0 + a1) / 2) * d
        txt = (f"#{r['rank']}  {r['median_px'] * nm_per_px:.0f} nm"
               if nm_per_px else f"#{r['rank']}  {r['median_px']:.0f} px")
        tb = dr.textbbox((0, 0), txt, font=font)
        pad = fsize // 4
        dr.rectangle([mid[0] - pad, mid[1] - pad,
                      mid[0] + (tb[2] - tb[0]) + pad,
                      mid[1] + (tb[3] - tb[1]) + 2 * pad], fill=(0, 0, 0))
        dr.text((mid[0], mid[1]), txt, fill=fd.LABEL_COLOR, font=font)
    if canvas.width > fd.PREVIEW_MAX_WIDTH:
        sc = fd.PREVIEW_MAX_WIDTH / canvas.width
        canvas = canvas.resize((fd.PREVIEW_MAX_WIDTH, int(canvas.height * sc)),
                               Image.LANCZOS)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)


# ------------------------------------------------------------ main
FIELDS = ["image", "mag_nm_per_px", "fiber_rank",
          "diameter_nm", "diameter_px", "sd_nm", "sd_px",
          "min_nm", "max_nm", "length_nm", "length_px",
          "n_chords", "depth_rank_value"]


def row_for(name, nm, r):
    f = (lambda v: v * nm) if nm else (lambda v: "")
    return {
        "image": name,
        "mag_nm_per_px": f"{nm:.4f}" if nm else "",
        "fiber_rank": r["rank"],
        "diameter_nm": f"{f(r['median_px']):.1f}" if nm else "",
        "diameter_px": f"{r['median_px']:.1f}",
        "sd_nm": f"{f(r['sd_px']):.1f}" if nm else "",
        "sd_px": f"{r['sd_px']:.1f}",
        "min_nm": f"{f(r['min_px']):.1f}" if nm else "",
        "max_nm": f"{f(r['max_px']):.1f}" if nm else "",
        "length_nm": f"{f(r['length_px']):.0f}" if nm else "",
        "length_px": f"{r['length_px']:.0f}",
        "n_chords": r["n_chords"],
        "depth_rank_value": f"{r['depth']:.0f}",
    }


def main():
    ap = argparse.ArgumentParser(description="Full nanofiber pipeline.")
    ap.add_argument("input", help="folder of original SEM TIFFs")
    ap.add_argument("--model", default=MODEL, choices=["small", "base", "large"])
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--downscale", type=int, default=DOWNSCALE)
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--scale-from", default=None,
                    help="read nm/px from an existing sem_scale summary.txt")
    ap.add_argument("--skip-crop", action="store_true")
    ap.add_argument("--skip-depth", action="store_true",
                    help="reuse depth maps already in depth_output")
    args = ap.parse_args()

    try:
        import torch
        torch.set_num_threads(args.threads)
    except Exception:
        pass

    if not os.path.isdir(args.input):
        sys.exit(f"error: not a folder: {args.input}")
    files = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))
             if f.lower().endswith(EXTS)]
    if not files:
        sys.exit(f"no TIFFs in {args.input}")

    print(f"\n=== pipeline: {len(files)} images, model={args.model} ===\n")

    # STEP 1 -----------------------------------------------------------
    scale_table = build_scale_table(files, args.scale_from)
    print()

    all_rows = []
    ok = fail = 0
    for i, src in enumerate(files, 1):
        name = os.path.basename(src)
        stem = os.path.splitext(name)[0]
        print(f"[{i}/{len(files)}] {name}")
        try:
            nm = scale_table.get(name)
            if nm:
                print(f"   scale    {nm:.4f} nm/px")
            else:
                print("   scale    UNKNOWN -> pixels only")

            # STEP 2 crop
            crop_path = os.path.join(CROP_DIR, f"{stem}_crop.tif")
            if args.skip_crop and os.path.exists(crop_path):
                print("   crop     (reused)")
            else:
                cut, size = do_crop(src, crop_path)
                print(f"   crop     -{cut}px -> {size[0]}x{size[1]}")

            # STEP 3 depth
            depth_png = os.path.join(DEPTH_DIR, f"{stem}_crop_topdepth_depth.png")
            if args.skip_depth and os.path.exists(depth_png):
                depth_full = np.asarray(Image.open(depth_png).convert("L"))
                print("   depth    (reused)")
            else:
                depth_full = do_depth(crop_path, depth_png, args.model)
                print(f"   depth    -> {os.path.basename(depth_png)}")

            # STEP 4 measure
            results = do_measure(depth_full, args.top, args.downscale)
            if not results:
                print("   measure  no fibers found")
                fail += 1
                continue

            # STEP 5 report
            rows = [row_for(name, nm, r) for r in results]
            all_rows.extend(rows)
            csv_path = os.path.join(DIAM_DIR, f"{stem}_diameters.csv")
            os.makedirs(DIAM_DIR, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

            overlay = os.path.join(DIAM_DIR, f"{stem}_diameter.png")
            draw_overlay(depth_full, results, args.downscale, overlay, nm)

            for r in results:
                if nm:
                    print(f"   fiber #{r['rank']}  {r['median_px']*nm:8.1f} nm "
                          f"(+/-{r['sd_px']*nm:.1f})   [{r['median_px']:.0f} px]")
                else:
                    print(f"   fiber #{r['rank']}  {r['median_px']:.0f} px "
                          f"(+/-{r['sd_px']:.0f})")
            print(f"   csv      -> {os.path.basename(csv_path)}")
            ok += 1

        except Exception as e:
            print(f"   FAILED: {e}")
            traceback.print_exc(limit=1)
            fail += 1
        print()

    # combined summary --------------------------------------------------
    if all_rows:
        combined = os.path.join(DIAM_DIR, "ALL_diameters.csv")
        with open(combined, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(all_rows)
        print(f"combined CSV -> {combined}")

    print(f"\ndone: {ok} succeeded, {fail} failed")


if __name__ == "__main__":
    main()

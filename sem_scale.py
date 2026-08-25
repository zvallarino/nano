"""
sem_scale.py — pixel size (nm/px) for JEOL SEM TIFFs (JSM-IT500HR and
other models using the 127 mm polaroid reference).

Reads JEOL acquisition metadata (Mag, AccV, WD, PC, Vac, Detector, ...)
from the IPTC-NAA tag that survives Photoshop re-saves, and converts the
recorded magnification to a pixel size using JEOL's standard reference:

    pixel_size_nm = (127_000_000 nm / Mag) / image_width_px

Usage:
    python sem_scale.py image.tif                     # single file, prints to stdout
    python sem_scale.py folder                        # every *.tif/*.tiff inside, prints to stdout
    python sem_scale.py folder -o out_folder          # writes one .txt per image + summary
    python sem_scale.py image1.tif image2.tif ...     # mix files and folders freely
    python sem_scale.py image.tif --inspect           # dump raw JEOL keywords
    python sem_scale.py image.tif --ref-mm 127        # override reference

Requires:
    pip install tifffile
"""

import argparse
import sys
from pathlib import Path

try:
    import tifffile
except ImportError:
    sys.stderr.write("Missing dependency. Install with:\n    pip install tifffile\n")
    sys.exit(1)


# JEOL's classic 127 mm (5 inch) polaroid reference. Confirmed for the
# JSM-IT500HR by matching Mag against burned-in scale bars (10000x -> 1 um
# in 12.7 um FOV, 5000x -> 5 um in 25.4 um FOV, 500x -> 50 um in 254 um FOV).
JEOL_REF_MM_DEFAULT = 127.0

# Expected scale-bar length in nm for common mags on this instrument
# (used only for the human-readable sanity-check line).
EXPECTED_BAR_NM = {
    50:      500_000,
    100:     500_000,
    200:     100_000,
    500:      50_000,
    1000:     10_000,
    2000:     10_000,
    5000:      5_000,
    10000:     1_000,
    20000:       500,
    50000:       500,
}


# ------------------------------------------------------- IPTC parsing
def parse_iptc_keywords(iptc_bytes):
    """Return every record=2, tag=25 ('Keywords') payload as a string."""
    out = []
    i, n = 0, len(iptc_bytes)
    while i < n - 4:
        if (iptc_bytes[i] == 0x1C
                and iptc_bytes[i + 1] == 0x02
                and iptc_bytes[i + 2] == 0x19):
            length = (iptc_bytes[i + 3] << 8) | iptc_bytes[i + 4]
            data = iptc_bytes[i + 5: i + 5 + length]
            out.append(data.decode('latin-1', errors='replace'))
            i += 5 + length
        else:
            i += 1
    return out


def parse_jeol_metadata(iptc_bytes):
    """Parse JEOL 'Key/Value' keywords into a dict."""
    meta = {}
    for kw in parse_iptc_keywords(iptc_bytes):
        if '/' in kw:
            k, v = kw.split('/', 1)
            meta[k] = v
    return meta


# ----------------------------------------------------------- conversion
def pixel_size_nm_from_mag(mag, image_width_px, ref_mm=JEOL_REF_MM_DEFAULT):
    """Physical pixel size (nm/px) from JEOL Mag and image width."""
    field_width_nm = ref_mm * 1_000_000.0 / mag
    return field_width_nm / image_width_px


def _tag_str(page, name):
    t = page.tags.get(name)
    if not t:
        return None
    v = t.value
    return v.decode('latin-1', errors='replace') if isinstance(v, bytes) else str(v)


def analyze_tiff(path, ref_mm=JEOL_REF_MM_DEFAULT):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        width = page.imagewidth
        height = page.imagelength
        make = _tag_str(page, 'Make')
        model = _tag_str(page, 'Model')
        iptc_tag = page.tags.get('IPTCNAA') or page.tags.get(33723)
        iptc_bytes = bytes(iptc_tag.value) if iptc_tag else b''
    jeol = parse_jeol_metadata(iptc_bytes) if iptc_bytes else {}
    result = {
        'path': str(path),
        'width_px': width, 'height_px': height,
        'make': make, 'model': model,
        'jeol': jeol,
        'pixel_size_nm': None,
        'field_width_um': None,
        'mag': None,
        'ref_mm': ref_mm,
    }
    if 'Mag' in jeol:
        try:
            mag = float(jeol['Mag'])
            px_nm = pixel_size_nm_from_mag(mag, width, ref_mm)
            result['pixel_size_nm'] = px_nm
            result['field_width_um'] = px_nm * width / 1000.0
            result['mag'] = mag
        except ValueError:
            pass
    return result


# --------------------------------------------------------- formatting
def format_result(r):
    """Return the multi-line text block for one image."""
    lines = []
    name = Path(r['path']).name
    if r['pixel_size_nm'] is None:
        lines.append(f'{name}: could not parse magnification')
        if r['jeol']:
            lines.append(f'  IPTC keys found: {sorted(r["jeol"].keys())}')
        return '\n'.join(lines)

    px_nm = r['pixel_size_nm']
    fov_um = r['field_width_um']
    mag = int(r['mag'])
    j = r['jeol']

    # sanity-check line
    bar_nm = EXPECTED_BAR_NM.get(mag)
    if bar_nm:
        bar_str = (f'{bar_nm/1000:g} um' if bar_nm >= 1000
                   else f'{bar_nm:g} nm')
        sanity = f'{bar_str} scale bar sanity check'
    else:
        sanity = 'expected scale bar unknown for this mag'

    # main headline (matches the format you asked for)
    lines.append(
        f'{name}  {mag}x -> {px_nm:.2f} nm/px, '
        f'{fov_um:.1f} um FOV, {sanity}'
    )
    # extra detail below
    lines.append(f'  microscope:    {r["make"]} {r["model"]}')
    lines.append(f'  image size:    {r["width_px"]} x {r["height_px"]} px')
    lines.append(f'  pixel size:    {px_nm:.4f} nm/px ({px_nm/1000:.5f} um/px)')
    lines.append(f'  field of view: {fov_um:.3f} um wide')
    lines.append(f'  reference:     JEOL {r["ref_mm"]} mm polaroid')
    if j:
        acq = ', '.join(f'{k}={v}' for k, v in j.items() if k != 'Mag')
        lines.append(f'  acquisition:   Mag={mag}x, {acq}')
    return '\n'.join(lines)


def format_summary(results):
    """One-line-per-file summary for a batch."""
    rows = []
    for r in results:
        name = Path(r['path']).name
        if r['pixel_size_nm'] is None:
            rows.append(f'{name}\t-\t-\t-\tno mag')
        else:
            rows.append(
                f'{name}\t{int(r["mag"])}x'
                f'\t{r["pixel_size_nm"]:.4f} nm/px'
                f'\t{r["field_width_um"]:.3f} um FOV'
            )
    header = 'file\tmag\tpixel_size\tFOV'
    return '\n'.join([header, *rows])


# ---------------------------------------------------------- path expansion
def expand_paths(paths):
    """Turn the mix of files and folders on the CLI into a flat list of tiffs."""
    out = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for pattern in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
                out.extend(sorted(pp.glob(pattern)))
        elif pp.is_file():
            out.append(pp)
        else:
            print(f'{p}: not found, skipping', file=sys.stderr)
    # de-duplicate while preserving order (Windows case-insensitive globs may
    # produce duplicates for *.tif vs *.TIF depending on filesystem)
    seen, uniq = set(), []
    for p in out:
        k = str(p).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


# ---------------------------------------------------------- inspect mode
def inspect(path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        iptc = page.tags.get('IPTCNAA') or page.tags.get(33723)
        if not iptc:
            print(f'{path}: no IPTCNAA tag')
            return
        keys = parse_iptc_keywords(bytes(iptc.value))
        print(f'{path}: IPTC keywords ({len(keys)}):')
        for k in keys:
            print(f'    {k}')


# ---------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('paths', nargs='+',
                   help='one or more TIFF files and/or folders of TIFFs')
    p.add_argument('-o', '--outdir', default=None,
                   help='write one .txt per image plus summary.txt here. '
                        'Without this, results are printed to stdout only.')
    p.add_argument('--ref-mm', type=float, default=JEOL_REF_MM_DEFAULT,
                   help=f'JEOL polaroid reference in mm '
                        f'(default {JEOL_REF_MM_DEFAULT})')
    p.add_argument('--inspect', action='store_true',
                   help='dump raw IPTC keywords instead of computing scale')
    args = p.parse_args()

    files = expand_paths(args.paths)
    if not files:
        print('No TIFFs found.', file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        for f in files:
            inspect(f)
            print()
        return

    results = []
    for f in files:
        try:
            r = analyze_tiff(f, ref_mm=args.ref_mm)
        except Exception as e:
            print(f'{f}: ERROR - {e}', file=sys.stderr)
            continue
        results.append(r)
        text = format_result(r)
        print(text)
        print()
        if outdir:
            (outdir / (f.stem + '.txt')).write_text(text + '\n',
                                                    encoding='utf-8')

    if outdir and results:
        summary = format_summary(results)
        (outdir / 'summary.txt').write_text(summary + '\n', encoding='utf-8')
        print(f'\nWrote {len(results)} per-image .txt files + summary.txt '
              f'to {outdir}')


if __name__ == '__main__':
    main()
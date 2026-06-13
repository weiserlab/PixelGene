#!/usr/bin/env python3
"""
Compare colorized output images against ground-truth color images and produce a
self-contained HTML report (thumbnails embedded as base64, sortable columns,
search filter, summary footer, modal viewer).

Folder structure (mirrors your setup):
    <pred_dir>/<...subfolders...>/img.png   e.g. output/colorisation_size512/Apple/0001.png
    <gt_dir>/<...subfolders...>/img.jpg      e.g. datasets/test_color/Apple/0001.jpg

Pairs are matched by RELATIVE PATH + filename stem (extension-agnostic), so
Apple/0001.png is compared with Apple/0001.jpg.

Per-image metrics (all vs the matched ground truth):
    PSNR        - pixel fidelity (dB, higher better)
    SSIM        - structural / detail fidelity (higher better)
    GMSD        - gradient magnitude similarity deviation (lower better)
    HueDiff     - mean hue angular difference, degrees 0-180 (lower better)
    SatDiff     - mean saturation difference, pred - gt (closer to 0 better)
    BriDiff     - mean brightness (V) difference, pred - gt (closer to 0 better)
    CFI         - Hasler-Susstrunk colorfulness of the prediction
    Sharpness   - variance of Laplacian of the prediction (detail proxy)

Usage:
    python compare_colorization_html.py \
        --pred_dir output/colorisation_size512 \
        --gt_dir   datasets/test_color \
        --out_html comparison_report.html \
        --thumb_w  120
"""

import os
import io
import glob
import math
import base64
import argparse
import json

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


# ----------------------------- IO / matching -----------------------------

def list_images(root):
    """Return {relative_stem (posix): full_path} for every image under root."""
    out = {}
    for f in glob.glob(os.path.join(root, '**', '*.*'), recursive=True):
        if not f.lower().endswith(IMG_EXTS):
            continue
        rel = os.path.relpath(f, root)
        stem = os.path.splitext(rel)[0].replace('\\', '/')
        out[stem] = f
    return out


def read_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'could not read image: {path}')
    return img


def thumb_b64(bgr, width):
    """Resize to given width (keep aspect) and return base64 JPEG data URI."""
    h, w = bgr.shape[:2]
    if w > width:
        nh = max(1, int(round(h * width / w)))
        bgr = cv2.resize(bgr, (width, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ''
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def path_to_file_url(abspath):
    """Convert an absolute filesystem path to a file:// URL the browser can load.

    Handles Windows backslashes/drive letters and percent-encodes spaces and
    other unsafe characters in the path.
    """
    from urllib.request import pathname2url
    return 'file:///' + pathname2url(abspath).lstrip('/')


def parse_caption_log(log_path, pred_dir=None):
    """Parse a run log into {stem: caption}.

    Each set in the log looks like:
        <caption line>
        [Tiled VAE]: ...
        [Tiled VAE]: Done in ...
        saved <path/to/image.png>      (or 'image saved at <path>')

    The caption for an image is the last line before its 'saved' line that is
    NOT a [Tiled VAE] line and NOT a saved line. Keyed by the saved path's stem
    relative to pred_dir (folder/filename-without-ext, posix) so it matches the
    report's row keys. Handles './' segments and loose (root-level) images.
    """
    import re

    def norm_parts(path):
        s = path.replace('\\', '/')
        out = []
        for p in s.split('/'):
            if p in ('', '.'):
                continue
            if p == '..':
                if out:
                    out.pop()
            else:
                out.append(p)
        return out

    pred_parts = norm_parts(pred_dir) if pred_dir else []

    def to_stem(path):
        parts = norm_parts(path)
        if not parts:
            return None
        parts[-1] = os.path.splitext(parts[-1])[0]
        # anchor on the LAST component of pred_dir (the output-root folder name),
        # which appears in the log path regardless of abs/rel prefix differences;
        # everything after that anchor is the stem.
        if pred_parts:
            anchor = pred_parts[-1]
            # search from the right so a repeated name resolves to the deepest match
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == anchor:
                    rel = parts[i + 1:]
                    return '/'.join(rel) if rel else None
        # fallback: folder/filename (or just filename if at root)
        if len(parts) >= 2:
            return f'{parts[-2]}/{parts[-1]}'
        return parts[-1]

    captions = {}
    last_caption = None
    save_re = re.compile(r'(?:saved|image saved at)\s+(.+\.\w+)\s*$', re.IGNORECASE)
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\n').strip()
            if not line:
                continue
            m = save_re.search(line)
            if m:
                path = m.group(1).strip().strip('"').strip("'")
                stem = to_stem(path)
                if stem and last_caption is not None:
                    captions[stem] = last_caption
                last_caption = None  # consumed; next set starts fresh
                continue
            if line.startswith('[Tiled VAE]'):
                continue
            last_caption = line
    return captions


# ----------------------------- metrics -----------------------------

def circular_hue_mean(hue_deg):
    rad = np.deg2rad(hue_deg.astype(np.float64))
    return math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0


def hue_diff_deg(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180 else 360 - d


def colorfulness(bgr):
    b, g, r = cv2.split(bgr.astype(np.float64))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return math.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * math.sqrt(rg.mean()**2 + yb.mean()**2)


def sharpness(bgr):
    gray = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def gmsd(pred_bgr, gt_bgr):
    """Gradient Magnitude Similarity Deviation on luminance (lower = better)."""
    p = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gp_x = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=3)
    gp_y = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=3)
    gg_x = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gg_y = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mp = np.sqrt(gp_x**2 + gp_y**2)
    mg = np.sqrt(gg_x**2 + gg_y**2)
    c = 0.0026 * (255**2)
    gms = (2 * mp * mg + c) / (mp**2 + mg**2 + c)
    return float(np.std(gms))


def lab_scaled(bgr):
    """OpenCV 8-bit LAB -> true CIELAB units (L 0-100, a/b ~ -128..127)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[:, :, 0] * 100.0 / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    return L, a, b


def chroma_psnr(pred_bgr, gt_bgr):
    """PSNR computed on the color (a,b) channels only, ignoring luminance.

    a,b are mapped back to 8-bit [0,255] range so the PSNR is on a familiar
    scale; higher = closer color.
    """
    p_lab = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    g_lab = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    p_ab = p_lab[:, :, 1:3]
    g_ab = g_lab[:, :, 1:3]
    mse = np.mean((p_ab - g_ab) ** 2)
    if mse <= 1e-9:
        return 99.0
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def compare_pair(pred_bgr, gt_bgr):
    if pred_bgr.shape[:2] != gt_bgr.shape[:2]:
        pred_bgr = cv2.resize(pred_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]),
                              interpolation=cv2.INTER_AREA)

    # color-channel PSNR (a,b only)
    cpsnr = chroma_psnr(pred_bgr, gt_bgr)

    # LAB perceptual color difference
    pL, pa, pb = lab_scaled(pred_bgr)
    gL, ga, gb = lab_scaled(gt_bgr)
    # mean deltaE (full CIE76: includes L, but L is ~unchanged so this is
    # effectively chroma-driven; gives the perceptual color distance per pixel)
    dE = np.sqrt((pL - gL) ** 2 + (pa - ga) ** 2 + (pb - gb) ** 2)
    mean_de = float(dE.mean())
    # a/b mean absolute error (pure color axes, luminance excluded)
    ab_mae = float((np.abs(pa - ga) + np.abs(pb - gb)).mean() / 2.0)

    # HSV-based descriptive color stats
    p_hsv = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    g_hsv = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)

    return {
        'ColorPSNR': round(cpsnr, 3),
        'DeltaE': round(mean_de, 3),
        'abMAE': round(ab_mae, 3),
        'CFI': round(colorfulness(pred_bgr), 2),
    }


# ----------------------------- HTML -----------------------------

# columns: (key, label, unit, better)  better in {'up','down','zero'}
COLUMNS = [
    ('ColorPSNR', 'Color PSNR', 'dB', 'up'),
    ('DeltaE', '\u0394E (LAB)', '', 'down'),
    ('abMAE', 'ab-MAE', '', 'down'),
    ('CFI', 'Colorfulness', '', 'up'),
]
COL_TITLES = {
    'ColorPSNR': 'PSNR on the LAB a,b (color) channels only \u2014 luminance excluded (higher better)',
    'DeltaE': 'Mean per-pixel CIE76 color difference in LAB (lower = closer color)',
    'abMAE': 'Mean absolute error of the LAB a,b color axes (lower = closer color)',
    'CFI': 'Hasler-Susstrunk colorfulness of the prediction',
}


def build_html(rows, pred_dir, gt_dir):
    keys = [c[0] for c in COLUMNS]

    # summary stats
    def stat(fn, k):
        vals = [r['metrics'][k] for r in rows]
        return fn(vals) if vals else 0.0

    means = {k: round(float(np.mean([r['metrics'][k] for r in rows])), 4) if rows else 0 for k in keys}
    stds = {k: round(float(np.std([r['metrics'][k] for r in rows])), 4) if rows else 0 for k in keys}
    mins = {k: round(stat(min, k), 4) for k in keys}
    maxs = {k: round(stat(max, k), 4) for k in keys}

    better_map = {c[0]: c[3] for c in COLUMNS}

    # header cells
    arrow = {'up': '\u2191', 'down': '\u2193', 'zero': '\u22480'}
    caption_th = '<th style="text-align:left;min-width:180px">Caption (from log)</th>'
    score_th = ('<th class="sortable score-col" data-col="SCORE" '
                'title="Weighted average of normalized metrics (0-100). '
                'Set weights above and click Compute.">Score<span class="unit">/100</span>'
                '<span class="better">\u2191</span><span class="sort-icon">\u21c5</span></th>')
    head_cells = caption_th + score_th + ''.join(
        f'<th class="sortable" data-col="{k}" title="{COL_TITLES[k]}">{lbl}'
        f'<span class="unit">{unit}</span><span class="better">{arrow[b]}</span>'
        f'<span class="sort-icon">\u21c5</span></th>'
        for (k, lbl, unit, b) in COLUMNS
    )

    import html as _html
    # body rows
    body = []
    for i, r in enumerate(rows, 1):
        m = r['metrics']
        cap = _html.escape(r.get('caption', '') or '')
        cells = f'<td class="caption-cell">{cap}</td>'
        cells += '<td class="score-cell" data-val="">\u2014</td>'  # filled in by JS
        for (k, _, _, _) in COLUMNS:
            cells += f'<td data-val="{m[k]}">{m[k]}</td>'
        pred_url = path_to_file_url(r['pred_path'])
        gt_url = path_to_file_url(r['gt_path'])
        body.append(f"""
        <tr>
          <td class="idx-cell">{i}</td>
          <td class="img-cell">
            <div class="pair">
              <div class="pair-item">
                <img src="{r['pred_thumb']}" class="thumb thumb-pred"
                     data-pred="{pred_url}" data-gt="{gt_url}" data-stem="{r['stem']}"
                     title="click to enlarge pred + GT" />
                <div class="cap">pred</div>
              </div>
              <div class="pair-item">
                <img src="{r['gt_thumb']}" class="thumb thumb-gt"
                     data-pred="{pred_url}" data-gt="{gt_url}" data-stem="{r['stem']}"
                     title="click to enlarge pred + GT" />
                <div class="cap">GT</div>
              </div>
            </div>
            <div class="fname">{r['stem']}</div>
            <div class="folder">{r['folder']}</div>
          </td>
          {cells}
        </tr>""")
    body_html = ''.join(body)

    def sum_row(label, d, bold=False):
        style = ' style="font-weight:700;"' if bold else ''
        cells = '<td class="sum-cell"></td>'  # under Caption column
        cells += '<td class="sum-cell"></td>'  # under Score column
        cells += ''.join(f'<td class="sum-cell"{style}>{d[k]}</td>' for k in keys)
        return f'<tr class="sum-row"><td class="sum-label" colspan="2">{label}</td>{cells}</tr>'

    foot_html = (sum_row('Mean', means, True) + sum_row('Std Dev', stds)
                 + sum_row('Min', mins) + sum_row('Max', maxs))

    data_json = json.dumps([dict(file=r['stem'], folder=r['folder'],
                                 caption=r.get('caption', ''), **r['metrics']) for r in rows])
    # td index: 0=#, 1=image, 2=Caption, 3=Score, then metrics start at 4
    colidx = {'SCORE': 3}
    colidx.update({k: i + 4 for i, k in enumerate(keys)})
    colidx_json = json.dumps(colidx)
    lower_json = json.dumps({k: 1 for k in keys if better_map[k] == 'down'})
    zero_json = json.dumps({k: 1 for k in keys if better_map[k] == 'zero'})
    labels_json = json.dumps({c[0]: c[1] for c in COLUMNS})
    ranges_json = json.dumps({k: [mins[k], maxs[k]] for k in keys})

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Colorization vs Ground-Truth Report</title>
<style>
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:13px;background:#f8f9fa;color:#1a1a1a;}}
  .page{{max-width:1600px;margin:0 auto;padding:24px 20px;}}
  .header h1{{font-size:22px;font-weight:700;color:#111;margin-bottom:4px;}}
  .header p{{color:#555;}}
  .meta{{display:flex;gap:24px;margin-top:12px;flex-wrap:wrap;}}
  .meta-item{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px 18px;}}
  .meta-item .val{{font-size:20px;font-weight:700;}}
  .meta-item .lbl{{font-size:11px;color:#888;margin-top:2px;}}
  .controls{{display:flex;gap:12px;margin:16px 0 14px;align-items:center;flex-wrap:wrap;}}
  .controls input{{border:1px solid #d1d5db;border-radius:6px;padding:6px 12px;font-size:13px;width:220px;outline:none;}}
  .controls input:focus{{border-color:#6366f1;}}
  .btn{{padding:6px 14px;border-radius:6px;border:1px solid #d1d5db;background:#fff;cursor:pointer;font-size:12px;color:#374151;}}
  .btn:hover{{background:#f3f4f6;}}
  .btn.active{{background:#6366f1;color:#fff;border-color:#6366f1;}}
  select{{border:1px solid #d1d5db;border-radius:6px;padding:6px 10px;font-size:12px;background:#fff;cursor:pointer;}}
  .table-wrap{{overflow-x:auto;border-radius:10px;border:1px solid #e5e7eb;box-shadow:0 1px 4px rgba(0,0,0,.06);}}
  table{{border-collapse:collapse;width:100%;background:#fff;}}
  thead th{{background:#f1f5f9;border-bottom:2px solid #e2e8f0;padding:10px;text-align:center;font-size:12px;font-weight:600;color:#374151;white-space:nowrap;position:sticky;top:0;z-index:2;user-select:none;}}
  thead th.sortable{{cursor:pointer;}}
  thead th.sortable:hover{{background:#e2e8f0;}}
  thead th.sort-asc .sort-icon::after{{content:" \u25b2";}}
  thead th.sort-desc .sort-icon::after{{content:" \u25bc";}}
  .sort-icon{{color:#9ca3af;font-size:10px;}}
  .unit{{font-size:10px;color:#9ca3af;margin-left:2px;}}
  .better{{font-size:10px;color:#6366f1;margin-left:3px;font-weight:700;}}
  tbody tr{{border-bottom:1px solid #f1f5f9;}}
  tbody tr:hover{{background:#f8fafc;}}
  tbody tr.hidden{{display:none;}}
  td{{padding:8px 10px;text-align:center;font-size:12px;vertical-align:middle;}}
  .idx-cell{{color:#9ca3af;font-size:11px;width:32px;}}
  .img-cell{{text-align:left;min-width:230px;padding:6px 8px;}}
  .pair{{display:flex;gap:6px;}}
  .pair-item{{text-align:center;}}
  .thumb{{width:104px;height:78px;object-fit:cover;border-radius:4px;display:block;cursor:pointer;}}
  .thumb-pred{{border:2px solid #6366f1;}}
  .thumb-gt{{border:2px solid #10b981;}}
  .thumb:hover{{opacity:.85;}}
  .cap{{font-size:9px;color:#888;margin-top:2px;}}
  .fname{{font-size:10px;font-weight:600;margin-top:4px;word-break:break-all;}}
  .folder{{font-size:9px;color:#9ca3af;}}
  .sum-row td{{background:#f8fafc;font-size:11px;color:#374151;border-top:2px solid #e2e8f0;}}
  .sum-label{{text-align:left;font-weight:700;padding-left:16px;color:#6366f1;}}
  .sum-cell{{font-weight:600;}}
  tr.highlight-best{{outline:2px solid #10b981;outline-offset:-1px;}}
  tr.highlight-worst{{outline:2px solid #ef4444;outline-offset:-1px;}}
  .weights{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px;}}
  .wt{{display:flex;align-items:center;gap:6px;background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;padding:5px 10px;font-size:12px;color:#3730a3;}}
  .wt input[type=checkbox]{{cursor:pointer;}}
  .wt input[type=number]{{width:54px;border:1px solid #c7d2fe;border-radius:4px;padding:2px 5px;font-size:12px;text-align:right;}}
  .wt .dirhint{{font-size:10px;color:#6b7280;}}
  .sort-hint{{font-size:11px;color:#9ca3af;margin-bottom:14px;max-width:1000px;line-height:1.5;}}
  .score-col{{background:#1e1b4b !important;color:#c7d2fe !important;}}
  .score-cell{{font-weight:700;color:#3730a3;}}
  .caption-cell{{text-align:left;font-size:11px;color:#374151;min-width:180px;max-width:260px;line-height:1.4;}}
  .legend{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px 16px;margin-bottom:14px;font-size:12px;color:#374151;}}
  .legend-title{{font-weight:700;color:#111;display:block;margin-bottom:6px;}}
  .legend ul{{margin:0;padding-left:18px;}}
  .legend li{{margin-bottom:3px;line-height:1.45;}}
  .legend b{{color:#3730a3;}}
  .thresh-box{{display:inline-flex;align-items:center;gap:6px;background:#fff;border:1px solid #d1d5db;border-radius:6px;padding:4px 10px;font-size:12px;}}
  .thresh-box input[type=number]{{width:56px;border:1px solid #d1d5db;border-radius:4px;padding:3px 5px;font-size:12px;text-align:right;}}
  .thresh-filter{{display:inline-flex;align-items:center;gap:4px;color:#6b7280;cursor:pointer;}}
  .thresh-result{{font-size:13px;margin-bottom:14px;color:#1a1a1a;}}
  .thresh-result b{{color:#3730a3;}}
  .thresh-result .bar{{display:inline-block;vertical-align:middle;width:160px;height:8px;border-radius:4px;background:#e5e7eb;margin-left:10px;overflow:hidden;}}
  .thresh-result .bar > span{{display:block;height:100%;background:#6366f1;}}
  .modal-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;align-items:center;justify-content:center;}}
  .modal-overlay.open{{display:flex;}}
  .modal-box{{background:#fff;border-radius:12px;padding:20px;max-width:95vw;max-height:92vh;overflow:auto;position:relative;}}
  .modal-close{{position:absolute;top:10px;right:14px;font-size:22px;cursor:pointer;color:#555;background:none;border:none;line-height:1;}}
  .modal-title{{font-size:13px;font-weight:600;color:#111;margin-bottom:12px;padding-right:24px;word-break:break-all;}}
  .modal-pair{{display:flex;gap:16px;flex-wrap:wrap;justify-content:center;}}
  .modal-pair figure{{margin:0;text-align:center;}}
  .modal-pair img{{max-width:44vw;max-height:74vh;display:block;border-radius:6px;border:1px solid #e5e7eb;}}
  .modal-pair figcaption{{font-size:12px;color:#555;margin-top:6px;font-weight:600;}}
  .modal-pair figure:first-child img{{border:2px solid #6366f1;}}
  .modal-pair figure:last-child img{{border:2px solid #10b981;}}
</style></head><body>
<div class="page">
  <div class="header">
    <h1>Colorization vs Ground-Truth Report</h1>
    <p>Each colorized output compared against its matching ground-truth color image (paired by folder + filename).</p>
    <div class="meta">
      <div class="meta-item"><div class="val">{len(rows)}</div><div class="lbl">Pairs evaluated</div></div>
      <div class="meta-item"><div class="val">{means['ColorPSNR']}</div><div class="lbl">Mean Color PSNR (dB)</div></div>
      <div class="meta-item"><div class="val">{means['DeltaE']}</div><div class="lbl">Mean \u0394E (LAB)</div></div>
      <div class="meta-item"><div class="val">{means['abMAE']}</div><div class="lbl">Mean ab-MAE</div></div>
    </div>
    <p style="margin-top:10px;font-size:11px;color:#888;">pred = <span style="color:#6366f1;font-weight:700;">blue border</span> &nbsp;|&nbsp; GT = <span style="color:#10b981;font-weight:700;">green border</span> &nbsp;\u00b7&nbsp; click a thumbnail to enlarge \u00b7 click a header to sort</p>
  </div>

  <div class="controls">
    <input type="text" id="search" placeholder="Filter by name/folder\u2026" oninput="filterRows()">
    <button class="btn" onclick="computeScore()">Compute weighted score</button>
    <button class="btn" onclick="resetWeights()">Reset weights</button>
    <span class="thresh-box">
      Score \u2265 <input type="number" id="threshold" value="80" min="0" max="100" step="1">
      <button class="btn" onclick="applyThreshold()">Count</button>
      <label class="thresh-filter"><input type="checkbox" id="thresh-filter-cb" onchange="applyThreshold()"> show only these</label>
    </span>
    <button class="btn" id="btn-best" onclick="highlightBest()">Highlight best</button>
    <button class="btn" id="btn-worst" onclick="highlightWorst()">Highlight worst</button>
    <button class="btn" onclick="clearHighlight()">Clear hl</button>
  </div>

  <div id="thresh-result" class="thresh-result"></div>

  <div id="weights" class="weights"></div>
  <div class="sort-hint">Tick the metrics to include, type a weight for each (any positive numbers; they're normalized so only the ratio matters). Each metric is scaled 0\u20131 across the dataset (lower-is-better and closest-to-zero metrics are flipped so higher is always better), then combined as a weighted average \u00d7100. Click <b>Compute weighted score</b> to fill the Score column and sort by it.</div>

  <div class="legend">
    <span class="legend-title">What the columns mean (all vs ground truth, except Colorfulness):</span>
    <ul>
      <li><b>Color PSNR</b> (dB, \u2191 better) \u2014 fidelity of the predicted colors on the LAB a,b channels only; luminance excluded.</li>
      <li><b>\u0394E (LAB)</b> (\u2193 better) \u2014 mean per-pixel perceptual color difference; the closest single number to \u201chow different the colors look.\u201d</li>
      <li><b>ab-MAE</b> (\u2193 better) \u2014 mean absolute error of the LAB a,b color axes; a simpler, outlier-tolerant cousin of \u0394E.</li>
      <li><b>Colorfulness</b> (\u2191 = more vivid) \u2014 saturation/vividness of the prediction itself (no ground-truth comparison); describes character, not correctness.</li>
    </ul>
  </div>

  <div class="table-wrap"><table id="main-table">
    <thead><tr>
      <th style="width:32px">#</th>
      <th style="text-align:left;min-width:230px">Prediction vs Ground Truth</th>
      {head_cells}
    </tr></thead>
    <tbody id="table-body">{body_html}</tbody>
    <tfoot>{foot_html}</tfoot>
  </table></div>
</div>

<div class="modal-overlay" id="modal" onclick="closeModal(event)">
  <div class="modal-box">
    <button class="modal-close" onclick="document.getElementById('modal').classList.remove('open')">\u2715</button>
    <div class="modal-title" id="modal-title"></div>
    <div class="modal-pair">
      <figure>
        <img id="modal-pred" src="" alt="prediction">
        <figcaption>Prediction</figcaption>
      </figure>
      <figure>
        <img id="modal-gt" src="" alt="ground truth">
        <figcaption>Ground Truth</figcaption>
      </figure>
    </div>
  </div>
</div>

<script>
const DATA={data_json};
const COL_IDX={colidx_json};
const BETTER_LOWER={lower_json};
const BETTER_ZERO={zero_json};
const LABELS={labels_json};
const RANGES={ranges_json};   // {{metric:[min,max]}} across dataset

function filterRows(){{
  const q=document.getElementById('search').value.toLowerCase();
  const cb=document.getElementById('thresh-filter-cb');
  const ci=COL_IDX['SCORE'];
  const thr=parseFloat(document.getElementById('threshold').value);
  const useThresh=cb && cb.checked && !isNaN(thr);
  document.querySelectorAll('#table-body tr').forEach(tr=>{{
    const name=((tr.querySelector('.fname')?.textContent||'')+' '+(tr.querySelector('.folder')?.textContent||'')+' '+(tr.querySelector('.caption-cell')?.textContent||'')).toLowerCase();
    const okSearch=!q||name.includes(q);
    let okScore=true;
    if(useThresh){{
      const raw=tr.querySelectorAll('td')[ci]?.dataset?.val;
      const v=(raw===''||raw==null)?-Infinity:parseFloat(raw);
      okScore=v>=thr;
    }}
    tr.classList.toggle('hidden', !(okSearch&&okScore));
  }});
}}
let sortState={{}};

// normalize one metric value to 0..1 "goodness" (higher = better) using
// dataset min/max, flipping lower-better and zero-better metrics.
function goodness(col, v){{
  const [lo,hi]=RANGES[col];
  if(hi===lo) return 1;                       // constant column -> neutral
  if(BETTER_ZERO[col]){{
    // closest to 0 is best; goodness by absolute distance from 0 over the
    // largest absolute bound in the dataset
    const bound=Math.max(Math.abs(lo),Math.abs(hi))||1;
    return 1 - Math.min(1, Math.abs(v)/bound);
  }}
  let t=(v-lo)/(hi-lo);                        // 0..1, higher raw value
  if(BETTER_LOWER[col]) t=1-t;                 // flip so higher = better
  return t;
}}

function buildWeightPanel(){{
  const box=document.getElementById('weights');
  box.innerHTML='';
  Object.keys(LABELS).forEach(col=>{{
    const dir = BETTER_ZERO[col] ? '\u22480 best' : (BETTER_LOWER[col] ? 'lower best' : 'higher best');
    const w=document.createElement('label');
    w.className='wt';
    w.innerHTML=
      `<input type="checkbox" data-col="${{col}}" checked> `+
      `<span>${{LABELS[col]}}</span>`+
      `<input type="number" data-wcol="${{col}}" value="1" min="0" step="0.1">`+
      `<span class="dirhint">${{dir}}</span>`;
    box.appendChild(w);
  }});
}}

function resetWeights(){{
  document.querySelectorAll('#weights input[type=checkbox]').forEach(c=>c.checked=true);
  document.querySelectorAll('#weights input[type=number]').forEach(n=>n.value='1');
}}

function currentWeights(){{
  const out={{}};
  document.querySelectorAll('#weights .wt').forEach(w=>{{
    const cb=w.querySelector('input[type=checkbox]');
    const num=w.querySelector('input[type=number]');
    const col=cb.dataset.col;
    if(cb.checked){{
      const val=parseFloat(num.value);
      if(!isNaN(val) && val>0) out[col]=val;
    }}
  }});
  return out;
}}

function computeScore(){{
  const weights=currentWeights();
  const cols=Object.keys(weights);
  if(cols.length===0){{ alert('Tick at least one metric with a positive weight.'); return; }}
  const wsum=cols.reduce((s,c)=>s+weights[c],0);

  const tb=document.getElementById('table-body');
  const rows=Array.from(tb.querySelectorAll('tr'));
  rows.forEach(r=>{{
    const tds=r.querySelectorAll('td');
    let acc=0;
    cols.forEach(c=>{{
      const v=parseFloat(tds[COL_IDX[c]]?.dataset?.val??NaN);
      const g=isNaN(v)?0:goodness(c,v);
      acc+=weights[c]*g;
    }});
    const score=100*acc/wsum;
    const cell=tds[COL_IDX['SCORE']];
    cell.dataset.val=score.toFixed(2);
    cell.textContent=score.toFixed(1);
  }});
  // sort by score desc
  rows.sort((a,b)=>parseFloat(b.querySelectorAll('td')[COL_IDX['SCORE']].dataset.val)
                  -parseFloat(a.querySelectorAll('td')[COL_IDX['SCORE']].dataset.val));
  rows.forEach(r=>tb.appendChild(r));
  document.querySelectorAll('thead th.sortable').forEach(th=>th.classList.remove('sort-asc','sort-desc'));
  document.querySelector('thead th[data-col="SCORE"]')?.classList.add('sort-desc');
  applyThreshold();
}}

function applyThreshold(){{
  const ci=COL_IDX['SCORE'];
  const thr=parseFloat(document.getElementById('threshold').value);
  const allRows=Array.from(document.querySelectorAll('#table-body tr'));
  // only consider rows that have a computed score
  const scored=allRows.filter(r=>{{
    const raw=r.querySelectorAll('td')[ci]?.dataset?.val;
    return raw!=='' && raw!=null;
  }});
  const res=document.getElementById('thresh-result');
  if(scored.length===0){{ res.innerHTML='Compute the weighted score first, then set a threshold.'; return; }}
  if(isNaN(thr)){{ res.innerHTML='Enter a numeric threshold.'; return; }}

  let pass=0;
  scored.forEach(r=>{{
    const v=parseFloat(r.querySelectorAll('td')[ci].dataset.val);
    if(v>=thr) pass++;
  }});
  const total=scored.length;
  const pct=total? (100*pass/total) : 0;

  // apply visibility (search + optional threshold filter handled together)
  filterRows();

  res.innerHTML=`<b>${{pass}}</b> of <b>${{total}}</b> images score \u2265 <b>${{thr}}</b> `+
                `(<b>${{pct.toFixed(1)}}%</b>)`+
                `<span class="bar"><span style="width:${{pct.toFixed(1)}}%"></span></span>`;
}}

// header click = single-column sort
function sortTable(col){{
  const tb=document.getElementById('table-body');
  const rows=Array.from(tb.querySelectorAll('tr'));
  const asc=sortState[col]!=='asc';
  sortState={{[col]:asc?'asc':'desc'}};
  document.querySelectorAll('thead th.sortable').forEach(th=>{{
    th.classList.remove('sort-asc','sort-desc');
    if(th.dataset.col===col) th.classList.add(asc?'sort-asc':'sort-desc');
  }});
  const ci=COL_IDX[col];
  rows.sort((a,b)=>{{
    const fa=parseFloat(a.querySelectorAll('td')[ci]?.dataset?.val??NaN);
    const fb=parseFloat(b.querySelectorAll('td')[ci]?.dataset?.val??NaN);
    const va=isNaN(fa)?-Infinity:fa, vb=isNaN(fb)?-Infinity:fb;
    return asc?va-vb:vb-va;
  }});
  rows.forEach(r=>tb.appendChild(r));
}}
document.querySelectorAll('thead th.sortable').forEach(th=>th.addEventListener('click',()=>sortTable(th.dataset.col)));

function pickExtreme(best){{
  clearHighlight();
  const ci=COL_IDX['SCORE'];
  const rows=Array.from(document.querySelectorAll('#table-body tr:not(.hidden)'));
  let target=null,tv=null,any=false;
  rows.forEach(r=>{{
    const raw=r.querySelectorAll('td')[ci]?.dataset?.val;
    if(raw==='' || raw==null) return;
    any=true;
    let v=parseFloat(raw);
    let s=best?v:-v;
    if(tv===null||s>tv){{tv=s;target=r;}}
  }});
  if(!any){{ alert('Compute the weighted score first.'); return; }}
  target?.classList.add(best?'highlight-best':'highlight-worst');
  document.getElementById(best?'btn-best':'btn-worst').classList.add('active');
}}
function highlightBest(){{pickExtreme(true);}}
function highlightWorst(){{pickExtreme(false);}}
function clearHighlight(){{
  document.querySelectorAll('.highlight-best,.highlight-worst').forEach(r=>r.classList.remove('highlight-best','highlight-worst'));
  document.querySelectorAll('.btn.active').forEach(b=>b.classList.remove('active'));
}}

buildWeightPanel();

function openPairModal(predSrc,gtSrc,stem){{
  const p=document.getElementById('modal-pred');
  const g=document.getElementById('modal-gt');
  p.onerror=()=>{{p.alt='could not load file:\\n'+predSrc;}};
  g.onerror=()=>{{g.alt='could not load file:\\n'+gtSrc;}};
  p.alt='prediction'; g.alt='ground truth';
  p.src=predSrc; g.src=gtSrc;
  document.getElementById('modal-title').textContent=stem;
  document.getElementById('modal').classList.add('open');
}}
function closeModal(e){{if(e.target.id==='modal')document.getElementById('modal').classList.remove('open');}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')document.getElementById('modal').classList.remove('open');}});
document.querySelectorAll('.thumb').forEach(img=>img.addEventListener('click',()=>{{
  openPairModal(img.dataset.pred, img.dataset.gt, img.dataset.stem);
}}));
</script>
</body></html>"""


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', required=True, help='colorized output root, e.g. output/colorisation_size512')
    ap.add_argument('--gt_dir', required=True, help='ground-truth color root, e.g. datasets/test_color')
    ap.add_argument('--out_html', default='comparison_report.html')
    ap.add_argument('--thumb_w', type=int, default=120, help='thumbnail width in px')
    ap.add_argument('--log', default=None, help='optional run log (e.g. pasd.o1150650) to extract per-image captions')
    args = ap.parse_args()

    preds = list_images(args.pred_dir)
    gts = list_images(args.gt_dir)
    common = sorted(set(preds) & set(gts))
    only_pred = sorted(set(preds) - set(gts))
    only_gt = sorted(set(gts) - set(preds))

    print(f'predictions: {len(preds)} | ground truth: {len(gts)} | matched pairs: {len(common)}')
    if only_pred:
        print(f'  {len(only_pred)} predictions with NO matching GT (first few): {only_pred[:5]}')
    if only_gt:
        print(f'  {len(only_gt)} GT images with NO matching prediction (first few): {only_gt[:5]}')

    captions = {}
    if args.log:
        captions = parse_caption_log(args.log, args.pred_dir)
        matched_caps = sum(1 for s in common if s in captions)
        print(f'captions parsed from log: {len(captions)} | matched to pairs: {matched_caps}')

    rows = []
    for stem in common:
        try:
            pred = read_bgr(preds[stem])
            gt = read_bgr(gts[stem])
            metrics = compare_pair(pred, gt)
        except Exception as e:
            print(f'  SKIP {stem}: {e}')
            continue
        rows.append({
            'stem': stem,
            'folder': os.path.dirname(stem) or '(root)',
            'metrics': metrics,
            'caption': captions.get(stem, ''),
            'pred_thumb': thumb_b64(pred, args.thumb_w),
            'gt_thumb': thumb_b64(gt, args.thumb_w),
            'pred_path': os.path.abspath(preds[stem]),
            'gt_path': os.path.abspath(gts[stem]),
        })

    if not rows:
        print('No comparable pairs found. Check --pred_dir / --gt_dir and folder structure.')
        # still write a minimal report so you can see the matching summary
    html = build_html(rows, args.pred_dir, args.gt_dir)
    with open(args.out_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\nReport written to: {args.out_html}  ({len(rows)} rows)')


if __name__ == '__main__':
    main()
"""
=============================================================
  Batch Image Quality Evaluator — Single LR → Many HR
  One LR grayscale image as reference; evaluates every HR
  output in a folder against it. Useful for comparing
  multiple diffusion model runs / seeds on the same input.
=============================================================
Usage:
    python batch_evaluate.py --lr  input_gray.png \
                             --hr_dir ./hr_outputs \
                             --report report.html

  Every image found in --hr_dir is evaluated against the
  single --lr reference image.

  Accepted extensions: .png .jpg .jpeg .bmp .tif .tiff .webp

Optional:
    --workers N          parallel workers (default: 4)
    --thumb_size WxH     thumbnail size in report (default: 120x80)
    --no_brisque         skip BRISQUE (faster, no brisque package needed)

Install:
    pip install opencv-python numpy scipy scikit-image matplotlib tabulate tqdm
    pip install brisque   # optional
"""

import argparse
import base64
import concurrent.futures
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import entropy as scipy_entropy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

warnings.filterwarnings("ignore")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ─────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────

def load_lr(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    return img

def load_hr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    return img

def resize_to(src, h, w):
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC)

def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def thumb_b64(hr_bgr, tw, th):
    """Return base64-encoded JPEG thumbnail of HR image."""
    t = cv2.resize(hr_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()

def lr_thumb_b64(lr_gray, tw, th):
    lr_up = cv2.resize(lr_gray, (tw, th), interpolation=cv2.INTER_CUBIC)
    _, buf = cv2.imencode(".jpg", lr_up, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()

# ─────────────────────────────────────────────────────────────────
# Metrics (same as single-image script)
# ─────────────────────────────────────────────────────────────────

def m_psnr(lr, hr):
    lr_up = resize_to(lr, *hr.shape[:2])
    hr_y  = to_gray(hr)
    return round(float(psnr(lr_up, hr_y, data_range=255)), 3)

def m_ssim(lr, hr):
    lr_up = resize_to(lr, *hr.shape[:2])
    hr_y  = to_gray(hr)
    return round(float(ssim(lr_up, hr_y, data_range=255)), 4)

def m_gmsd(lr, hr):
    lr_up = resize_to(lr, *hr.shape[:2]).astype(np.float32)
    hr_y  = to_gray(hr).astype(np.float32)
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32) / 3.
    ky = kx.T
    gm_lr = np.sqrt(cv2.filter2D(lr_up,-1,kx)**2 + cv2.filter2D(lr_up,-1,ky)**2)
    gm_hr = np.sqrt(cv2.filter2D(hr_y,-1,kx)**2  + cv2.filter2D(hr_y,-1,ky)**2)
    c = 170.
    gms = (2*gm_lr*gm_hr + c) / (gm_lr**2 + gm_hr**2 + c)
    return round(float(np.std(gms)), 5)

def m_eps(lr, hr):
    lr_up = resize_to(lr, *hr.shape[:2])
    hr_y  = to_gray(hr)
    def sobel(img):
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sx**2 + sy**2).ravel()
    a, b = sobel(lr_up), sobel(hr_y)
    return round(float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8)), 4)

def m_cfi(hr):
    f = hr.astype(np.float32)
    R,G,B = f[:,:,2], f[:,:,1], f[:,:,0]
    rg, yb = R-G, 0.5*(R+G)-B
    return round(float(np.std(rg)+np.std(yb)+0.3*np.sqrt(np.mean(rg)**2+np.mean(yb)**2)), 2)

def m_msat(hr):
    hsv = cv2.cvtColor(hr, cv2.COLOR_BGR2HSV)
    return round(float(np.mean(hsv[:,:,1])), 2)

def m_hue_ent(hr):
    hsv = cv2.cvtColor(hr, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0]
    sat = hsv[:,:,1]
    mask = sat > 20
    if mask.sum() < 100:
        return 0.0
    h, _ = np.histogram(hue[mask], bins=180, range=(0,180), density=True)
    h = h[h > 0]
    return round(float(scipy_entropy(h, base=2)), 3)

def m_cni(hr):
    hsv = cv2.cvtColor(hr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[:,:,0], hsv[:,:,1]/255., hsv[:,:,2]/255.
    sm = S > 0.10
    skin    = sm & (H>=4)  & (H<=18)  & (S>=0.15) & (S<=0.72)
    foliage = sm & (H>=30) & (H<=70)  & (S>=0.15)
    sky     = sm & (H>=90) & (H<=130) & (S>=0.20)  & (V>=0.35)
    n = sm.sum() + 1
    sp = max(0, np.mean(S[sm]) - 0.80) if sm.any() else 0
    return round(float(min(1.0,(skin.sum()*3+foliage.sum()*2+sky.sum()*2)/n+0.25-sp)), 4)

def m_brisque(hr):
    try:
        from brisque import BRISQUE
        rgb = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        return round(float(BRISQUE().score(rgb)), 2)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────
# Per-image worker
# ─────────────────────────────────────────────────────────────────

def evaluate_pair(lr_path, hr_path, tw, th, use_brisque):
    result = {"file": Path(hr_path).name, "error": None}
    try:
        lr = load_lr(lr_path)
        hr = load_hr(hr_path)
        result["lr_thumb"] = lr_thumb_b64(lr, tw, th)
        result["hr_thumb"] = thumb_b64(hr, tw, th)
        result["lr_size"]  = f"{lr.shape[1]}×{lr.shape[0]}"
        result["hr_size"]  = f"{hr.shape[1]}×{hr.shape[0]}"

        result["PSNR"]    = m_psnr(lr, hr)
        result["SSIM"]    = m_ssim(lr, hr)
        result["GMSD"]    = m_gmsd(lr, hr)
        result["EPS"]     = m_eps(lr, hr)
        result["CFI"]     = m_cfi(hr)
        result["MSAT"]    = m_msat(hr)
        result["HUE_ENT"] = m_hue_ent(hr)
        result["CNI"]     = m_cni(hr)
        result["BRISQUE"] = m_brisque(hr) if use_brisque else None
        result["SCORE"]   = compute_weighted_score(result)
    except Exception as e:
        result["error"] = str(e)
    return result

# ─────────────────────────────────────────────────────────────────
# Rating helpers
# ─────────────────────────────────────────────────────────────────

def rating(name, v):
    if v is None: return "gray"
    try: v = float(v)
    except: return "gray"
    thresholds = {
        "PSNR":    [(40,"green"),(30,"blue"),(25,"amber"),(0,"red")],
        "SSIM":    [(0.90,"green"),(0.75,"blue"),(0.60,"amber"),(0,"red")],
        "GMSD":    [(1,"green")] if v<=0.03 else [(1,"blue")] if v<=0.06 else [(1,"amber")] if v<=0.10 else [(1,"red")],
        "EPS":     [(0.97,"green"),(0.93,"blue"),(0.85,"amber"),(0,"red")],
        "CFI":     [(59,"green"),(45,"blue"),(33,"amber"),(0,"red")],
        "MSAT":    [(140,"green"),(90,"blue"),(50,"amber"),(0,"red")],
        "HUE_ENT": [(5.24,"green"),(3.74,"blue"),(2.24,"amber"),(0,"red")],
        "CNI":     [(0.50,"green"),(0.35,"blue"),(0.20,"amber"),(0,"red")],
        "BRISQUE": [(1,"green")] if v<=20 else [(1,"blue")] if v<=40 else [(1,"amber")] if v<=60 else [(1,"red")],
    }
    if name == "GMSD" or name == "BRISQUE":
        # lower is better — already handled above
        return thresholds[name][0][1] if name in thresholds else "gray"
    rules = thresholds.get(name, [])
    for threshold, color in rules:
        if v >= threshold:
            return color
    return "red"

def fmt(name, v):
    if v is None: return "—"
    return str(v)

# ─────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────

METRIC_KEYS = ["PSNR","SSIM","GMSD","EPS","CFI","MSAT","HUE_ENT","CNI","BRISQUE"]

# Weights must sum to 1.0 (BRISQUE weight redistributed if score is None)
WEIGHTS = {
    "PSNR":    0.20,  # detail fidelity — luminance
    "SSIM":    0.20,  # structural similarity
    "GMSD":    0.10,  # fine detail / sharpness
    "EPS":     0.10,  # edge preservation
    "CFI":     0.12,  # color vividness
    "MSAT":    0.10,  # saturation
    "HUE_ENT": 0.08,  # color richness
    "CNI":     0.05,  # color naturalness
    "BRISQUE": 0.05,  # perceptual quality
}

# Normalise each metric to 0-1 (higher = better always after this)
NORM = {
    "PSNR":    lambda v: min(float(v) / 50.0, 1.0),
    "SSIM":    lambda v: max(float(v), 0.0),
    "GMSD":    lambda v: max(1.0 - float(v) / 0.15, 0.0),   # invert, lower is better
    "EPS":     lambda v: max(float(v), 0.0),
    "CFI":     lambda v: min(float(v) / 82.0, 1.0),
    "MSAT":    lambda v: min(float(v) / 255.0, 1.0),
    "HUE_ENT": lambda v: min(float(v) / 7.49, 1.0),
    "CNI":     lambda v: max(float(v), 0.0),
    "BRISQUE": lambda v: max(1.0 - float(v) / 100.0, 0.0),  # invert, lower is better
}

def compute_weighted_score(r):
    """Compute a 0-100 weighted average quality score for a result row."""
    total_w = 0.0
    total_s = 0.0
    for k, w in WEIGHTS.items():
        v = r.get(k)
        if v is None:
            continue  # skip missing metrics, redistribute weight implicitly
        norm = NORM[k](v)
        total_s += norm * w
        total_w += w
    if total_w == 0:
        return None
    return round((total_s / total_w) * 100, 2)  # scale to 0-100

def compute_summary(rows):
    summary = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if r.get("error") is None and r.get(k) is not None]
        if not vals:
            summary[k] = {"mean":None,"min":None,"max":None,"std":None}
            continue
        arr = np.array(vals, dtype=np.float64)
        summary[k] = {
            "mean": round(float(np.mean(arr)),4),
            "min":  round(float(np.min(arr)),4),
            "max":  round(float(np.max(arr)),4),
            "std":  round(float(np.std(arr)),4),
        }
    return summary

# ─────────────────────────────────────────────────────────────────
# HTML report generator
# ─────────────────────────────────────────────────────────────────

METRIC_INFO = {
    "PSNR":    {"label":"PSNR",    "unit":"dB",   "better":"↑","tip":"Luminance pixel fidelity vs upscaled LR"},
    "SSIM":    {"label":"SSIM",    "unit":"",     "better":"↑","tip":"Structural similarity (luminance)"},
    "GMSD":    {"label":"GMSD",    "unit":"",     "better":"↓","tip":"Gradient magnitude deviation — fine detail"},
    "EPS":     {"label":"EPS",     "unit":"",     "better":"↑","tip":"Edge preservation (Sobel cosine similarity)"},
    "CFI":     {"label":"CFI",     "unit":"",     "better":"↑","tip":"Colorfulness index (Hasler & Süsstrunk 2003)"},
    "MSAT":    {"label":"MSAT",    "unit":"/255", "better":"↑","tip":"Mean HSV saturation — color vividness"},
    "HUE_ENT": {"label":"Hue Ent","unit":"bits", "better":"↑","tip":"Hue diversity — palette richness"},
    "CNI":     {"label":"CNI",     "unit":"",     "better":"↑","tip":"Color naturalness (skin/foliage/sky)"},
    "BRISQUE": {"label":"BRISQUE","unit":"",      "better":"↓","tip":"Blind perceptual quality (lower = better)"},
}

def generate_html(rows, summary, tw, th, elapsed, lr_path=None):
    ok_rows   = [r for r in rows if not r.get("error")]
    fail_rows = [r for r in rows if r.get("error")]

    # Build the shared LR reference thumbnail shown once at the top
    lr_ref_html = ""
    if lr_path is not None and ok_rows and ok_rows[0].get("lr_thumb"):
        lr_b64   = ok_rows[0]["lr_thumb"]
        lr_size  = ok_rows[0].get("lr_size", "")
        lr_name  = Path(lr_path).name
        lr_ref_html = f"""
        <div class="lr-ref-box">
          <div class="lr-ref-label">LR reference image (shared across all rows)</div>
          <div class="lr-ref-inner">
            <img src="data:image/jpeg;base64,{lr_b64}"
                 class="lr-ref-img" title="{lr_name}" id="lr-ref-img"
                 onclick="openModal(this.src, this.title)">
            <div>
              <div class="lr-ref-name">{lr_name}</div>
              <div class="lr-ref-size">{lr_size}</div>
              <div class="lr-ref-note">All HR outputs are evaluated against this image</div>
            </div>
          </div>
        </div>"""

    # colour map used both in CSS vars and JS
    COLOR_CSS = {
        "green": ("#1a7a4a","#d4f0e1"),
        "blue":  ("#1a5fa8","#d2e8fb"),
        "amber": ("#8a5e00","#fef3c7"),
        "red":   ("#9b1c1c","#fee2e2"),
        "gray":  ("#555555","#f3f4f6"),
    }

    def cell_style(name, v):
        c = rating(name, v)
        fg, bg = COLOR_CSS.get(c, COLOR_CSS["gray"])
        return f'style="background:{bg};color:{fg};font-weight:600;"'

    def spark_bar(name, v, s):
        """Small horizontal bar relative to dataset range."""
        mn = s.get("min"); mx = s.get("max")
        if mn is None or mx is None or v is None: return ""
        rng = mx - mn
        if rng < 1e-9: pct = 100
        else: pct = int(100*(float(v)-mn)/rng)
        c = rating(name, v)
        fg, _ = COLOR_CSS.get(c, COLOR_CSS["gray"])
        return (f'<div style="height:4px;border-radius:2px;background:#e5e7eb;margin-top:3px;">'
                f'<div style="width:{pct}%;height:100%;border-radius:2px;background:{fg};"></div></div>')

    def score_color(s):
        if s is None: return "#888", "#f3f4f6"
        s = float(s)
        if s >= 75: return "#065f46", "#d1fae5"
        if s >= 55: return "#1e40af", "#dbeafe"
        if s >= 35: return "#92400e", "#fef3c7"
        return "#991b1b", "#fee2e2"

    def score_ring(s):
        """SVG circular progress ring for the score."""
        if s is None: return "—"
        s = float(s)
        r = 18; circ = 2 * 3.14159 * r
        dash = circ * s / 100
        fg, bg = score_color(s)
        return (f'<svg width="46" height="46" style="display:block;margin:0 auto 2px;">'
                f'<circle cx="23" cy="23" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="4"/>'
                f'<circle cx="23" cy="23" r="{r}" fill="none" stroke="{fg}" stroke-width="4" '
                f'stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round" '
                f'transform="rotate(-90 23 23)"/>'
                f'<text x="23" y="27" text-anchor="middle" font-size="10" font-weight="700" fill="{fg}">{s:.1f}</text>'
                f'</svg>')

    # header columns — Score first, then metrics
    cols_html = (f'<th class="sortable score-col" data-col="SCORE" title="Weighted average quality score (0-100)">'
                 f'Score<span class="unit">/100</span>'
                 f'<span class="better">↑</span>'
                 f'<span class="sort-icon">⇅</span></th>\n')
    for k in METRIC_KEYS:
        info = METRIC_INFO[k]
        tip = info["tip"]
        cols_html += (f'<th class="sortable" data-col="{k}" title="{tip}">'
                      f'{info["label"]}'
                      f'<span class="unit">{info["unit"]}</span>'
                      f'<span class="better">{info["better"]}</span>'
                      f'<span class="sort-icon">⇅</span></th>\n')

    # data rows
    data_rows_html = ""
    for i, r in enumerate(ok_rows):
        score = r.get("SCORE")
        fg, bg = score_color(score)
        cells = (f'<td class="score-cell" data-val="{score if score is not None else -1}" '
                 f'style="background:{bg};">'
                 f'{score_ring(score)}'
                 f'</td>')
        for k in METRIC_KEYS:
            v = r.get(k)
            cells += f'<td {cell_style(k,v)} data-val="{v if v is not None else -9999}">{fmt(k,v)}{spark_bar(k,v,summary[k])}</td>'
        data_rows_html += f"""
        <tr>
          <td class="idx-cell">{i+1}</td>
          <td class="img-cell">
            <img src="data:image/jpeg;base64,{r['hr_thumb']}" title="{r['file']}" class="thumb thumb-hr">
            <div class="fname">{r['file']}</div>
            <div class="sizes">{r['hr_size']}</div>
          </td>
          {cells}
        </tr>"""

    # summary rows
    def sum_row(stat_key, label):
        # Score summary
        score_vals = [r.get("SCORE") for r in ok_rows if r.get("SCORE") is not None]
        if stat_key == "mean":   sv = round(float(np.mean(score_vals)),2) if score_vals else None
        elif stat_key == "std":  sv = round(float(np.std(score_vals)),2)  if score_vals else None
        elif stat_key == "min":  sv = round(float(np.min(score_vals)),2)  if score_vals else None
        elif stat_key == "max":  sv = round(float(np.max(score_vals)),2)  if score_vals else None
        else: sv = None
        cells = f'<td class="sum-label" colspan="2">{label}</td>'
        cells += f'<td class="sum-cell" style="font-weight:700;">{sv if sv is not None else "—"}</td>'
        for k in METRIC_KEYS:
            v = summary[k].get(stat_key)
            disp = f"{v:.4f}" if v is not None else "—"
            cells += f'<td class="sum-cell">{disp}</td>'
        return f'<tr class="sum-row">{cells}</tr>'

    sum_html  = sum_row("mean","Mean")
    sum_html += sum_row("std","Std Dev")
    sum_html += sum_row("min","Min")
    sum_html += sum_row("max","Max")

    # error rows
    err_html = ""
    if fail_rows:
        err_html = "<h3 style='color:#b91c1c;margin-top:24px;'>Failed images</h3><ul>"
        for r in fail_rows:
            err_html += f"<li><code>{r['file']}</code>: {r['error']}</li>"
        err_html += "</ul>"

    # legend
    legend_html = ""
    for label, (fg, bg) in [("Excellent",COLOR_CSS["green"]),("Good",COLOR_CSS["blue"]),
                              ("Fair",COLOR_CSS["amber"]),("Poor",COLOR_CSS["red"])]:
        legend_html += (f'<span class="badge" style="background:{bg};color:{fg};">'
                        f'{label}</span>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Batch Image Quality Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    background: #f8f9fa;
    color: #1a1a1a;
  }}
  .page {{ max-width: 1600px; margin: 0 auto; padding: 24px 20px; }}

  /* Header */
  .header {{ margin-bottom: 20px; }}
  .header h1 {{ font-size: 22px; font-weight: 700; color: #111; margin-bottom: 4px; }}
  .header p  {{ color: #555; font-size: 13px; }}
  .meta {{ display:flex; gap:24px; margin-top:10px; flex-wrap:wrap; }}
  .meta-item {{ background:#fff; border:1px solid #e5e7eb; border-radius:8px;
                padding:10px 18px; }}
  .meta-item .val {{ font-size:20px; font-weight:700; color:#1a1a1a; }}
  .meta-item .lbl {{ font-size:11px; color:#888; margin-top:2px; }}

  /* LR reference box */
  .lr-ref-box {{
    display:flex; flex-direction:column; gap:8px;
    background:#fff; border:2px solid #6366f1; border-radius:10px;
    padding:14px 18px; margin-bottom:20px;
  }}
  .lr-ref-label {{
    font-size:10px; font-weight:700; letter-spacing:.6px;
    text-transform:uppercase; color:#6366f1;
  }}
  .lr-ref-inner {{ display:flex; gap:16px; align-items:flex-start; }}
  .lr-ref-img {{
    width:{tw*2}px; height:{th*2}px; object-fit:cover;
    border-radius:6px; border:1px solid #e5e7eb; cursor:pointer;
    filter:grayscale(100%);
  }}
  .lr-ref-img:hover {{ opacity:.85; }}
  .lr-ref-name {{ font-size:13px; font-weight:600; color:#111; }}
  .lr-ref-size {{ font-size:11px; color:#888; margin-top:3px; }}
  .lr-ref-note {{ font-size:11px; color:#6366f1; margin-top:6px; font-style:italic; }}

  /* Legend */
  .legend {{ display:flex; gap:8px; align-items:center; margin-bottom:16px; flex-wrap:wrap; }}
  .legend-label {{ font-size:11px; color:#888; margin-right:4px; }}
  .badge {{ padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; }}

  /* Controls */
  .controls {{ display:flex; gap:12px; margin-bottom:14px; align-items:center; flex-wrap:wrap; }}
  .controls input {{
    border:1px solid #d1d5db; border-radius:6px; padding:6px 12px;
    font-size:13px; width:220px; outline:none;
  }}
  .controls input:focus {{ border-color:#6366f1; }}
  .btn {{
    padding:6px 14px; border-radius:6px; border:1px solid #d1d5db;
    background:#fff; cursor:pointer; font-size:12px; color:#374151;
  }}
  .btn:hover {{ background:#f3f4f6; }}
  .btn.active {{ background:#6366f1; color:#fff; border-color:#6366f1; }}
  select {{
    border:1px solid #d1d5db; border-radius:6px; padding:6px 10px;
    font-size:12px; background:#fff; cursor:pointer;
  }}

  /* Table wrapper */
  .table-wrap {{ overflow-x:auto; border-radius:10px; border:1px solid #e5e7eb;
                 box-shadow:0 1px 4px rgba(0,0,0,.06); }}
  table {{
    border-collapse: collapse;
    width: 100%;
    background: #fff;
  }}
  thead th {{
    background: #f1f5f9;
    border-bottom: 2px solid #e2e8f0;
    padding: 10px 10px;
    text-align: center;
    font-size: 12px;
    font-weight: 600;
    color: #374151;
    white-space: nowrap;
    position: sticky;
    top: 0;
    z-index: 2;
    user-select: none;
  }}
  thead th.sortable {{ cursor:pointer; }}
  thead th.sortable:hover {{ background:#e2e8f0; }}
  thead th.sort-asc .sort-icon::after  {{ content:" ▲"; }}
  thead th.sort-desc .sort-icon::after {{ content:" ▼"; }}
  .sort-icon {{ color:#9ca3af; font-size:10px; }}
  .unit   {{ font-size:10px; color:#9ca3af; margin-left:2px; }}
  .better {{ font-size:10px; color:#6366f1; margin-left:3px; font-weight:700; }}
  tbody tr {{ border-bottom:1px solid #f1f5f9; transition:background .12s; }}
  tbody tr:hover {{ background:#f8fafc; }}
  tbody tr.hidden {{ display:none; }}
  td {{ padding:8px 10px; text-align:center; font-size:12px; vertical-align:middle; }}
  .idx-cell {{ color:#9ca3af; font-size:11px; width:32px; }}
  .img-cell {{ text-align:left; min-width:{tw+16}px; max-width:{tw+30}px; padding:6px 8px; }}
  .thumb    {{ width:{tw}px; height:{th}px; object-fit:cover; border-radius:4px; display:block;
               border:2px solid #6366f1; cursor:pointer; margin-bottom:4px; }}
  .thumb:hover {{ opacity:.85; }}
  .fname  {{ font-size:10px; font-weight:600; color:#1a1a1a; word-break:break-all; }}
  .sizes  {{ font-size:9px; color:#9ca3af; margin-top:1px; }}

  /* Summary rows */
  .sum-row td {{ background:#f8fafc; font-size:11px; color:#374151;
                  border-top:2px solid #e2e8f0; }}
  .sum-label {{ text-align:left; font-weight:700; padding-left:16px; color:#6366f1; }}
  .sum-cell  {{ font-weight:600; }}

  /* Metric group headers */
  .group-header th {{
    background: #ede9fe; color: #4c1d95; font-size: 11px; padding: 5px;
    text-align:center; letter-spacing:.5px; text-transform:uppercase;
  }}
  .group-header th.color-group {{ background:#d1fae5; color:#065f46; }}
  .group-header th.perceptual-group {{ background:#ffe4e6; color:#9f1239; }}

  /* Image modal */
  .modal-overlay {{
    display:none; position:fixed; inset:0; background:rgba(0,0,0,.75);
    z-index:100; align-items:center; justify-content:center;
  }}
  .modal-overlay.open {{ display:flex; }}
  .modal-box {{
    background:#fff; border-radius:12px; padding:20px; max-width:90vw;
    max-height:90vh; overflow:auto; position:relative;
  }}
  .modal-box img {{ max-width:80vw; max-height:75vh; display:block; border-radius:6px; }}
  .modal-close {{
    position:absolute; top:10px; right:14px; font-size:22px; cursor:pointer; color:#555;
    background:none; border:none; line-height:1;
  }}
  .modal-caption {{ font-size:12px; color:#555; margin-top:8px; text-align:center; }}

  /* Error section */
  .err-section {{ margin-top:24px; padding:16px; background:#fff5f5;
                  border:1px solid #fed7d7; border-radius:8px; }}
  .err-section h3 {{ color:#c53030; margin-bottom:8px; }}
  .err-section li {{ font-size:12px; color:#744210; margin-bottom:4px; }}

  .score-col {{ background:#1e1b4b !important; color:#c7d2fe !important; }}
  .score-cell {{ text-align:center; vertical-align:middle; min-width:58px; }}
  tr.highlight-best  {{ outline:2px solid #10b981; outline-offset:-1px; }}
  tr.highlight-worst {{ outline:2px solid #ef4444; outline-offset:-1px; }}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <h1>Batch Image Quality Report</h1>
    <p>Single LR Grayscale → Multiple HR Outputs &nbsp;·&nbsp; Diffusion Model Comparison</p>
    <div class="meta">
      <div class="meta-item"><div class="val">{len(ok_rows)}</div><div class="lbl">HR outputs evaluated</div></div>
      <div class="meta-item"><div class="val">{len(fail_rows)}</div><div class="lbl">Failed</div></div>
      <div class="meta-item"><div class="val">{elapsed:.1f}s</div><div class="lbl">Total time</div></div>
      <div class="meta-item">
        <div class="val">{summary['PSNR']['mean'] if summary['PSNR']['mean'] else '—'} dB</div>
        <div class="lbl">Mean PSNR</div>
      </div>
      <div class="meta-item">
        <div class="val">{summary['SSIM']['mean'] if summary['SSIM']['mean'] else '—'}</div>
        <div class="lbl">Mean SSIM</div>
      </div>
      <div class="meta-item">
        <div class="val">{summary['CFI']['mean'] if summary['CFI']['mean'] else '—'}</div>
        <div class="lbl">Mean Colorfulness (CFI)</div>
      </div>
    </div>
  </div>

  {lr_ref_html}

  <div class="legend">
    <span class="legend-label">Rating:</span>
    {legend_html}
    &nbsp;·&nbsp;
    <span class="legend-label">Hover column header for description &nbsp;·&nbsp; Click header to sort &nbsp;·&nbsp; Click thumbnail to enlarge</span>
  </div>

  <div class="controls">
    <input type="text" id="search" placeholder="Filter by filename…" oninput="filterRows()">
    <select id="sort-metric" onchange="sortBySelect()">
      <option value="">Sort by metric…</option>
      {''.join(f'<option value="{k}">{METRIC_INFO[k]["label"]}</option>' for k in METRIC_KEYS)}
    </select>
    <button class="btn" id="btn-best"  onclick="highlightBest()">Highlight best</button>
    <button class="btn" id="btn-worst" onclick="highlightWorst()">Highlight worst</button>
    <button class="btn" onclick="clearHighlight()">Clear</button>
    <button class="btn" onclick="exportCSV()">Export CSV</button>
  </div>

  <div class="table-wrap">
  <table id="main-table">
    <thead>
      <tr>
        <th style="width:32px">#</th>
        <th style="text-align:left;min-width:{tw+16}px">HR output</th>
        {cols_html}
      </tr>
      <tr class="group-header">
        <th colspan="2" style="background:#f1f5f9;"></th>
        <th style="background:#1e1b4b;color:#c7d2fe;">Weighted</th>
        <th colspan="4">Detail &amp; Structural Fidelity</th>
        <th colspan="4" class="color-group">Color Quality</th>
        <th class="perceptual-group">Perceptual</th>
      </tr>
    </thead>
    <tbody id="table-body">
      {data_rows_html}
    </tbody>
    <tfoot>
      {sum_html}
    </tfoot>
  </table>
  </div>

  {('<div class="err-section"><h3>Failed images</h3><ul>' +
     ''.join(f'<li><code>{r["file"]}</code>: {r["error"]}</li>' for r in fail_rows) +
     '</ul></div>') if fail_rows else ''}

</div>

<!-- Modal for enlarged thumbnails -->
<div class="modal-overlay" id="modal" onclick="closeModal(event)">
  <div class="modal-box">
    <button class="modal-close" onclick="document.getElementById('modal').classList.remove('open')">✕</button>
    <img id="modal-img" src="" alt="">
    <p class="modal-caption" id="modal-caption"></p>
  </div>
</div>

<script>
const DATA = {json.dumps([
    {k: r.get(k) for k in ["file","SCORE","PSNR","SSIM","GMSD","EPS","CFI","MSAT","HUE_ENT","CNI","BRISQUE"]}
    for r in ok_rows
])};
const BETTER_LOWER = {{"GMSD":1,"BRISQUE":1}};
const COL_IDX = {json.dumps({"SCORE": 2, **{k: i+3 for i,k in enumerate(METRIC_KEYS)}})};

function filterRows() {{
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('#table-body tr').forEach(tr => {{
    const name = tr.querySelector('.fname')?.textContent?.toLowerCase() || '';
    tr.classList.toggle('hidden', q && !name.includes(q));
  }});
}}

let sortState = {{}};
function sortTable(col) {{
  const tbody = document.getElementById('table-body');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  const asc   = sortState[col] !== 'asc';
  sortState    = {{[col]: asc ? 'asc' : 'desc'}};
  document.querySelectorAll('thead th.sortable').forEach(th => {{
    th.classList.remove('sort-asc','sort-desc');
    if (th.dataset.col === col) th.classList.add(asc ? 'sort-asc' : 'sort-desc');
  }});
  rows.sort((a, b) => {{
    const ci = COL_IDX[col];
    const fa = parseFloat(a.querySelectorAll('td')[ci]?.dataset?.val ?? -9999);
    const fb = parseFloat(b.querySelectorAll('td')[ci]?.dataset?.val ?? -9999);
    return asc ? fa - fb : fb - fa;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
function sortBySelect() {{
  const v = document.getElementById('sort-metric').value;
  if (v) sortTable(v);
}}
document.querySelectorAll('thead th.sortable').forEach(th => {{
  th.addEventListener('click', () => sortTable(th.dataset.col));
}});
// Default sort: Score descending (already sorted server-side, just mark the header)
document.querySelector('thead th[data-col="SCORE"]')?.classList.add('sort-desc');
sortState = {{"SCORE":"desc"}};

function highlightBest() {{
  clearHighlight();
  const col = document.getElementById('sort-metric').value || 'SCORE';
  const lowerBetter = !!BETTER_LOWER[col];
  const rows = Array.from(document.querySelectorAll('#table-body tr:not(.hidden)'));
  let best = null, bestVal = lowerBetter ? Infinity : -Infinity;
  rows.forEach(r => {{
    const ci = COL_IDX[col];
    const v  = parseFloat(r.querySelectorAll('td')[ci]?.dataset?.val ?? -9999);
    if (lowerBetter ? v < bestVal : v > bestVal) {{ bestVal = v; best = r; }}
  }});
  best?.classList.add('highlight-best');
  document.getElementById('btn-best').classList.add('active');
}}
function highlightWorst() {{
  clearHighlight();
  const col = document.getElementById('sort-metric').value || 'SCORE';
  const lowerBetter = !!BETTER_LOWER[col];
  const rows = Array.from(document.querySelectorAll('#table-body tr:not(.hidden)'));
  let worst = null, worstVal = lowerBetter ? -Infinity : Infinity;
  rows.forEach(r => {{
    const ci = COL_IDX[col];
    const v  = parseFloat(r.querySelectorAll('td')[ci]?.dataset?.val ?? -9999);
    if (lowerBetter ? v > worstVal : v < worstVal) {{ worstVal = v; worst = r; }}
  }});
  worst?.classList.add('highlight-worst');
  document.getElementById('btn-worst').classList.add('active');
}}
function clearHighlight() {{
  document.querySelectorAll('.highlight-best,.highlight-worst').forEach(r => {{
    r.classList.remove('highlight-best','highlight-worst');
  }});
  document.querySelectorAll('.btn.active').forEach(b => b.classList.remove('active'));
}}

function openModal(src, caption) {{
  document.getElementById('modal-img').src = src;
  document.getElementById('modal-caption').textContent = caption;
  document.getElementById('modal').classList.add('open');
}}
function closeModal(e) {{
  if (e.target.id === 'modal') document.getElementById('modal').classList.remove('open');
}}
document.querySelectorAll('.thumb').forEach(img => {{
  img.addEventListener('click', () => openModal(img.src, img.title));
}});

function exportCSV() {{
  const headers = ['file','SCORE','PSNR','SSIM','GMSD','EPS','CFI','MSAT','HUE_ENT','CNI','BRISQUE'];
  const lines   = [headers.join(',')];
  DATA.forEach(r => lines.push(headers.map(h => r[h] ?? '').join(',')));
  const blob = new Blob([lines.join('\\n')], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'quality_metrics.csv';
  a.click();
}}
</script>
</body>
</html>"""
    return html

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all HR outputs in a folder against a single LR grayscale reference."
    )
    parser.add_argument("--lr",        required=True, help="Single LR grayscale image path")
    parser.add_argument("--hr_dir",    required=True, help="Folder containing HR colorized outputs")
    parser.add_argument("--report",    default="quality_report.html", help="Output HTML report path")
    parser.add_argument("--workers",   type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--thumb_size",default="120x80", help="Thumbnail WxH (default: 120x80)")
    parser.add_argument("--no_brisque",action="store_true", help="Skip BRISQUE computation")
    args = parser.parse_args()

    tw, th      = map(int, args.thumb_size.lower().split("x"))
    use_brisque = not args.no_brisque
    lr_path     = Path(args.lr)
    hr_dir      = Path(args.hr_dir)

    if not lr_path.exists():  sys.exit(f"LR image not found: {lr_path}")
    if not lr_path.is_file(): sys.exit(f"--lr must be a file, not a directory: {lr_path}")
    if not hr_dir.exists():   sys.exit(f"HR folder not found: {hr_dir}")

    hr_files = sorted(p for p in hr_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    print(f"\n{'='*60}")
    print(f"  Batch Image Quality Evaluator  (single LR → many HR)")
    print(f"{'='*60}")
    print(f"  LR reference : {lr_path}")
    print(f"  HR folder    : {hr_dir}  ({len(hr_files)} images)")
    print(f"  Workers      : {args.workers}")
    print(f"  BRISQUE      : {'yes' if use_brisque else 'no'}")
    print(f"{'='*60}\n")

    if not hr_files:
        sys.exit("No images found in HR folder.")

    t0      = time.time()
    results = []
    tasks   = [(lr_path, hr_p, tw, th, use_brisque) for hr_p in hr_files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_pair, *t): t[1].name for t in tasks}
        for fut in tqdm(concurrent.futures.as_completed(futures),
                        total=len(futures), desc="Evaluating", unit="img"):
            results.append(fut.result())

    # Sort by weighted score descending (best first)
    results.sort(key=lambda r: r.get("SCORE") or 0, reverse=True)

    elapsed = time.time() - t0
    ok_rows = [r for r in results if not r.get("error")]
    summary = compute_summary(ok_rows)

    # ── Terminal table ─────────────────────────────────────────────
    from tabulate import tabulate as tb
    compact = [[r["file"], f"{r.get('SCORE','—')}/100"] + [fmt(k, r.get(k)) for k in METRIC_KEYS] for r in ok_rows]
    print(tb(compact, headers=["File","Score"] + list(METRIC_KEYS),
             tablefmt="rounded_outline", maxcolwidths=[28] + [10]*10))

    print(f"\n  Evaluated {len(ok_rows)} HR images in {elapsed:.1f}s")
    print(f"  Mean PSNR : {summary['PSNR']['mean']} dB")
    print(f"  Mean SSIM : {summary['SSIM']['mean']}")
    print(f"  Mean CFI  : {summary['CFI']['mean']}")

    # ── HTML report ────────────────────────────────────────────────
    html = generate_html(results, summary, tw, th, elapsed, lr_path)
    out  = Path(args.report)
    out.write_text(html, encoding="utf-8")
    print(f"\n  HTML report → {out.resolve()}")
    print(f"  Open in browser: file://{out.resolve()}\n")

if __name__ == "__main__":
    main()
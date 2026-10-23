"""
Keyframe chroma propagation — with global homography alignment.

Middle ground between the two earlier versions:
    * ECC homography at coarse res handles GLOBAL misalignment (whole-frame
      shift/tilt between keyframe and current frame).
    * No dense optical flow — per-object motion inside the scene is handled
      by the confidence check (fade toward neutral), not by warping.

Pipeline:
    1. Downscale both lumas and the keyframe chroma to coarse resolution.
    2. Estimate a homography (cur -> key) with ECC on the coarse lumas.
    3. Warp keyframe chroma AND keyframe luma with that single homography.
    4. Confidence = warped-key-luma vs current-luma residual.
    5. Fade low-confidence chroma toward neutral.
    6. Guided-filter upsample chroma to full res, guided by current luma.
    7. Composite: current luma (untouched) + chroma -> RGB.

Requires: numpy, opencv-python (opencv-contrib for cv2.ximgproc; a pure-NumPy
guided-filter fallback is included).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Guided filter (ximgproc if present, NumPy fallback otherwise)
# --------------------------------------------------------------------------- #

def _box(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.blur(img, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int,
                   eps: float) -> np.ndarray:
    """He et al. guided filter, single-channel float32 guide/src in [0,1]."""
    # plain opencv-python exposes a cv2.ximgproc namespace without guidedFilter
    # in it, so probe for the function itself rather than the module
    _gf = getattr(getattr(cv2, "ximgproc", None), "guidedFilter", None)
    if _gf is not None:
        return _gf(guide, src, radius, eps)
    mean_g = _box(guide, radius)
    mean_s = _box(src, radius)
    corr_gs = _box(guide * src, radius)
    corr_gg = _box(guide * guide, radius)
    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return _box(a, radius) * guide + _box(b, radius)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class ChromaResult:
    rgb: np.ndarray            # (H, W, 3) uint8 — the PASD conditioning image
    confidence: np.ndarray     # (H, W) float32 in [0,1], full res
    mean_luma_residual: float  # drift metric; trigger a new keyframe when high
    homography: np.ndarray     # (3, 3) float32, cur -> key at coarse scale


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def propagate_keyframe_chroma(
    key_rgb: np.ndarray,
    cur_gray: np.ndarray,
    chroma_downscale: int = 4,
    use_homography: bool = True,
    residual_soft_thresh: float = 12.0,
    residual_hard_thresh: float = 35.0,
    guided_radius: int = 8,
    guided_eps: float = 1e-3,
    desat_target: float = 0.35,
) -> ChromaResult:
    """Apply a color keyframe's chroma to the current grey frame.

    Global camera motion between the frames is absorbed by an ECC-estimated
    homography; residual per-object motion is confidence-gated. See module
    docstring for the pipeline. Tune the two residual thresholds (0-255 luma
    units) on real footage from your sensor.
    """
    # ---- normalize inputs ------------------------------------------------- #
    if cur_gray.ndim == 3:
        cur_gray = cur_gray[..., 0]
    if key_rgb.shape[:2] != cur_gray.shape[:2]:
        raise ValueError(
            f"size mismatch: key {key_rgb.shape[:2]} vs cur {cur_gray.shape[:2]}"
        )
    h, w = cur_gray.shape
    s = chroma_downscale
    lw, lh = max(w // s, 16), max(h // s, 16)

    key_ycc = cv2.cvtColor(key_rgb, cv2.COLOR_RGB2YCrCb)

    # ---- 1. coarse working copies ------------------------------------------ #
    key_y_lo = cv2.resize(key_ycc[..., 0], (lw, lh),
                          interpolation=cv2.INTER_AREA)
    cur_y_lo = cv2.resize(cur_gray, (lw, lh), interpolation=cv2.INTER_AREA)
    chroma_lo = cv2.resize(key_ycc[..., 1:], (lw, lh),
                           interpolation=cv2.INTER_AREA).astype(np.float32)

    # ---- 2. global homography (cur -> key) via ECC on coarse lumas ---------- #
    warp_h = np.eye(3, dtype=np.float32)
    if use_homography:
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
            _, warp_h = cv2.findTransformECC(
                cur_y_lo, key_y_lo, warp_h, cv2.MOTION_HOMOGRAPHY, crit,
                None, 3,
            )
        except cv2.error:
            warp_h = np.eye(3, dtype=np.float32)  # no convergence -> identity

    # ---- 3. single direct warp of chroma + key luma -------------------------- #
    warped_chroma_lo = cv2.warpPerspective(
        chroma_lo, warp_h, (lw, lh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_key_y_lo = cv2.warpPerspective(
        key_y_lo, warp_h, (lw, lh),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)

    # ---- 4. confidence from warped-luma residual ------------------------------ #
    resid = cv2.GaussianBlur(
        np.abs(warped_key_y_lo - cur_y_lo.astype(np.float32)), (5, 5), 0
    )
    conf_lo = np.clip(
        (residual_hard_thresh - resid)
        / max(residual_hard_thresh - residual_soft_thresh, 1e-6),
        0.0, 1.0,
    ).astype(np.float32)
    mean_residual = float(resid.mean())

    # ---- 5. fade rejected regions toward neutral (128 in CrCb) ---------------- #
    pull = (1.0 - conf_lo) * desat_target
    warped_chroma_lo = warped_chroma_lo + (128.0 - warped_chroma_lo) * pull[..., None]

    # ---- 6. edge-aware upsample guided by CURRENT luma ------------------------- #
    guide = cur_gray.astype(np.float32) / 255.0
    chroma_up = cv2.resize(warped_chroma_lo, (w, h),
                           interpolation=cv2.INTER_LINEAR)
    out_chroma = np.empty_like(chroma_up)
    for c in range(2):
        src = chroma_up[..., c].astype(np.float32) / 255.0
        out_chroma[..., c] = _guided_filter(guide, src, guided_radius,
                                            guided_eps) * 255.0

    conf_full = cv2.resize(conf_lo, (w, h), interpolation=cv2.INTER_LINEAR)

    # ---- 7. composite: current luma + propagated chroma ------------------------ #
    ycc = np.empty((h, w, 3), dtype=np.uint8)
    ycc[..., 0] = cur_gray                                # luma is never touched
    ycc[..., 1:] = np.clip(out_chroma, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

    return ChromaResult(rgb=rgb, confidence=conf_full,
                        mean_luma_residual=mean_residual, homography=warp_h)


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #

KEY_PATH = "C:/colorizationData/datasets/test_color/Strawberry/Strawberry19.jpg"
GREY_PATH = "C:/PASD-main/examples/Strawberry19.jpg"


def _load_or_synthesize():
    """Load the strawberry pair; fall back to synthetic images if missing."""
    key_bgr = cv2.imread(KEY_PATH, cv2.IMREAD_COLOR)
    cur_grey = cv2.imread(GREY_PATH, cv2.IMREAD_GRAYSCALE)
    if key_bgr is not None and cur_grey is not None:
        if key_bgr.shape[:2] != cur_grey.shape[:2]:
            key_bgr = cv2.resize(key_bgr,
                                 (cur_grey.shape[1], cur_grey.shape[0]))
        return key_bgr, cur_grey, True

    print("[warn] image files not found - using synthetic test images")
    rng = np.random.default_rng(0)
    H, W = 225, 225
    key = np.zeros((H, W, 3), np.uint8)
    key[..., 0], key[..., 1], key[..., 2] = 40, 40, 190       # bright bg (BGR)
    key[60:170, 60:170] = (30, 30, 160)                        # dark red square
    key = cv2.add(key, rng.integers(0, 15, key.shape, dtype=np.uint8))
    cur_grey = cv2.cvtColor(key, cv2.COLOR_BGR2GRAY)
    return key, cur_grey, False


def _show(name: str, img: np.ndarray, headless: bool):
    if headless:
        cv2.imwrite(f"out_{name.replace(' ', '_')}.png", img)
    else:
        cv2.imshow(name, img)


if __name__ == "__main__":
    key_bgr, cur_grey, from_files = _load_or_synthesize()
    key_rgb = cv2.cvtColor(key_bgr, cv2.COLOR_BGR2RGB)
    import os
    headless = os.name != "nt" and not os.environ.get("DISPLAY")

    # ---- test 1: aligned pair ---------------------------------------------- #
    res = propagate_keyframe_chroma(key_rgb, cur_grey)
    out_bgr = cv2.cvtColor(res.rgb, cv2.COLOR_RGB2BGR)
    print("[aligned]  residual: %.2f   min conf: %.2f"
          % (res.mean_luma_residual, res.confidence.min()))

    _show("input grey", cur_grey, headless)
    _show("input color", key_bgr, headless)
    _show("output", out_bgr, headless)
    _show("confidence", (res.confidence * 255).astype(np.uint8), headless)
    if not headless:
        cv2.waitKey(0)

    # ---- test 2: whole frame shifted down 15 px (global camera motion) ------ #
    # warpAffine, not slice-paste: one clean transform, edge-replicated border
    M = np.float32([[1, 0, 0], [0, 1, 15]])
    shifted = cv2.warpAffine(cur_grey, M,
                             (cur_grey.shape[1], cur_grey.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)

    res_s = propagate_keyframe_chroma(key_rgb, shifted)          # homography ON
    res_n = propagate_keyframe_chroma(key_rgb, shifted,
                                      use_homography=False)      # homography OFF
    print("[shifted]  with homography:    residual %.2f   mean conf %.2f"
          % (res_s.mean_luma_residual, res_s.confidence.mean()))
    print("[shifted]  without homography: residual %.2f   mean conf %.2f"
          % (res_n.mean_luma_residual, res_n.confidence.mean()))
    print("estimated homography (coarse px):\n", np.round(res_s.homography, 2))

    _show("input grey_shift", shifted, headless)
    _show("output_shift_homog", cv2.cvtColor(res_s.rgb, cv2.COLOR_RGB2BGR),
          headless)
    _show("output_shift_nohomog", cv2.cvtColor(res_n.rgb, cv2.COLOR_RGB2BGR),
          headless)
    if not headless:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("(headless build: wrote out_*.png instead of showing windows)")
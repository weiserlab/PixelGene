"""
test_pasd_sequence.py

Fixed-camera sequence pipeline: K color keyframes + N grey frames
-> N+K colored, realisr-ed images.

With several color keyframes, each grey frame takes its chroma from the latest
keyframe captured BEFORE it (the scene moves, so a later keyframe is not used).
A grey frame with no earlier keyframe falls back to the nearest one after it.

Chroma is carried along a dense optical flow by default (--chroma_flow 1), so
colour follows moving objects; --chroma_flow 0 restores the single-homography
propagation. With --chroma_chain 1 (default) consecutive grey frames hand the
colour on frame to frame -- each one is propagated from the previous grey
frame's colorized output rather than from the (older) keyframe, which keeps the
motion between reference and target small. The chain restarts at every color
keyframe. Snapshot of the homography-only version: test_pasd_sequence_HomoChroma.py

Per grey frame:
    1. CHROMA PROPAGATION -- chroma from the color keyframe is warped onto the
       grey frame (chroma_fixed_cam.propagate_keyframe_chroma, ECC-homography
       aligned, confidence-gated). Its RGB output is the PASD conditioning
       image.
    2. COLORIZATION -- the PASD color model (control_type='grayscale') runs on
       that conditioning image; afterwards the grey frame's own luma is
       restored and the chroma comes from either the propagated keyframe or
       the model (--chroma_prop), optionally mixed per pixel by the ECC
       confidence so poorly aligned regions fall back to the model
       (--chroma_by_confidence).
    3. REALISR -- the PASD realisr model runs on the colorized frame.

Each color keyframe is realisr-ed once (not once per pair) and emitted as
<name>_key_final.png alongside the N grey-frame outputs.

Reused from the repo (imported, not copied):
    test_pasd_NVDIA : load_pasd_pipeline, load_high_level_net,
                      get_validation_prompt
    chroma_fixed_cam: propagate_keyframe_chroma
    pasd.myutils    : wavelet_color_fix
    BenchmarkPASD   : GPUTimer

The two stages run as two passes (colorize every frame, then realisr every
frame) rather than interleaved, so only one diffusion pipeline is resident at
a time. When both stages point at the same checkpoint -- the default -- the
pipeline is loaded once and only its scheduler is swapped between stages.

Usage:
    python test_pasd_sequence.py --image_path examples/sequence \
        --use_pasd_light --process_size 512 --num_inference_steps 50 \
        --mixed_precision fp16 --output_tag seq01
"""

import os
import re
import sys
import copy
import glob
import argparse
import time

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import UniPCMultistepScheduler

# --- reuse the existing PASD inference code -------------------------------- #
from test_pasd_NVDIA import (
    load_pasd_pipeline,
    load_high_level_net,
    get_validation_prompt,
)
from chroma_fixed_cam import propagate_keyframe_chroma, propagate_keyframe_chroma_flow
from pasd.myutils.wavelet_color_fix import wavelet_color_fix
from BenchmarkPASD import GPUTimer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
FRAME_RE = re.compile(r'^(?P<stem>.+)_(?P<idx>\d+)$')

DEFAULT_COLOR_MODEL = "runs/pasd_light_color/checkpoint-180000"
DEFAULT_REALISR_MODEL = "runs/pasd_light_color/checkpoint-180000"


# --------------------------------------------------------------------------- #
# Progress logging
# --------------------------------------------------------------------------- #
# PBS redirects stdout to pasd.o<jobid>, and Python block-buffers a redirected
# stdout -- so without an explicit flush nothing shows up until the job ends.
# Every line here is flushed immediately and mirrored into <output_dir>/run.log.

_LOG_FH = None
_T0 = time.perf_counter()

if hasattr(sys.stdout, 'reconfigure'):      # also unbuffers prints from imports
    sys.stdout.reconfigure(line_buffering=True)


def log(msg='', prefix=''):
    """Print one timestamped progress line, flushed, and tee it to run.log."""
    if msg:
        line = (f'[{time.strftime("%H:%M:%S")} +{time.perf_counter() - _T0:7.1f}s] '
                f'{prefix}{msg}')
    else:
        line = ''
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + '\n')
        _LOG_FH.flush()


def open_log(path):
    """Start mirroring log() output to `path`."""
    global _LOG_FH
    try:
        _LOG_FH = open(path, 'a', encoding='utf-8')
    except OSError as e:
        print(f'[warn] could not open log file {path}: {e}', flush=True)


def close_log():
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


def step_logger(label, total, every):
    """Build a pipeline callback that logs denoising progress with an ETA.

    The pipeline calls callback(step, timestep, latents) every `every` steps.
    Returns None when step logging is switched off (--log_every 0).
    """
    if every <= 0:
        return None
    t0 = time.perf_counter()

    def _cb(step, _timestep, _latents):
        done = step + 1
        elapsed = time.perf_counter() - t0
        eta = elapsed / done * (total - done) if done else 0.0
        log(f'{label} step {done:3d}/{total}  {elapsed:5.1f}s elapsed, '
            f'~{eta:5.1f}s remaining', prefix='      ')

    return _cb


def resolve_model_paths(args):
    """Settle which checkpoint each stage uses.

    Precedence: per-stage flag > --pasd_model_path (the shared flag the other
    test_pasd_*.py scripts take) > built-in default. --control_type is accepted
    for the same CLI compatibility but has no meaning here: this script always
    runs the grayscale colorization stage and then the realisr stage.
    """
    args.color_model_path = (args.color_model_path or args.pasd_model_path
                             or DEFAULT_COLOR_MODEL)
    args.realisr_model_path = (args.realisr_model_path or args.pasd_model_path
                               or DEFAULT_REALISR_MODEL)
    if args.control_type is not None:
        print(f'[note] --control_type {args.control_type} is ignored: this script runs '
              'grayscale colorization and then realisr on every frame')
    return args


# --------------------------------------------------------------------------- #
# Input discovery
# --------------------------------------------------------------------------- #

_SKIP_DIRS = {'.git', '__pycache__', 'runs', 'checkpoints', 'output', 'outputs',
              '.vscode', '.metadata', 'node_modules'}


def _suggest_folders(folder, limit=8, max_depth=4):
    """Find directories under cwd whose name matches the one asked for, so a
    wrong --image_path on a cluster reports where the images actually are."""
    target = os.path.basename(os.path.normpath(folder))
    if not target:
        return []
    root = os.getcwd()
    hits = []
    for dirpath, dirnames, _files in os.walk(root):
        if dirpath[len(root):].count(os.sep) >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames
                       if not d.startswith('.') and d not in _SKIP_DIRS]
        for d in dirnames:
            if d == target:
                hits.append(os.path.relpath(os.path.join(dirpath, d), root))
                if len(hits) >= limit:
                    return hits
    return hits


def _mean_saturation(path):
    """Mean |max(R,G,B) - min(R,G,B)| of an image, 0 for a true grey frame."""
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    return float((arr.max(axis=2) - arr.min(axis=2)).mean())


def _frame_order(p):
    """Sort key: trailing _<index> when the name has one, else the file name
    (capture names like 00002_gray_14-41-54 sort by their leading number)."""
    m = FRAME_RE.match(os.path.splitext(os.path.basename(p))[0])
    return (0, int(m.group('idx'))) if m else (1, os.path.basename(p))


def find_sequence(folder, color_image=None, sat_thresh=2.0):
    """Split `folder` into ([color keyframe paths], [grey frame paths]), both in
    frame order.

    Every image that actually carries chroma is a keyframe. Pass --color_image
    to force a single keyframe instead (every other image is then treated as
    grey).
    """
    if not os.path.isdir(folder):
        msg = f'--image_path is not a folder: {folder}  (cwd: {os.getcwd()})'
        hits = _suggest_folders(folder)
        if hits:
            msg += '\n  found these instead -- did you mean one of them?\n'
            msg += '\n'.join(f'    ./{h}' for h in hits)
        raise NotADirectoryError(msg)

    paths = sorted(p for p in glob.glob(os.path.join(folder, '*.*'))
                   if p.lower().endswith(IMAGE_EXTS))
    if not paths:
        raise FileNotFoundError(f'no images found in {folder}')

    if color_image is not None:
        key = color_image if os.path.isabs(color_image) or os.path.exists(color_image) \
            else os.path.join(folder, color_image)
        if not os.path.exists(key):
            raise FileNotFoundError(f'--color_image not found: {key}')
        key = os.path.normpath(key)
        keys = [key]
        greys = [p for p in paths if os.path.normpath(p) != key]
    else:
        sats = {p: _mean_saturation(p) for p in paths}
        keys = [p for p in paths if sats[p] > sat_thresh]
        if not keys:
            raise ValueError(
                f'no color image found in {folder} (all frames look grey; '
                'pass --color_image to name the keyframe explicitly)')
        greys = [p for p in paths if p not in keys]

    keys.sort(key=_frame_order)
    greys.sort(key=_frame_order)
    return keys, greys


def assign_keyframes(keys, greys):
    """Map each grey frame to the keyframe it takes its chroma from.

    The scene moves between frames, so a grey frame uses the latest color
    keyframe captured BEFORE it -- never a later one. Only a grey frame with no
    earlier keyframe (the sequence starts grey) falls back to the nearest
    keyframe after it.

    Returns (mapping grey -> keyframe, [grey frames that used the fallback]).
    """
    order = sorted(keys + greys, key=_frame_order)
    pos = {p: i for i, p in enumerate(order)}
    out, fallback = {}, []
    for g in greys:
        before = [k for k in keys if pos[k] < pos[g]]
        if before:
            out[g] = before[-1]
        else:
            out[g] = min((k for k in keys if pos[k] > pos[g]), key=lambda k: pos[k])
            fallback.append(g)
    return out, fallback


# --------------------------------------------------------------------------- #
# Per-stage args / helpers
# --------------------------------------------------------------------------- #

def stage_args(args, control_type, model_path, **overrides):
    """Shallow copy of `args` describing one stage.

    load_pasd_pipeline() and the pipeline call both read their settings off an
    args object, so each stage gets its own copy with the right control_type,
    checkpoint, step count and prompt rather than mutating a shared one.
    """
    s = copy.copy(args)
    s.control_type = control_type
    s.pasd_model_path = model_path
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_resize_preproc(stage):
    """Same resize policy as test_pasd_NVDIA.main, per control type."""
    return transforms.Compose([
        transforms.Resize(stage.process_size,
                          interpolation=transforms.InterpolationMode.BILINEAR),
    ] if stage.control_type == "realisr" else [
        transforms.Resize(stage.process_size, max_size=stage.process_size * 2,
                          interpolation=transforms.InterpolationMode.BILINEAR),
    ])


def snap8(image):
    """Round a PIL image down to a multiple of 8 in both dimensions."""
    return image.resize((image.size[0] // 8 * 8, image.size[1] // 8 * 8))


_CAPTION_PRIMARY_OK = True      # flips off after the first captioner failure


def safe_caption(stage, image, nets, device):
    """get_validation_prompt(), but never fatal.

    open_clip's CoCa defaults to beam search, which calls into transformers'
    BeamSearchScorer; its internals moved in newer transformers, so the call
    dies with "Batch dimension of `input_ids` should be 0, but is 6". We retry
    with plain sampling, and if even that fails we carry on with the bare
    prompt -- a missing caption is not worth losing a long run over.
    """
    global _CAPTION_PRIMARY_OK
    model, preprocess, category = nets
    fallback = stage.prompt or ''

    if _CAPTION_PRIMARY_OK:
        try:
            return get_validation_prompt(stage, image, model, preprocess, category, device)
        except Exception as e:
            _CAPTION_PRIMARY_OK = False
            log(f'[warn] captioner failed ({type(e).__name__}: {e})', prefix='    ')
            log(f'[warn] retrying captions with {stage.caption_generation} sampling '
                'for the rest of the run', prefix='    ')

    # fallback path: CoCa without beam search
    if stage.high_level_info == 'caption' and not stage.use_blip and model is not None:
        try:
            import open_clip
            with torch.no_grad():
                generated = model.generate(
                    preprocess(image).unsqueeze(0),
                    generation_type=stage.caption_generation)
            caption = open_clip.decode(generated[0]).split('<end_of_text>')[0] \
                .replace('<start_of_text>', '')
            caption = caption.replace('blurry', 'clear').replace('noisy', 'clean')
            return caption if stage.prompt == '' else f'{caption} {stage.prompt}'
        except Exception as e:
            log(f'[warn] {stage.caption_generation} retry also failed '
                f'({type(e).__name__}: {e}); continuing without a caption', prefix='    ')

    return fallback


def reseed(generator, args):
    """Re-seed before every diffusion call so a frame's result does not depend
    on how many frames were processed before it."""
    if args.seed is not None:
        generator.manual_seed(args.seed)
    return generator


# --------------------------------------------------------------------------- #
# Stage 1: chroma propagation + PASD colorization
# --------------------------------------------------------------------------- #

def colorize_frame(cargs, pipeline, nets, device, generator, grey_path, key_rgb,
                   resize_preproc, debug_dir=None, label=''):
    """Colorize one grey frame using the keyframe's chroma. Returns
    (PIL RGB image, ChromaResult, prompt used)."""
    model, preprocess, category = nets

    grey_pil = Image.open(grey_path).convert("L")
    cur_gray = np.asarray(grey_pil)
    # luma reference at native size; RGB so the YUV recomposition below matches
    # test_pasd_NVDIA (R=G=B, so the BGR/RGB convention there is a no-op)
    orig_img = grey_pil.convert("RGB")

    key = key_rgb
    if key.shape[:2] != cur_gray.shape[:2]:
        key = cv2.resize(key, (cur_gray.shape[1], cur_gray.shape[0]),
                         interpolation=cv2.INTER_AREA)

    # ---- 1. chroma from the keyframe -> conditioning image ---------------- #
    method = 'optical flow' if cargs.chroma_flow else 'homography'
    log(f'chroma propagation ({method}) on {cur_gray.shape[1]}x{cur_gray.shape[0]} grey frame'
        f'{" (reference resized to match)" if key is not key_rgb else ""}',
        prefix='    ')
    if cargs.chroma_flow:
        chroma = propagate_keyframe_chroma_flow(
            key, cur_gray,
            residual_soft_thresh=cargs.residual_soft_thresh,
            residual_hard_thresh=cargs.residual_hard_thresh,
            fb_soft_thresh=cargs.fb_soft_thresh,
            fb_hard_thresh=cargs.fb_hard_thresh,
            guided_radius=cargs.guided_radius,
            guided_eps=cargs.guided_eps,
            desat_target=cargs.desat_target,
        )
    else:
        chroma = propagate_keyframe_chroma(
            key, cur_gray,
            chroma_downscale=cargs.chroma_downscale,
            use_homography=not cargs.no_homography,
            residual_soft_thresh=cargs.residual_soft_thresh,
            residual_hard_thresh=cargs.residual_hard_thresh,
            guided_radius=cargs.guided_radius,
            guided_eps=cargs.guided_eps,
            desat_target=cargs.desat_target,
        )
    cond_image = Image.fromarray(chroma.rgb)
    log(f'chroma done: luma residual {chroma.mean_luma_residual:.2f}, '
        f'mean confidence {chroma.confidence.mean():.3f}', prefix='    ')

    if debug_dir is not None:
        name = os.path.splitext(os.path.basename(grey_path))[0]
        cond_image.save(os.path.join(debug_dir, f'{name}_conditioning.png'))
        cv2.imwrite(os.path.join(debug_dir, f'{name}_confidence.png'),
                    (chroma.confidence * 255).astype(np.uint8))
        log(f'wrote {name}_conditioning.png / {name}_confidence.png', prefix='    ')

    # ---- 2. PASD colorization on that conditioning image ------------------ #
    validation_prompt = safe_caption(cargs, cond_image, nets, device)
    validation_prompt = validation_prompt.replace("black and white", "color")
    negative_prompt = cargs.color_negative_prompt
    log(f'caption: "{validation_prompt.strip()}"', prefix='    ')

    validation_image = snap8(resize_preproc(cond_image))
    log(f'colorizing at {validation_image.size[0]}x{validation_image.size[1]}, '
        f'{cargs.num_inference_steps} steps, cond scale {cargs.conditioning_scale}',
        prefix='    ')

    image = pipeline(
        cargs, validation_prompt, validation_image,
        num_inference_steps=cargs.num_inference_steps, generator=reseed(generator, cargs),
        guidance_scale=cargs.guidance_scale, negative_prompt=negative_prompt,
        conditioning_scale=cargs.conditioning_scale,
        callback=step_logger(f'{label}colorize', cargs.num_inference_steps, cargs.log_every),
        callback_steps=max(cargs.log_every, 1),
    ).images[0]

    # ---- 3. recompose: grey frame's luma + chosen chroma ------------------ #
    # Luma always comes from the grey frame (as in the grayscale branch of
    # test_pasd_NVDIA.main). The chroma is a linear blend, in YUV, between the
    # propagated keyframe chroma and the model's own:
    #     --chroma_source 1.0 -> propagated only (faithful to the keyframe)
    #     --chroma_source 0.0 -> model only (the original behaviour)
    # Both chroma channels are offsets about 128, so interpolating them
    # linearly is well defined. ascontiguousarray only keeps a negative-stride
    # array from reaching cv2.
    np_image = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    color_np = cv2.resize(np_image, orig_img.size)
    orig_np = np.asarray(orig_img)
    cond_np = np.ascontiguousarray(np.asarray(cond_image)[:, :, ::-1])
    if cond_np.shape[:2] != color_np.shape[:2]:
        cond_np = cv2.resize(cond_np, (color_np.shape[1], color_np.shape[0]))

    color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
    cond_yuv = cv2.cvtColor(cond_np, cv2.COLOR_BGR2YUV)
    orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)

    w = 1.0 if cargs.chroma_prop else 0.0
    if cargs.chroma_by_confidence and w:
        # The confidence picks the source per pixel -- SWAPPED: where the
        # propagation is confident the MODEL's chroma is used, where it is not
        # the propagated (optical-flow) chroma is used.
        #     confidence 1 -> model chroma, confidence 0 -> propagated chroma
        # (wmap below is the weight of the propagated chroma.)
        conf = np.clip(chroma.confidence, 0.0, 1.0).astype(np.float32)
        if conf.shape[:2] != color_np.shape[:2]:
            conf = cv2.resize(conf, (color_np.shape[1], color_np.shape[0]))
        wmap = 1.0 - conf[..., None]
        detail = (f'confidence-weighted, swapped (mean propagated weight {wmap.mean():.2f}, '
                  f'{100 * (wmap < 0.5).mean():.1f}% of pixels mostly model)')
    else:
        wmap = w
        detail = ('all propagated' if w else 'all model') + ', no confidence weighting'

    chroma_mix = (wmap * cond_yuv[:, :, 1:3].astype(np.float32)
                  + (1.0 - wmap) * color_yuv[:, :, 1:3].astype(np.float32))

    hires = np.copy(orig_yuv)
    hires[:, :, 1:3] = np.clip(np.rint(chroma_mix), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)

    log(f'chroma {detail} (mean |model-propagated| = '
        f'{np.abs(color_yuv[:, :, 1:3].astype(np.float32) - cond_yuv[:, :, 1:3].astype(np.float32)).mean():.1f})',
        prefix='    ')

    return Image.fromarray(bgr[:, :, ::-1]), chroma, validation_prompt


# --------------------------------------------------------------------------- #
# Stage 2: realisr
# --------------------------------------------------------------------------- #

def realisr_frame(rargs, pipeline, nets, device, generator, image, resize_preproc,
                  label=''):
    """Run realisr on a PIL image. Mirrors the realisr branch of
    test_pasd_NVDIA.main."""
    model, preprocess, category = nets

    validation_image = image.convert("RGB")
    validation_prompt = safe_caption(rargs, validation_image, nets, device)
    validation_prompt += rargs.added_prompt
    negative_prompt = rargs.negative_prompt
    log(f'caption: "{validation_prompt.strip()}"', prefix='    ')

    ori_width, ori_height = validation_image.size
    rscale = rargs.upscale

    validation_image = validation_image.resize(
        (validation_image.size[0] * rscale, validation_image.size[1] * rscale))
    if min(validation_image.size) < rargs.process_size:
        validation_image = resize_preproc(validation_image)
    validation_image = snap8(validation_image)
    log(f'realisr at {validation_image.size[0]}x{validation_image.size[1]} '
        f'(from {ori_width}x{ori_height}, upscale {rscale}), '
        f'{rargs.num_inference_steps} steps, cond scale {rargs.conditioning_scale}',
        prefix='    ')

    out = pipeline(
        rargs, validation_prompt, validation_image,
        num_inference_steps=rargs.num_inference_steps, generator=reseed(generator, rargs),
        guidance_scale=rargs.guidance_scale, negative_prompt=negative_prompt,
        conditioning_scale=rargs.conditioning_scale,
        callback=step_logger(f'{label}realisr', rargs.num_inference_steps, rargs.log_every),
        callback_steps=max(rargs.log_every, 1),
    ).images[0]

    log('wavelet color fix + resize back', prefix='    ')
    out = wavelet_color_fix(out, validation_image)
    out = out.resize((ori_width * rscale, ori_height * rscale))

    if rargs.realisr_keep_chroma:
        # wavelet_color_fix keeps the INPUT's low frequencies but the MODEL's
        # high frequencies, and a sharp colour boundary (leaf against berry) is
        # a high-frequency event -- so the model gets to invent the colour right
        # where it matters most. Lock chroma to the input instead: realisr then
        # only sharpens luma and cannot shift any colour.
        src = image.convert("RGB")
        if src.size != out.size:
            src = src.resize(out.size, Image.LANCZOS)
        out_yuv = cv2.cvtColor(np.ascontiguousarray(np.asarray(out)[:, :, ::-1]),
                               cv2.COLOR_BGR2YUV)
        src_yuv = cv2.cvtColor(np.ascontiguousarray(np.asarray(src)[:, :, ::-1]),
                               cv2.COLOR_BGR2YUV)
        moved = np.abs(out_yuv[:, :, 1:3].astype(np.float32)
                       - src_yuv[:, :, 1:3].astype(np.float32)).mean()
        out_yuv[:, :, 1:3] = src_yuv[:, :, 1:3]
        out = Image.fromarray(cv2.cvtColor(out_yuv, cv2.COLOR_YUV2BGR)[:, :, ::-1])
        log(f'chroma locked to the colorized input (realisr had moved it by '
            f'{moved:.1f} on average)', prefix='    ')

    return out, validation_prompt


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def main(args, enable_xformers_memory_efficient_attention=True):
    run_t0 = time.perf_counter()
    args = resolve_model_paths(args)

    if args.disable_xformers:
        enable_xformers_memory_efficient_attention = False

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None:
        set_seed(args.seed)
    if not accelerator.is_main_process:
        return

    # every file this run produces goes straight into output/<output_tag>/
    out_dir = os.path.join(args.output_dir, args.output_tag) if args.output_tag \
        else args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = out_dir if args.save_debug else None
    open_log(os.path.join(out_dir, 'run.log'))

    log(f'=== run start {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
    log(f'device {accelerator.device}, mixed precision {args.mixed_precision}, '
        f'process size {args.process_size}, seed {args.seed}')
    log(f'colorization ckpt: {args.color_model_path}')
    log(f'realisr ckpt     : {args.realisr_model_path}')
    log(f'chroma           : {"optical flow" if args.chroma_flow else "homography"}'
        f'{", chained frame to frame" if args.chroma_chain else ", always from keyframe"}')
    # where the captioner will look for (or drop) its downloaded weights
    log(f'TORCH_HOME={os.environ.get("TORCH_HOME", "unset -> ~/.cache/torch")}  '
        f'HF_HOME={os.environ.get("HF_HOME", "unset -> ~/.cache/huggingface")}')

    key_paths, grey_paths = find_sequence(args.image_path, args.color_image)
    key_of, fallback = assign_keyframes(key_paths, grey_paths)
    log(f'keyframes  : {", ".join(os.path.basename(k) for k in key_paths)}')
    log(f'grey frames: {", ".join(os.path.basename(p) for p in grey_paths)}')
    for g in grey_paths:
        log(f'  {os.path.basename(g)} <- {os.path.basename(key_of[g])}'
            f'{"  (no earlier color frame: nearest one after)" if g in fallback else ""}')
    n_out = len(grey_paths) + len(key_paths)
    log(f'output dir : {out_dir}  ({n_out} *_final.png expected)')
    total_diffusions = len(grey_paths) * 2 + len(key_paths)
    log(f'plan       : {len(grey_paths)} colorize + {n_out} realisr '
        f'= {total_diffusions} diffusion runs')

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    # per-stage settings
    cargs = stage_args(
        args, "grayscale", args.color_model_path,
        num_inference_steps=(args.color_num_inference_steps
                             if args.color_num_inference_steps is not None
                             else args.num_inference_steps),
        conditioning_scale=args.color_conditioning_scale,
        prompt=args.color_prompt,
    )
    rargs = stage_args(args, "realisr", args.realisr_model_path)

    log(f'loading caption model ({"blip" if args.use_blip else "coca"}) ...')
    with GPUTimer() as t_net:
        nets = load_high_level_net(args, accelerator.device)
    log(f'caption model loaded in {t_net.elapsed:.2f} s')

    # ---------------- stage 1: colorize every grey frame ------------------- #
    log('')
    log(f'=== STAGE 1/2: colorization ({len(grey_paths)} frames) ===')
    log('loading colorization pipeline ...')
    with GPUTimer() as t_load:
        pipeline = load_pasd_pipeline(cargs, accelerator,
                                      enable_xformers_memory_efficient_attention)
    log(f'colorization pipeline loaded in {t_load.elapsed:.2f} s')

    key_rgbs = {k: np.asarray(Image.open(k).convert("RGB")) for k in key_paths}
    color_resize = make_resize_preproc(cargs)
    colorized = []

    # (keyframe, name, colorized image) of the previous grey frame -- the chain
    # source. It is only reused while the keyframe stays the same, i.e. no new
    # color frame came in between; fallback frames never start a chain.
    prev = None
    for n, grey_path in enumerate(grey_paths, start=1):
        name = os.path.splitext(os.path.basename(grey_path))[0]
        label = f'[colorize {n}/{len(grey_paths)}] '
        key = key_of[grey_path]
        if args.chroma_chain and prev is not None and prev[0] == key:
            ref_rgb = np.asarray(prev[2].convert("RGB"))
            ref_desc = f'chained from {prev[1]}_color (key {os.path.basename(key)})'
        else:
            ref_rgb = key_rgbs[key]
            ref_desc = f'key {os.path.basename(key)}'
        log(f'--- colorize {n}/{len(grey_paths)}: {os.path.basename(grey_path)} '
            f'({ref_desc}) ---')
        with GPUTimer() as t:
            image, chroma, prompt = colorize_frame(
                cargs, pipeline, nets, accelerator.device, generator,
                grey_path, ref_rgb, color_resize, debug_dir,
                label=label)
        path = os.path.join(out_dir, f'{name}_color.png')
        image.save(path)
        colorized.append((name, image))
        prev = None if grey_path in fallback else (key, name, image)
        log(f'colorize {n}/{len(grey_paths)} done in {t.elapsed:.2f} s '
            f'-> {os.path.basename(path)}', prefix='    ')

    # ---------------- stage 2: realisr on all N+1 frames ------------------- #
    # LCM lora is fused into the pipeline in stage 1 and its scheduler cannot
    # be swapped back to UniPC, so sharing is off whenever it is in use
    share = (not args.no_share_pipeline
             and not args.use_lcm_lora
             and args.color_model_path == args.realisr_model_path)
    log('')
    log(f'=== STAGE 2/2: realisr ({len(colorized) + len(key_paths)} frames) ===')
    if share:
        # same weights, different scheduler -- swap it instead of reloading
        pipeline.scheduler = UniPCMultistepScheduler.from_pretrained(
            args.pretrained_model_path, subfolder="scheduler")
        log('reusing the loaded pipeline (same checkpoint, scheduler swapped)')
    else:
        log('freeing colorization pipeline and loading the realisr one ...')
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with GPUTimer() as t_load:
            pipeline = load_pasd_pipeline(rargs, accelerator,
                                          enable_xformers_memory_efficient_attention)
        log(f'realisr pipeline loaded in {t_load.elapsed:.2f} s')

    realisr_resize = make_resize_preproc(rargs)
    jobs = [(os.path.splitext(os.path.basename(k))[0] + '_key',
             Image.open(k).convert("RGB")) for k in key_paths] + colorized

    for n, (name, image) in enumerate(jobs, start=1):
        label = f'[realisr {n}/{len(jobs)}] '
        log(f'--- realisr {n}/{len(jobs)}: {name} ---')
        with GPUTimer() as t:
            out, prompt = realisr_frame(rargs, pipeline, nets, accelerator.device,
                                        generator, image, realisr_resize, label=label)
        path = os.path.join(out_dir, f'{name}_final.png')
        out.save(path)
        log(f'realisr {n}/{len(jobs)} done in {t.elapsed:.2f} s '
            f'-> {os.path.basename(path)}', prefix='    ')

    log('')
    log(f'=== DONE: {len(jobs)} final images in {out_dir} '
        f'(total {time.perf_counter() - run_t0:.2f} s) ===')
    close_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- input / output --------------------------------------------------- #
    parser.add_argument("--image_path", type=str, default="examples/sequence",
                        help="folder holding one color keyframe + the grey frames")
    parser.add_argument("--color_image", type=str, default=None,
                        help="name/path of the color keyframe (default: auto-detect by saturation)")
    parser.add_argument("--output_dir", type=str, default="output", help="output folder")
    parser.add_argument("--output_tag", type=str, default="sequence", help="tag for this run")
    parser.add_argument("--save_debug", action="store_true",
                        help="also save the chroma conditioning + confidence maps")
    parser.add_argument("--log_every", type=int, default=10,
                        help="log denoising progress every N steps (0 turns step logging off); "
                             "all logs are flushed immediately and mirrored to <output>/run.log")

    # --- models ------------------------------------------------------------ #
    parser.add_argument("--pretrained_model_path", type=str,
                        default="checkpoints/stable-diffusion-v1-5", help="path of base SD model")
    parser.add_argument("--pasd_model_path", type=str, default=None,
                        help="PASD checkpoint for BOTH stages, as taken by the other "
                             "test_pasd_*.py scripts; the per-stage flags below override it")
    parser.add_argument("--color_model_path", type=str, default=None,
                        help=f"PASD checkpoint for the colorization stage "
                             f"(default: {DEFAULT_COLOR_MODEL})")
    parser.add_argument("--realisr_model_path", type=str, default=None,
                        help=f"PASD checkpoint for the realisr stage "
                             f"(default: {DEFAULT_REALISR_MODEL})")
    parser.add_argument("--control_type", choices=['realisr', 'grayscale'], nargs='?',
                        default=None,
                        help="accepted for CLI compatibility and ignored -- this script "
                             "always runs colorization then realisr")
    parser.add_argument("--lcm_lora_path", type=str, default="checkpoints/lcm-lora-sdv1-5",
                        help="path of LCM lora model")
    parser.add_argument("--personalized_model_path", type=str,
                        default="majicmixRealistic_v7.safetensors",
                        help="name of personalized dreambooth model in 'checkpoints/personalized_models'")
    parser.add_argument("--no_share_pipeline", action="store_true",
                        help="reload the pipeline between stages even when both use the same checkpoint")

    # --- prompting ---------------------------------------------------------- #
    parser.add_argument('--high_level_info', choices=['classification', 'detection', 'caption'],
                        nargs='?', default='caption', help="high level information for prompt generation")
    parser.add_argument("--prompt", type=str, default="", help="prompt for the realisr stage")
    parser.add_argument("--color_prompt", type=str, default="",
                        help="prompt for the colorization stage, e.g. 'highly saturated, vibrant colors'")
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k",
                        help="additional prompt for the realisr stage")
    parser.add_argument("--negative_prompt", type=str,
                        default="blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed",
                        help="negative prompt for the realisr stage")
    parser.add_argument("--color_negative_prompt", type=str, default="b&w, color bleeding",
                        help="negative prompt for the colorization stage")

    # --- chroma propagation (chroma_fixed_cam) ------------------------------ #
    parser.add_argument("--chroma_downscale", type=int, default=4,
                        help="coarse-resolution divisor for chroma/homography estimation")
    parser.add_argument("--no_homography", action="store_true",
                        help="skip ECC alignment between keyframe and grey frame")
    parser.add_argument("--residual_soft_thresh", type=float, default=12.0,
                        help="luma residual (0-255) where chroma confidence starts dropping")
    parser.add_argument("--residual_hard_thresh", type=float, default=35.0,
                        help="luma residual (0-255) where chroma is fully rejected")
    parser.add_argument("--guided_radius", type=int, default=8, help="guided-filter radius")
    parser.add_argument("--guided_eps", type=float, default=1e-3, help="guided-filter epsilon")
    parser.add_argument("--realisr_keep_chroma", type=int, choices=(0, 1), default=1,
                        help="1 = after realisr, keep only its luma and take the chroma "
                             "back from the colorized input, so realisr sharpens detail "
                             "without altering colour; 0 = keep realisr's own chroma "
                             "(wavelet_color_fix alone, the original behaviour)")
    parser.add_argument("--chroma_prop", type=int, choices=(0, 1), default=1,
                        help="1 = take the final chroma from the propagated keyframe, "
                             "0 = take it from the colorization model (the original "
                             "behaviour). Luma always comes from the grey frame.")
    parser.add_argument("--chroma_by_confidence", type=int, choices=(0, 1), default=1,
                        help="1 = blend per pixel by the propagation confidence, "
                             "SWAPPED: confidence 1 -> model chroma, confidence 0 -> "
                             "propagated chroma; 0 = use the propagated chroma "
                             "everywhere. Ignored when --chroma_prop is 0.")
    parser.add_argument("--chroma_flow", type=int, choices=(0, 1), default=1,
                        help="1 = carry keyframe chroma along dense optical flow so "
                             "colour follows moving objects; 0 = single global "
                             "homography (the _HomoChroma behaviour)")
    parser.add_argument("--chroma_chain", type=int, choices=(0, 1), default=1,
                        help="1 = each grey frame takes its chroma from the previous "
                             "grey frame's colorized output (restarting at every color "
                             "keyframe); 0 = always from the keyframe itself")
    parser.add_argument("--fb_soft_thresh", type=float, default=2.0,
                        help="optical flow: forward-backward error (px) below which "
                             "the flow is fully trusted")
    parser.add_argument("--fb_hard_thresh", type=float, default=6.0,
                        help="optical flow: forward-backward error (px) above which "
                             "the flow is rejected (model chroma takes over)")
    parser.add_argument("--desat_target", type=float, default=0.0,
                        help="how far propagate_keyframe_chroma pulls low-confidence "
                             "chroma toward neutral before the blend. Left at 0 so the "
                             "confidence handling lives in --chroma_by_confidence alone; "
                             "0.35 restores the chroma_fixed_cam default.")

    # --- diffusion ---------------------------------------------------------- #
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        help="mixed precision mode")  # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="classifier-free guidance scale")
    parser.add_argument("--conditioning_scale", type=float, default=1.0,
                        help="controlnet conditioning scale for the realisr stage")
    parser.add_argument("--color_conditioning_scale", type=float, default=0.7,
                        help="controlnet conditioning scale for the colorization stage")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="denoising steps for the realisr stage")
    parser.add_argument("--color_num_inference_steps", type=int, default=None,
                        help="denoising steps for the colorization stage (default: same as realisr)")
    parser.add_argument("--process_size", type=int, default=768,
                        help="minimal input size for processing")
    parser.add_argument("--blending_alpha", type=float, default=1.0,
                        help="blending alpha for personalized model")
    parser.add_argument("--multiplier", type=float, default=0.6,
                        help="multiplier for personalized lora model")
    parser.add_argument("--decoder_tiled_size", type=int, default=224, help="decoder tile size")
    parser.add_argument("--encoder_tiled_size", type=int, default=1024, help="encoder tile size")
    parser.add_argument("--latent_tiled_size", type=int, default=320, help="unet latent tile size")
    parser.add_argument("--latent_tiled_overlap", type=int, default=8, help="unet latent overlap")
    parser.add_argument("--upscale", type=int, default=1, help="upsampling scale")
    parser.add_argument("--use_personalized_model", action="store_true", help="use personalized model or not")
    parser.add_argument("--use_pasd_light", action="store_true", help="use pasd or pasd_light")
    parser.add_argument("--use_lcm_lora", action="store_true", help="use lcm-lora or not")
    parser.add_argument("--use_blip", action="store_true", help="use blip captioner")
    parser.add_argument("--caption_generation", choices=['top_p', 'top_k', 'beam_search'],
                        default='top_p',
                        help="CoCa decoding to fall back to when the default beam search "
                             "trips over the installed transformers version")
    parser.add_argument("--init_latent_with_noise", action="store_true",
                        help="initial latent with pure noise or not")
    parser.add_argument("--added_noise_level", type=int, default=900, help="additional noise level")
    parser.add_argument("--offset_noise_scale", type=float, default=0.0, help="offset noise scale, not used")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument('--disable_xformers', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        import traceback
        log('RUN FAILED:\n' + traceback.format_exc())
        close_log()
        raise

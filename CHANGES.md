# Changes log — fixed-camera chroma sequence work

Everything below was done in one session with Claude. It is split into files
that are **new**, existing repo files that were **modified**, and **data** that
was overwritten. Where an existing file was touched, the exact change and the
reason are given, so the diff against your original is reviewable.

Nothing in `test_pasd_NVDIA.py`, `test_pasd_AMD.py`, `test_pasd_sdxl.py`,
`train_pasd.py`, `BenchmarkPASD.py` or anything under `pasd/pipelines/` was
modified.

---

## 1. New files

### `shift_sequence.py`

Shifts the frames of an image sequence horizontally, in place, to create a
synthetic camera-motion test set.

Frame N is derived from a fixed unshifted reference frame (by default the frame
before `--start_index`) and displaced by `--shift * (N - reference_index)`
pixels, so frames 2–5 end up at +15, +30, +45 and +60 px. Deriving every frame
from the reference rather than from the previous output makes the script safe
to re-run: offsets never compound. The strip exposed at the leading edge is
filled with the background colour sampled from the image's own top-left pixel.

Options: `--image_path`, `--shift`, `--start_index`, `--cumulative` /
`--no-cumulative`, `--reference_index`.

### `test_pasd_sequence.py`

The sequence pipeline: one colour keyframe plus N grey frames in, N+1 colour
realisr-ed images out.

Per grey frame it runs chroma propagation from the keyframe, then PASD
colorization conditioned on that result, then PASD realisr. The keyframe itself
is realisr-ed once and emitted as the final output. The two diffusion stages run
as two passes so only one pipeline is resident at a time; when both stages point
at the same checkpoint the pipeline is loaded once and only its scheduler is
swapped.

Reused from the repo by import, not copied:

```python
from test_pasd_NVDIA import load_pasd_pipeline, load_high_level_net, get_validation_prompt
from chroma_fixed_cam import propagate_keyframe_chroma
from pasd.myutils.wavelet_color_fix import wavelet_color_fix
from BenchmarkPASD import GPUTimer
```

Note that `chroma_fixed_cam.py` already existed in the repo but was not imported
by anything — it had only its own `__main__` test harness. This script is the
first thing to wire it into a pipeline, which is why `desat_target` and the
confidence map suddenly affect real outputs.

---

## 2. Modified repo files

### `chroma_fixed_cam.py` — one bug fix, nothing else

`_guided_filter` probed for the wrong thing:

```python
# before
if hasattr(cv2, "ximgproc"):
    return cv2.ximgproc.guidedFilter(guide, src, radius, eps)

# after
_gf = getattr(getattr(cv2, "ximgproc", None), "guidedFilter", None)
if _gf is not None:
    return _gf(guide, src, radius, eps)
```

Plain `opencv-python` exposes a `cv2.ximgproc` namespace **without**
`guidedFilter` in it (that function ships only in `opencv-contrib-python`), so
the original guard passed and then raised `AttributeError` instead of falling
through to the NumPy fallback. The `pasd` environment has `opencv-python`, so
this crashed on the first call.

`desat_target`, the confidence map, the neutral fade at step 5 and the ECC
alignment are all **unchanged and original**. The committed file is your
original plus the three lines above.

### `pasd/myutils/wavelet_color_fix.py` — one bug fix, one behaviour change

**Bug.** `wavelet_reconstruction` dumped debug images of the style bands nine
lines before computing them, so it raised
`UnboundLocalError: local variable 'style_high_freq' referenced before assignment`
on its very first call. Both decompositions now happen before the dump. The
reconstruction maths is untouched — it still returns
`content_high_freq + style_low_freq`.

This bug meant **any** realisr run would crash, including `test_pasd_NVDIA.py`.
It went unnoticed because recent runs all used `--control_type grayscale`, which
never calls `wavelet_color_fix`.

**Behaviour change.** The debug dump is now opt-in, off by default, behind an
environment variable:

```
PASD_WAVELET_DEBUG=debug_steps python ...
```

It previously wrote four fixed-name PNGs on every call, which on a multi-image
run means only the last frame survives, and it silently no-opped when
`debug_steps/` did not exist (as on the cluster). The directory is now created
on demand and failures are reported.

---

## 3. Data overwritten

`examples/sequence/Strawberry19_2.jpg` … `_5.jpg` were replaced with versions
shifted right by 15, 30, 45 and 60 px respectively. `Strawberry19_1.jpg` (the
unshifted reference) and `Strawberry19.jpg` (the colour keyframe) are untouched.

---

## 4. New options in `test_pasd_sequence.py`

These do not exist anywhere in the original PASD code.

### Colour control

| Option | Default | Effect |
|---|---|---|
| `--chroma_prop` | 1 | 1 = final chroma from the propagated keyframe; 0 = from the colorization model (original PASD behaviour) |
| `--chroma_by_confidence` | 1 | 1 = weight the propagated chroma by per-pixel ECC confidence, so badly aligned pixels fall back to the model; 0 = propagated chroma everywhere |
| `--realisr_keep_chroma` | 1 | 1 = after realisr, keep only its luma and take chroma back from the colorized input; 0 = keep realisr's own chroma |
| `--desat_target` | 0.0 | Passed to `propagate_keyframe_chroma`. Its own default is 0.35; the script asks for 0 so confidence handling lives in `--chroma_by_confidence` alone |

**Why `--chroma_prop` exists.** The original grayscale path keeps the model's
chroma and the input's luma. Measured against the keyframe, the colorization
model desaturated the strawberry by 36 points, darkened it by 39 and shifted the
hue 13° toward orange, and invented green seeds on the shifted frames. The
propagated chroma it was conditioned on was accurate; the model overrode it.

**Why `--realisr_keep_chroma` exists.** `wavelet_color_fix` keeps the input's
*low* frequencies and the model's *high* frequencies. A sharp colour boundary —
a leaf against the berry — is high-frequency, so the model decided its colour.
Measured in the leaf region, the colorization stage left 0.0% red pixels on every
frame; after realisr, frame `_5` had 40.8%, and the chroma gradient across the
boundary rose ~55% on all frames. The technique used here (a YUV channel swap)
is borrowed from the original grayscale branch of `test_pasd_NVDIA.py` at lines
291–300, applied in the opposite direction: luma from the model, chroma from the
input.

### Progress logging

`--log_every` (default 10) logs denoising progress every N steps with an ETA,
via the pipeline's own `callback` hook; 0 turns step lines off.

All log lines are flushed immediately and stdout is set line-buffered, because
PBS redirects stdout to a file where Python block-buffers it — without this,
`pasd.o<jobid>` stays empty until the job ends. Every line is mirrored to
`<output_dir>/<output_tag>/run.log`, and a traceback is appended there if the
run dies. Note `run.log` is opened in append mode, so re-running with the same
tag concatenates rather than replaces.

### Robustness

| Option | Purpose |
|---|---|
| `--pasd_model_path`, `--control_type` | Accepted for compatibility with the other `test_pasd_*.py` CLIs. `--pasd_model_path` sets both stages; `--control_type` is accepted and ignored, since this script always runs colorization then realisr |
| `--color_model_path`, `--realisr_model_path` | Per-stage checkpoints; override `--pasd_model_path` |
| `--caption_generation` | CoCa decoding to fall back to (`top_p` default) |
| `--color_prompt`, `--color_conditioning_scale`, `--color_num_inference_steps`, `--color_negative_prompt` | Colorization-stage overrides, since it wants different settings from realisr |
| `--save_debug` | Also writes the chroma conditioning image and confidence map per frame |
| `--no_share_pipeline` | Reload between stages even when both use the same checkpoint |

**Captioning is non-fatal.** open_clip's CoCa defaults to beam search, which
calls transformers' `BeamSearchScorer`; its internals moved in newer
transformers, so it dies with
`ValueError: Batch dimension of 'input_ids' should be 0, but is 6`.
(`env_windows.yml` pins `transformers==5.8.0` while `requirements.txt` asks for
`4.44.0`.) `safe_caption()` tries the reused `get_validation_prompt`, retries
with top-p sampling, and finally falls back to the bare `--prompt` rather than
killing a long run over a caption.

**Input discovery.** The colour keyframe is detected by saturation, or named
with `--color_image`. Grey frames are ordered by their trailing `_<index>`. A
bad `--image_path` reports the working directory and lists directories of the
same name found nearby, instead of failing opaquely on a cluster.

---

## 5. Measurements worth keeping

Collected on `examples/sequence` with the shifted frames.

**ECC alignment absorbs the shifts.** With alignment on, luma residual stays
~3 and mean confidence ~0.997 at every shift out to 60 px. With
`--no_homography` the residual climbs 2.55 → 57.69 and confidence falls
0.997 → 0.472, producing a red ghost where the berry used to be and a
desaturated right flank.

**Confidence does not see the calyx error.** Chroma error against a synthetic
ground truth (the keyframe shifted by the known offset) sits at ~13.6 at the
leaf/berry boundary on *every* frame including the unshifted one, while
confidence stays ~0.998 there. Correlation between confidence and error is only
−0.13 to −0.21. This error is resolution loss — chroma is computed at
`chroma_downscale=4` and upsampled with a radius-8 guided filter, which cannot
resolve thin leaves — not misalignment.

**Guided-filter radius matters more than confidence here.**

```
guided_radius   calyx err   whole-frame err
        1           6.63          2.78
        2           6.89          2.90
        4           7.44          3.14
        8 (default) 8.27          3.48
       16           9.35          4.07
```

`--guided_radius 2` is worth trying. `--chroma_downscale` should stay at 4: 8
is much worse (edge error 22) and 1–2 are no better, because ECC gets less
robust at finer scales.

**realisr is colour-faithful when its input is.** realisr applied to the true
colour keyframe changed saturation by −2.1 and hue by −0.6°. Applied to a
degraded colorization it left global saturation alone (−1.0) but roughly doubled
chroma that had already crossed an edge (halo saturation 6.2 → 14.5 on frame 5).

---

## 6. Suggested starting command

```bash
export TORCH_HOME=/scratch/$USER/.torch     # so the captioner's weights are cached
export HF_HOME=/scratch/$USER/.hf           # between jobs rather than re-downloaded

SIZE=512
python test_pasd_sequence.py \
  --pasd_model_path runs/pasd_light_color/checkpoint-180000 \
  --use_pasd_light --use_blip --high_level_info caption \
  --image_path ./datasets/sequence \
  --output_dir output/timestamp \
  --output_tag "Strawberry19_size${SIZE}_sequence" \
  --process_size "$SIZE" --num_inference_steps 50 --mixed_precision no \
  --chroma_prop 1 --chroma_by_confidence 1 --realisr_keep_chroma 1 \
  --guided_radius 2 --save_debug
```

The BLIP captioner downloads a large checkpoint on first use (317 s in one
observed run); `TORCH_HOME` on shared scratch makes that a one-off. LAVIS's
`download_cached_file` skips the download when the file already exists.

---

## 7. Known open items

- The propagation's calyx error (section 5) is unaddressed beyond
  `--guided_radius`. Fixing it properly means resolving chroma at the leaf scale,
  e.g. an edge-aware upsample driven by the full-resolution luma.
- `--chroma_by_confidence` cannot help on this sequence, because a rigid
  translation is exactly what the homography undoes — confidence never drops.
  It is insurance for real footage with occlusion or per-object motion.
- Whether realisr should run *before* colorization was discussed but not
  implemented. Running it first, keeping only its luma, would give the
  chroma-propagation step a sharper guide and let the colour-defining stage
  run last.
- `run.log` appends rather than truncates.

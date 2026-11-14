# MComp Capstone: PASD for Image Colorization and Super-Resolution

This project builds on **[Pixel-Aware Stable Diffusion (PASD)](https://github.com/yangxy/PASD)** (ECCV 2024). On top of the original code it adds:

- **AMD DirectML support** for running locally on Windows with an AMD GPU (`test_pasd_AMD.py`), alongside the CUDA version (`test_pasd_NVDIA.py`).
- **A fixed-camera sequence pipeline** (`test_pasd_sequence.py`): colour keyframes plus grey frames in, colorized and super-resolved frames out, with keyframe chroma propagated by homography or optical flow (`chroma_fixed_cam.py`).
- **Evaluation tools**: HTML quality reports (`matrix.py`, `compareColor.py`) and a timing benchmark (`BenchmarkPASD.py`).
- **A PBS job script** for running on an HPC cluster (`parallel.pbs`).

## Requirements

- Python 3.10
- A GPU: NVIDIA with CUDA 12.1, or an AMD GPU on Windows through DirectML
- About 30 GB of free disk space for model weights
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sofialeotag/MComp_Capstone.git
cd MComp_Capstone
```

### 2. Create the environment

**NVIDIA (Linux or Windows)**

```bash
conda create -n pasd python=3.10 -y
conda activate pasd
pip install -r requirements.txt
```

Add BLIP captioning (needed for `--use_blip`):

```bash
pip install -r requirements-test.txt
```

**AMD on Windows (DirectML)**

```bash
conda create -n pasd python=3.10 -y
conda activate pasd
pip install -r requirements.txt
pip install torch-directml
```

Alternatively, `env_windows.yml` is a full export of the environment this project was developed in:

```bash
conda env create -f env_windows.yml
conda activate pasd
```

> Remove the last line (`prefix: ...`) from `env_windows.yml` first, since it points at a path on the original machine.

Optional packages for the evaluation scripts:

```bash
pip install scipy scikit-image matplotlib tabulate tqdm brisque
```

### 3. Download the model weights

Weights are not stored in this repository (they are git-ignored). You need two sets:

| What | Source | Goes in |
|---|---|---|
| Stable Diffusion v1.5 base | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `checkpoints/stable-diffusion-v1-5/` |
| PASD checkpoints | [yangtao9009/PASD](https://huggingface.co/yangtao9009/PASD/tree/main) | `runs/` |

`python download_sd.py` fetches both. After downloading, check that the folders match the layout below and move them if they do not:

```
checkpoints/
└── stable-diffusion-v1-5/        # unet/, vae/, text_encoder/, tokenizer/, scheduler/, feature_extractor/
runs/
├── pasd/checkpoint-100000/       # super-resolution (default model)
│   ├── controlnet/
│   └── unet/
├── pasd_light/checkpoint-120000/
├── pasd_rrdb/checkpoint-100000/
├── pasd_light_rrdb/checkpoint-100000/
└── pasd_light_color/             # colorization
    ├── checkpoint-180000/
    └── scheduler/scheduler_config.json
```

Only needed for specific options:

- `--high_level_info detection`: put [yolov8n.pt](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/yolov8n.pt) in `pasd/annotator/ckpts/`.
- `gradio_pasd.py`: put [RetinaFace-R50.pth](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/RetinaFace-R50.pth) in `pasd/annotator/ckpts/`.
- `--use_personalized_model`: put the model file in `checkpoints/personalized_models/`.

The captioning models (CoCa, or BLIP with `--use_blip`) download automatically on first run.

## Running

Use `test_pasd_NVDIA.py` on NVIDIA GPUs and `test_pasd_AMD.py` on AMD/DirectML; they take the same arguments. Results are written to `output/` (change with `--output_dir`).

### Super-resolution

```bash
python test_pasd_NVDIA.py --image_path examples/dog.png --upscale 2
```

### Colorization

```bash
python test_pasd_NVDIA.py \
  --pasd_model_path runs/pasd_light_color/checkpoint-180000 \
  --control_type grayscale --use_pasd_light \
  --image_path ./examples/test_realisr.png \
  --prompt "forest, grey house" \
  --num_inference_steps 50 --mixed_precision no \
  --output_tag ForestGreyHouse
```

On AMD, replace `test_pasd_NVDIA.py` with `test_pasd_AMD.py` and keep `--mixed_precision no`. On Windows `cmd`, put the command on one line or replace `\` with `^`.

`--image_path` accepts a single image or a folder. Sample inputs are in `examples/`.

### Image sequences (colour keyframe + grey frames)

```bash
python test_pasd_sequence.py \
  --pasd_model_path runs/pasd_light_color/checkpoint-180000 \
  --use_pasd_light \
  --image_path examples/sequence \
  --process_size 512 --num_inference_steps 50 --mixed_precision no \
  --output_tag seq01 --save_debug
```

The input folder holds one or more colour keyframes and the grey frames. Grey frames are ordered by the trailing `_<index>` in their file name; colour frames are detected automatically, or named explicitly with `--color_image`. Each run writes to `<output_dir>/<output_tag>/` with a `run.log`. This script needs a CUDA GPU.

`shift_sequence.py` creates a synthetic camera-motion test set by shifting frames horizontally.

### Common options

| Option | Default | Purpose |
|---|---|---|
| `--pasd_model_path` | `runs/pasd/checkpoint-100000` | PASD checkpoint to use |
| `--control_type` | `realisr` | `realisr` (super-resolution) or `grayscale` (colorization) |
| `--use_pasd_light` | off | Required when the checkpoint is a `pasd_light*` model |
| `--prompt` | empty | Text prompt; combined with the automatic caption |
| `--high_level_info` | `caption` | Prompt source: `caption`, `classification` or `detection` |
| `--use_blip` | off | Caption with BLIP instead of CoCa |
| `--process_size` | 768 | Minimum side length for processing; lower it to save memory |
| `--num_inference_steps` | 20 | Denoising steps |
| `--conditioning_scale` | 1.0 | ControlNet strength |
| `--guidance_scale` | 9.0 | Classifier-free guidance scale |
| `--upscale` | 1 | Upsampling factor |
| `--mixed_precision` | `fp16` | `no`, `fp16` or `bf16` |
| `--seed` | none | Fix for reproducible output |

Run any script with `-h` for the full list.

### Evaluation

Compare colorized outputs with ground-truth colour images:

```bash
python compareColor.py --pred_dir output/colorisation_size512 --gt_dir datasets/test_color --out_html report.html
```

Score many outputs against one grey input:

```bash
python matrix.py --lr input_gray.png --hr_dir ./output/my_run --report quality_report.html
```

Time a run:

```bash
python BenchmarkPASD.py cmd --repeats 3 -- python test_pasd_NVDIA.py --image_path examples/dog.png
```

### Running on an HPC cluster (PBS)

`parallel.pbs` is an example job script. Before submitting, change the `source activate ...` line to your own environment path, point the input paths at your own data, and adjust the `#PBS` resource lines for your cluster.

```bash
qsub parallel.pbs
```

Setting `TORCH_HOME` and `HF_HOME` to shared storage keeps the captioner weights cached between jobs.

### Training

Launch `train_pasd.py` through `accelerate launch`. Place training data under `datasets/` and see the [original PASD repository](https://github.com/yangxy/PASD) for dataset preparation.

## Project layout

```
pasd/                        PASD models, pipelines, data loaders, utilities
test_pasd_NVDIA.py           Single-image inference (CUDA)
test_pasd_AMD.py             Single-image inference (AMD DirectML)
test_pasd_sdxl.py            SDXL variant (needs SDXL weights)
test_pasd_sequence.py        Sequence pipeline (optical-flow chroma propagation)
test_pasd_sequence_HomoChroma.py   Snapshot of the homography-only sequence pipeline
chroma_fixed_cam.py          Keyframe chroma propagation
shift_sequence.py            Builds a shifted test sequence
compareColor.py, matrix.py   HTML quality reports
BenchmarkPASD.py             Timing benchmark
gradio_pasd.py               Gradio demo from the original PASD
train_pasd.py                Training
parallel.pbs                 PBS job script
examples/                    Sample inputs (incl. sequence/ and captures/)
output/                      Results from this project's runs
quality_report.html, report_colorization.html   Evaluation reports
meeting slides/              Project meeting slides
eoFiles/                     PBS job stdout/stderr logs
Claude outputs/              Diagnostic figures from the chroma analysis
```

## Troubleshooting

- **Out of memory**: lower `--process_size` (512 or 256) and `--latent_tiled_size`.
- **`xformers` errors**: add `--disable_xformers` (NVIDIA script and sequence script).
- **Captioning fails with `Batch dimension of 'input_ids' should be 0`**: this comes from a `transformers` version newer than the `4.44.0` in `requirements.txt`. Install that version, or pass your own `--prompt`.

## Credits and license

Based on [PASD](https://github.com/yangxy/PASD) by Tao Yang et al. Released under the Apache License 2.0; see [LICENSE](LICENSE).

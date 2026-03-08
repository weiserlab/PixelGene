# MComp Capstone: PASD for Authentic Image Generation

This project implements **Pixel-Aware Stable Diffusion (PASD)**, modified to support **AMD DirectML** for local execution on Windows.

## 🚀 Getting Started
1. **Clone the repo:**
   `git clone https://github.com/sofialeotag/MComp_Capstone.git`
2. **Install requirements:**
   `pip install -r requirements.txt`
3. **Download Models:**
   Run `python download_models.py` to fetch Stable Diffusion v1.5 and PASD checkpoints.
4. **Reference Page:**
   https://github.com/yangxy/PASD/?tab=readme-ov-file#colorization

## 🛠 Modifications
- Changed original CUDA/PyTorch logic to support **AMD DirectML** for Windows users.
- Optimized for local inference on consumer GPUs (RTX 4000 series / AMD Radeon).

## 🖼 Examples
- To be updated
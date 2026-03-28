# MComp Capstone: PASD for Authentic Image Generation

This project implements **[Pixel-Aware Stable Diffusion (PASD)](https://github.com/yangxy/PASD/?tab=readme-ov-file#colorization)**, modified to support **AMD DirectML** for local execution on Windows.

🚀 Getting StartedTo get this project running on your local machine or a GPU server, follow these steps.

1. InstallationFirst, clone the repository and install the necessary Python libraries:Bashgit clone git@github.com:sofialeotag/MComp_Capstone.git
cd MComp_Capstone
pip install -r requirements.txt

2. Model Weights SetupDue to file size limits, model checkpoints are not included in this repository. Expected files size 6-7GB. You must download and place them in the checkpoints/ directory following this structure:

   1. Stable Diffusion v1.5 checkpoints/stable-diffusion-v1-5/
   huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir checkpoints/stable-diffusion-v1-5
   
   2. PASD Weights runs/
   https://huggingface.co/yangtao9009/PASD/tree/main
   
   3. ControlNet / ResNetcheckpoints/controlnet/
   huggingface-cli download lllyasviel/ControlNet-v1-1 control_v11f1e_sd15_tile.pth --local-dir checkpoints/controlnet 

   Expected Directory Structure:
   PASD-main/
   ├── checkpoints/
   │   ├── stable-diffusion-v1-5/
   │   │   ├── unet/
   │   │   └── ... (tokenizer, vae, etc.)
   │   └── personalized_models/
   ├── runs/
   │   ├── pasd/
   │   │   └── checkpoints-100000
   │   ├── pasd_light/
   │   ├── pasd_light_color/
   │   │   └── checkpoints-180000
   │   │   └── scheduler
   │   ├── pasd_light_rrdb/
   │   ├── pasd_rrdb/
   ├── pasd/           # Core model architecture
   ├── test_pasd.py    # Main inference script
   └── run.bat
   └── ...

3. Running Inference
Once the weights are in place, you can run the super-resolution test script: python test_pasd.py
Outputs will be saved to the /output folder. sample running command can be found in run.bat.

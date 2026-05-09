# MComp Capstone: PASD for Authentic Image Generation

This project implements **[Pixel-Aware Stable Diffusion (PASD)](https://github.com/yangxy/PASD/?tab=readme-ov-file#colorization)**, modified to support **AMD DirectML** for local execution on Windows.

🚀 Getting StartedTo get this project running on your local machine or a GPU server, follow these steps.

1. InstallationFirst, clone the repository and install the necessary Python libraries:Bashgit clone git@github.com:sofialeotag/MComp_Capstone.git
cd MComp_Capstone
pip install -r requirements.txt (Recommended to build new environment)

2. Model Weights:
   https://huggingface.co/yangtao9009/PASD/tree/main


3. Running Inference
Once the weights are in place, under built environment, you can run the super-resolution test script: python test_pasd_AMD.py or python test_pasd_NVDIA.py

Colorization tests: (AMD)
python test_pasd_AMD.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png  --prompt "forest, grey house" --num_inference_steps 50 --mixed_precision no --output_tag ForestGreyHouse

Colorization tests: (NVDIA)
python test_pasd_NVDIA.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png  --prompt "forest, grey house" --num_inference_steps 50 --mixed_precision no --output_tag ForestGreyHouse

Outputs will be saved to the /output folder. sample running command can be found in run.bat.

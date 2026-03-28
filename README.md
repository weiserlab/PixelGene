# MComp Capstone: PASD for Authentic Image Generation

This project implements **[Pixel-Aware Stable Diffusion (PASD)](https://github.com/yangxy/PASD/?tab=readme-ov-file#colorization)**, modified to support **AMD DirectML** for local execution on Windows.

🚀 Getting StartedTo get this project running on your local machine or a GPU server, follow these steps.

1. Clone the repository and install the necessary Python libraries: git clone git@github.com:sofialeotag/MComp_Capstone.git
cd MComp_Capstone
pip install -r requirements.txt

2. Model Weights:
   https://huggingface.co/yangtao9009/PASD/tree/main

3. Running Inference
Once the weights are in place, you can run the super-resolution test script: python test_pasd.py
Outputs will be saved to the /output folder. sample running command can be found in run.bat.

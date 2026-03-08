import os
from huggingface_hub import snapshot_download

def setup_checkpoints():
    print("--- PASD Model Downloader ---")
    confirm = input("This will download several GBs of data. Proceed? (y/n): ").lower()
    
    if confirm != 'y':
        print("Download aborted.")
        return

    # 1. Download Stable Diffusion v1.5 Base
    print("\n[1/2] Downloading Stable Diffusion v1.5...")
    snapshot_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        local_dir="./checkpoints/stable-diffusion-v1-5",
        local_dir_use_symlinks=False
    )

    # 2. Download PASD Specific Checkpoints
    # Note: Many PASD scripts expect these in the 'runs/' folder
    print("\n[2/2] Downloading PASD Checkpoints from yangtao9009/PASD...")
    snapshot_download(
        repo_id="yangtao9009/PASD",
        local_dir="./runs/pasd", 
        local_dir_use_symlinks=False
    )

    print("\nDone! All models are placed in their respective folders.")
    print("Ensure these folders are in your .gitignore before pushing to GitHub.")

if __name__ == "__main__":
    setup_checkpoints()
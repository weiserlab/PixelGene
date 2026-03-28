call C:\Users\WEISER\miniconda3\Scripts\activate.bat
call conda activate pasd

set tag=mixed_vae32_conscale0.7_size320
set scale=0.7
python test_pasd.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png --process_size 320 --prompt "highly saturated, vibrant colors, vivid green forest, grey house" --num_inference_steps 30 --mixed_precision fp16 --conditioning_scale %scale% --output_tag %tag%

pause
REM set tag=mixed_conscale0.9
REM set scale=0.9
REM python test_pasd.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png --process_size 256 --prompt "highly saturated, vibrant colors, vivid green forest, grey house" --num_inference_steps 30 --conditioning_scale %scale% --mixed_precision no --output_tag %tag%

set tag=mixed_conscale0.7
set scale=0.7
python test_pasd.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png --process_size 256 --prompt "highly saturated, vibrant colors, vivid green forest, grey house" --num_inference_steps 30 --conditioning_scale %scale% --mixed_precision no --output_tag %tag%

REM set tag=mixed_conscale0.5
REM set scale=0.5
REM python test_pasd.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png --process_size 256 --prompt "highly saturated, vibrant colors, vivid green forest, grey house" --num_inference_steps 30 --conditioning_scale %scale% --mixed_precision no --output_tag %tag%

REM set tag=mixed_conscale0.3
REM set scale=0.3
REM python test_pasd.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png --process_size 256 --prompt "highly saturated, vibrant colors, vivid green forest, grey house" --num_inference_steps 30 --conditioning_scale %scale% --mixed_precision no --output_tag %tag%

REM set tag=mixed_conscale0.1
REM set scale=0.1
REM python test_pasd.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png --process_size 256 --prompt "highly saturated, vibrant colors, vivid green forest, grey house" --num_inference_steps 30 --conditioning_scale %scale% --mixed_precision no --output_tag %tag%

REM set tag=mixed_conscale0
REM set scale=0.0
REM python test_pasd.py --pasd_model_path runs/pasd_light_color/checkpoint-180000 --control_type grayscale --high_level_info --use_pasd_light --image_path ./examples/test_realisr.png --process_size 256 --prompt "highly saturated, vibrant colors, vivid green forest, grey house" --num_inference_steps 30 --conditioning_scale %scale% --mixed_precision no --output_tag %tag%
pause
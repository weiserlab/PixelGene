'''
# --------------------------------------------------------------------------------
#   Color fixed script from Li Yi (https://github.com/pkuliyi2015/sd-webui-stablesr/blob/master/srmodule/colorfix.py)
# --------------------------------------------------------------------------------
'''

import os

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from torchvision.transforms import ToTensor, ToPILImage
import cv2

def adain_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply adaptive instance normalization
    result_tensor = adaptive_instance_normalization(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def wavelet_color_fix(target: Image, source: Image):
    # Convert images to tensors
    to_tensor = ToTensor()
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)

    # Apply wavelet reconstruction
    result_tensor = wavelet_reconstruction(target_tensor, source_tensor)

    # Convert tensor back to image
    to_image = ToPILImage()
    result_image = to_image(result_tensor.squeeze(0).clamp_(0.0, 1.0))

    return result_image

def calc_mean_std(feat: Tensor, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.
    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.reshape(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().reshape(b, c, 1, 1)
    feat_mean = feat.reshape(b, c, -1).mean(dim=2).reshape(b, c, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat:Tensor, style_feat:Tensor):
    """Adaptive instance normalization.
    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.
    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output

def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += (image - low_freq)
        image = low_freq

    return high_freq, low_freq

def tensor_to_image(t: Tensor, normalize_residual: bool = False):
    """Convert a BCHW or CHW tensor to a uint8 BGR image for cv2."""
    if t.dim() == 4:
        t = t[0]                      # take first item in batch
    img = t.detach().float().cpu()

    if normalize_residual:
        # high-freq is a residual centered at 0 (has negatives)
        # shift so 0 -> gray (128), scale to make detail visible
        # img = img * 0.5 + 0.5         # assume roughly [-1, 1] -> [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    # if your low-freq is in [-1, 1] (diffusion convention), also rescale:
    # img = img * 0.5 + 0.5

    img = img.clamp(0, 1).mul(255).byte()
    img = img.permute(1, 2, 0).numpy()      # CHW -> HWC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# Set PASD_WAVELET_DEBUG to a folder name to dump the four wavelet bands of
# every wavelet_color_fix() call (off by default: it is a per-call side effect
# and the files overwrite each other on a multi-image run).
DEBUG_DIR = os.environ.get('PASD_WAVELET_DEBUG', '')


def _dump_bands(content_high, content_low, style_high, style_low):
    """Write the four wavelet bands to DEBUG_DIR for inspection."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    bands = (('content_high', content_high, True), ('content_low', content_low, False),
             ('style_high', style_high, True), ('style_low', style_low, False))
    for name, tensor, residual in bands:
        path = os.path.join(DEBUG_DIR, f'{name}.png')
        if not cv2.imwrite(path, tensor_to_image(tensor, normalize_residual=residual)):
            print(f'WARNING: failed to save {path}')


def wavelet_reconstruction(content_feat:Tensor, style_feat:Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of both features first -- the debug
    # dump below reads the style bands, so they have to exist by then
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)

    if DEBUG_DIR:
        _dump_bands(content_high_freq, content_low_freq,
                    style_high_freq, style_low_freq)

    del content_low_freq, style_high_freq
    # reconstruct the content feature with the style's low frequency
    return content_high_freq + style_low_freq

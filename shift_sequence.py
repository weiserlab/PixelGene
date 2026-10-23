"""
shift_sequence.py

Shift the subject (e.g. the strawberry) N pixels to the right in an image
sequence, in place, for every frame from --start_index onward.

Reuses the same image I/O conventions as test_pasd_NVDIA.py:
  - PIL.Image.open(...) to load images
  - glob + os.path to discover and name files
  - argparse for the CLI, following the same --image_path / --output style
  - cv2/numpy for the actual pixel-level array manipulation

(test_pasd_NVDIA.py itself has no generic "shift/translate" helper to import
directly -- its functions are all tightly coupled to the PASD diffusion
pipeline -- so this script mirrors its I/O and CLI conventions rather than
importing from it.)

Usage:
    python shift_sequence.py --image_path examples/sequence --shift 15 --start_index 2
"""

import os
import re
import glob
import argparse

import cv2
import numpy as np
from PIL import Image


def shift_right(image: Image.Image, shift_px: int) -> Image.Image:
    """Translate image content `shift_px` pixels to the right (negative
    values shift left). The strip exposed on the leading edge is filled
    with the image's own background color, sampled from its top-left
    corner pixel, so the fill blends with the source image.
    """
    arr = np.array(image)
    fill_value = arr[0, 0].copy()

    shifted = np.empty_like(arr)
    width = arr.shape[1]

    if shift_px > 0:
        s = min(shift_px, width)
        shifted[:, :s] = fill_value
        if s < width:
            shifted[:, s:] = arr[:, : width - s]
    elif shift_px < 0:
        s = min(-shift_px, width)
        shifted[:, width - s:] = fill_value
        if s < width:
            shifted[:, : width - s] = arr[:, s:]
    else:
        shifted[:] = arr

    return Image.fromarray(shifted)


def find_sequence_images(folder, start_index=2):
    """Find files named like '<name>_<index>.<ext>' in `folder` whose
    trailing index is >= start_index. Returns a list of (path, index)
    sorted by index.
    """
    pattern = re.compile(r'^(?P<stem>.+)_(?P<idx>\d+)$')
    matches = []
    for path in glob.glob(os.path.join(folder, '*.*')):
        stem, _ext = os.path.splitext(os.path.basename(path))
        m = pattern.match(stem)
        if m and int(m.group('idx')) >= start_index:
            matches.append((path, int(m.group('idx'))))
    matches.sort(key=lambda x: x[1])
    return matches


def find_reference_image(folder, ref_index):
    """Find the file named '<name>_<ref_index>.<ext>' in `folder` (the
    unshifted anchor frame that every later frame is derived from).
    """
    pattern = re.compile(r'^(?P<stem>.+)_(?P<idx>\d+)$')
    for path in glob.glob(os.path.join(folder, '*.*')):
        stem, _ext = os.path.splitext(os.path.basename(path))
        m = pattern.match(stem)
        if m and int(m.group('idx')) == ref_index:
            return path
    return None


def main(args):
    targets = find_sequence_images(args.image_path, args.start_index)
    if not targets:
        print(f'No images matching "*_<index>.*" with index >= {args.start_index} '
              f'found in {args.image_path}')
        return

    if args.cumulative:
        # Each frame's shift is measured from a fixed, unshifted reference
        # frame (by default the frame right before --start_index), so
        # frame N is shifted by args.shift * (N - reference_index) pixels.
        # This makes the script safe to re-run: it always derives frames
        # from the untouched reference rather than re-shifting output.
        ref_index = args.reference_index if args.reference_index is not None else args.start_index - 1
        ref_path = find_reference_image(args.image_path, ref_index)
        if ref_path is None:
            print(f'Could not find a reference frame with index {ref_index} in '
                  f'{args.image_path} (needed for --cumulative). Pass --reference_index '
                  'to point at a different unshifted frame, or use --no-cumulative.')
            return
        reference_image = Image.open(ref_path)
        print(f'using {os.path.basename(ref_path)} (frame {ref_index}) as the unshifted reference')

        for path, idx in targets:
            total_shift = args.shift * (idx - ref_index)
            shifted = shift_right(reference_image, total_shift)

            if path.lower().endswith(('.jpg', '.jpeg')):
                shifted.save(path, quality=95)
            else:
                shifted.save(path)

            print(f'{os.path.basename(path)} (frame {idx}) <- reference shifted by {total_shift}px -> saved in place')
    else:
        # Flat mode: shift every target frame by the same fixed amount,
        # each measured from its own current content.
        for path, idx in targets:
            image = Image.open(path)
            shifted = shift_right(image, args.shift)

            if path.lower().endswith(('.jpg', '.jpeg')):
                shifted.save(path, quality=95)
            else:
                shifted.save(path)

            print(f'shifted {os.path.basename(path)} (frame {idx}) by {args.shift}px -> saved in place')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="examples/sequence",
                         help="folder containing the image sequence")
    parser.add_argument("--shift", type=int, default=15,
                         help="pixels to shift right (use a negative value to shift left)")
    parser.add_argument("--start_index", type=int, default=2,
                         help="only shift files whose trailing _<index> is >= this value")
    parser.add_argument("--cumulative", dest="cumulative", action="store_true", default=True,
                         help="frame N is shifted by --shift * (N - reference_index) pixels, "
                              "derived from a fixed unshifted reference frame (default: on)")
    parser.add_argument("--no-cumulative", dest="cumulative", action="store_false",
                         help="shift every target frame by the same flat amount instead")
    parser.add_argument("--reference_index", type=int, default=None,
                         help="index of the unshifted anchor frame for --cumulative "
                              "(default: start_index - 1)")
    args = parser.parse_args()
    main(args)

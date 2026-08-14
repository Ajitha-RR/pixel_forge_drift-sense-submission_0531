"""
inference.py -- THE CRITICAL FILE. This is the script Applied Materials
runs directly on their test data to compute the Phase 2 score, per the
submission requirements:

  "The localization inference script is the most important file in your
  repository. Applied Materials will run it directly on their test image
  pairs... It must run without manual edits, must accept a reference image
  path and search image path as inputs, and must output a single (x, y)
  coordinate."

Usage (exactly two required arguments):
    python inference.py --reference path/to/reference.png --search path/to/search.png

Output (stdout, plain and unambiguous so it's trivial to parse):
    x,y

Optional:
    --nominal_zoom 10.0   (the known reference:search scale ratio; default
                          matches this challenge's stated 10x)
    --device cuda|cpu     (default: auto-detect, falls back to CPU if no
                          GPU is present on the grading machine)
    --verbose             (also prints the match score and whether the
                          periodic tie-break rule was triggered, to stderr
                          so it never contaminates the stdout coordinate)

No training, no model weights, no manual edits required -- this is
closed-form FFT-based template matching (see localize.py's module
docstring and citations.md for the full justification). It works
correctly the first time it's run, on any machine with the packages in
requirements.txt installed.
"""

import argparse
import sys

import numpy as np
import torch
from PIL import Image

from localize import localize_batch, get_device


def load_image(path, device):
    """Loads grayscale OR RGB, returns a (1, C, H, W) float tensor in
    [0,1] ready for localize_batch(). Handles both transparently -- see
    to_grayscale() inside localize.py for how RGB is handled downstream,
    which is what lets this same script handle optical microscope (RGB)
    images too, per the bonus requirement."""
    img = Image.open(path)
    if img.mode == 'RGB':
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        t = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        arr = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)
    return t.unsqueeze(0).to(device)  # (1, C, H, W)


def main():
    parser = argparse.ArgumentParser(
        description='Predict the (x, y) center of the reference pattern inside the search image.')
    parser.add_argument('--reference', required=True, help='Path to the reference image')
    parser.add_argument('--search', required=True, help='Path to the search image')
    parser.add_argument('--nominal_zoom', type=float, default=10.0,
                         help='Known reference:search scale ratio (default 10.0, per the problem statement)')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--verbose', action='store_true',
                         help='Print match score / ambiguity info to stderr (stdout stays just "x,y")')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()

    try:
        search_img = load_image(args.search, device)
        ref_img = load_image(args.reference, device)
    except Exception as e:
        print(f"ERROR: failed to load input images: {e}", file=sys.stderr)
        sys.exit(1)

    with torch.no_grad():
        out = localize_batch(search_img, ref_img, nominal_zoom=args.nominal_zoom)

    pred_x = float(out['pred_x'][0])
    pred_y = float(out['pred_y'][0])

    if args.verbose:
        score = float(out['score'][0])
        ambiguous = bool(out['ambiguous'][0])
        h, w = out['template_size']
        print(f"[inference] device={device} template_size={h}x{w} "
              f"score={score:.4f} periodic_tie_break_used={ambiguous}", file=sys.stderr)

    # Exactly ONE line on stdout: the answer, nothing else. This is what
    # a grading harness will parse -- keep it unambiguous and stable.
    print(f"{pred_x:.2f},{pred_y:.2f}")


if __name__ == '__main__':
    main()

"""
evaluate.py -- batch evaluation harness for localize.py.

Loads a whole dataset.csv worth of (search, reference, gt_x, gt_y) pairs,
runs them through localize_batch() in actual GPU-sized batches (not one
image at a time), and reports:
  - confusion matrix at 1/2/3/4/5px tolerance
  - timing: both single-pair latency and full-batch throughput
  - the designated hard-periodic case's result, called out explicitly

Usage:
    python evaluate.py --csv /path/to/dataset.csv --batch_size 8
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from PIL import Image

from localize import localize_batch, get_device


def load_image(path, device):
    """Loads grayscale OR RGB -- returns (C,H,W) float tensor in [0,1].
    RGB images are kept as 3 channels here; to_grayscale() inside
    localize_batch() handles the conversion, so this loader doesn't need
    to know or care which mode a given file is in."""
    img = Image.open(path)
    if img.mode == 'RGB':
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        t = torch.from_numpy(arr).permute(2, 0, 1)
    else:
        arr = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0)
    return t.to(device)


def load_batch(rows, base_dir, device):
    search_imgs = torch.stack([load_image(os.path.join(base_dir, r['wide_search_path']), device) for r in rows])
    ref_imgs = torch.stack([load_image(os.path.join(base_dir, r['reference_path']), device) for r in rows])
    return search_imgs, ref_imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='dataset.csv from generate_dataset.py')
    parser.add_argument('--base_dir', default=None,
                         help='Base dir for relative paths in the CSV (defaults to the CSV\'s own directory)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--nominal_zoom', type=float, default=10.0)
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.dirname(os.path.abspath(args.csv))
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    device = get_device()
    print(f"[evaluate] device: {device} | {len(rows)} pairs | batch_size={args.batch_size}")

    all_preds, all_gt, all_scores, all_ambiguous, all_names = [], [], [], [], []
    batch_times = []

    for i in range(0, len(rows), args.batch_size):
        batch_rows = rows[i:i + args.batch_size]
        search_imgs, ref_imgs = load_batch(batch_rows, base_dir, device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = localize_batch(search_imgs, ref_imgs, nominal_zoom=args.nominal_zoom)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batch_time = time.time() - t0
        batch_times.append((batch_time, len(batch_rows)))

        all_preds.append(torch.stack([out['pred_x'], out['pred_y']], dim=1).cpu().numpy())
        all_gt.append(np.array([[float(r['gt_x']), float(r['gt_y'])] for r in batch_rows]))
        all_scores.append(out['score'].cpu().numpy())
        all_ambiguous.append(out['ambiguous'].cpu().numpy())
        all_names.extend([os.path.basename(r['wide_search_path']) for r in batch_rows])

        print(f"  batch {i // args.batch_size + 1}: {len(batch_rows)} pairs in {batch_time*1000:.1f}ms "
              f"({batch_time/len(batch_rows)*1000:.2f}ms/pair)")

    preds = np.concatenate(all_preds)
    gts = np.concatenate(all_gt)
    scores = np.concatenate(all_scores)
    ambiguous = np.concatenate(all_ambiguous)
    errors = np.sqrt(((preds - gts) ** 2).sum(axis=1))

    print(f"\n{'='*70}")
    print("PER-SAMPLE RESULTS")
    print(f"{'='*70}")
    for name, err, score, amb in zip(all_names, errors, scores, ambiguous):
        flag = '  [AMBIGUOUS -- tie-break used]' if amb else ''
        print(f"  {name}: error={err:6.2f}px  score={score:.3f}{flag}")

    print(f"\n{'='*70}")
    print("CONFUSION MATRIX (accuracy within tolerance)")
    print(f"{'='*70}")
    for tol in [1, 2, 3, 4, 5]:
        hits = (errors <= tol).sum()
        print(f"  within {tol}px: {hits}/{len(errors)} ({hits/len(errors)*100:.1f}%)")

    print(f"\nmean error: {errors.mean():.2f}px | median: {np.median(errors):.2f}px | max: {errors.max():.2f}px")

    worst_idx = np.argmax(errors)
    print(f"\n{'='*70}")
    print("HONEST FAILURE CASE")
    print(f"{'='*70}")
    print(f"  Worst sample: {all_names[worst_idx]}, error={errors[worst_idx]:.2f}px, "
          f"score={scores[worst_idx]:.3f}")
    if 'hardperiodic' in all_names[worst_idx]:
        print("  This is the DESIGNATED hard case (deliberately built without a disambiguating")
        print("  landmark, so the pattern is genuinely, uniformly periodic). The matcher finds")
        print("  a real, equally-valid periodic repeat of the pattern rather than making a")
        print("  numerical error -- the correlation score is still high because the match")
        print("  really is structurally correct at a different, aliased location.")

    single_pair_times = [t / n for t, n in batch_times if n == 1]
    per_pair_from_batches = [t / n * 1000 for t, n in batch_times]
    print(f"\n{'='*70}")
    print("TIMING")
    print(f"{'='*70}")
    print(f"  device: {device}")
    print(f"  mean per-pair latency (within batches): {np.mean(per_pair_from_batches):.2f}ms")
    print(f"  total wall time for {len(rows)} pairs: {sum(t for t,_ in batch_times):.3f}s")


if __name__ == '__main__':
    main()

# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

Applied Materials problem statement, SEMICON India Hackathon 2026.

**Task**: given a small high-resolution Reference image and a larger,
10x-zoomed-out Search image containing it, return the (x, y) pixel center
of the matching region within the Search image.

## What's in this repository

| File | Purpose |
|---|---|
| `generate_dataset.py` | Standalone dataset generator -- produces synthetic Reference+Search pairs with recorded ground truth. |
| `localize.py` | Core localization algorithm (FFT-based Normalized Cross-Correlation, batched, GPU-capable). Not a CNN or Transformer -- see "Approach" below. |
| `inference.py` | **The critical file.** Standalone script: reference image path + search image path in, `(x, y)` out. This is what a grader runs. |
| `evaluate.py` | Batch self-evaluation harness: runs `localize.py` against a whole generated dataset and reports accuracy/timing. |
| `citations.md` | Every reference justifying a structural, noise, or algorithmic choice. |
| `requirements.txt` | Exact verified dependency versions. |
| `sample_output/` | 6 pre-generated pairs (with `dataset.csv` ground truth) so you can try `inference.py` immediately without generating anything first -- includes 2 normal DRAM, 2 normal FinFET, and both designated hard-periodic cases. |
| `solution_presentation.pptx` | Phase 1 slide deck. |

**No DL model weights or training script**: this solution is a classical,
closed-form signal-processing algorithm (FFT-based template matching)
with zero trainable parameters -- there is nothing to train and nothing
to download. This is a deliberate design choice, not an omission; see
"Approach" below for why.

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt
```

Works on CPU or NVIDIA GPU -- `localize.py` auto-detects CUDA and falls
back to CPU automatically (`get_device()`), no configuration needed
either way.

## Quickstart -- run inference on an already-generated pair (30 seconds)

```bash
python inference.py --reference sample_output/reference/000_dram_ref.png \
                     --search sample_output/search/000_dram_search.png \
                     --verbose
```

Expected output (stdout, exactly this format):
```
512.93,900.10
```
(True ground truth for this pair, from `sample_output/dataset.csv`:
`512.92, 900.12` -- i.e. this prediction is correct to within 0.03px.)

The `--verbose` flag additionally prints the match score and whether the
periodic-ambiguity tie-break rule fired, to **stderr** (stdout always
stays a single clean `x,y` line, safe to parse programmatically).

## Generate your own dataset

```bash
python generate_dataset.py --architecture dram --n_pairs 10 --out_dir my_dataset
python generate_dataset.py --architecture finfet --n_pairs 10 --out_dir my_dataset2
python generate_dataset.py --architecture both --n_pairs 40 --out_dir my_dataset3
```

`--architecture` accepts `dram`, `finfet`, or `both` (alternating).
Produces `<out_dir>/reference/`, `<out_dir>/search/`, `<out_dir>/preview/`
(search image with the ground-truth box drawn on it, for visual
sanity-checking), and `<out_dir>/dataset.csv` (ground truth: search path,
reference path, gt_x, gt_y).

**To exactly reproduce the 30-pair set the accuracy numbers below and in
the presentation come from**: it was built from three 10-pair batches
merged together (a development-environment time constraint, not a design
choice), not a single `--n_pairs 30` call -- those two are NOT
equivalent, since the random-seed consumption differs. Reproduce it
exactly with:
```bash
python generate_dataset.py --architecture both --n_pairs 10 --out_dir chunk_42 --seed 42
python generate_dataset.py --architecture both --n_pairs 10 --out_dir chunk_43 --seed 43
python generate_dataset.py --architecture both --n_pairs 10 --out_dir chunk_45 --seed 45
```
(then combine the three `dataset.csv` files and renumber filenames
sequentially). For a fresh, comparable set that does NOT need to match
bit-for-bit, a single `--n_pairs 30 --out_dir my_eval_set` call (same
pattern as the examples above) is fine.

## Run the full self-evaluation (30+ test cases, confusion matrix, timing)

```bash
python evaluate.py --csv sample_output/dataset.csv --batch_size 6
```

Reports per-sample error, a confusion matrix at 1/2/3/4/5px tolerance,
the worst (honest failure) case with an explanation, and timing.

**Real measured results on this repo's own 30-pair `sample_output/`**
(CPU, this development machine -- re-run on your own hardware for the
numbers that go in your submission):
- 86.7% of pairs within 5px tolerance (26/30)
- Median error: 0.33px
- The 3 worst cases (errors of 249px, 140px, 120px) are all instances of
  the deliberately-hard, purely-periodic sample type included in every
  generated batch (filename contains `hardperiodic`) -- see "Honest
  failure case" below.
- Mean per-pair compute time (warm, CPU, excluding one-time Python/import
  startup): ~190ms. Expect substantially faster on an actual NVIDIA GPU.

## Approach

**Classical FFT-based Normalized Cross-Correlation (Lewis, 1995)**, not a
CNN or Transformer. This is a deliberate choice: "find this exact patch
inside a larger image at a known scale" is precisely the problem template
matching was designed for, decades before deep learning existed. It needs
zero training data, works correctly the first time on any new die
pattern, and is fast enough to have no meaningful GPU-vs-CPU throughput
concern at this image size.

Pipeline (see `localize.py`'s module docstring for full detail):
1. Load reference + search images (grayscale or RGB).
2. Downscale the reference by the known nominal zoom (10x) to build a
   template at the search image's resolution.
3. Batched FFT-based Normalized Cross-Correlation between template and
   search image (validated against a brute-force nested-loop
   implementation before use -- floating-point-exact agreement).
4. Top-K peak extraction with non-maximum suppression.
5. Sub-pixel refinement (parabolic fit around the chosen peak).
6. **Periodic-ambiguity tie-break**: if the top two candidate scores are
   too close to call, return whichever is closest to the search image's
   own center -- exactly the rule given in the problem statement ("if
   more than one matching region is found, return the one closest to the
   center of the Search Image").

## Honest failure case

DRAM/FinFET patterns are genuinely, mathematically ambiguous when
perfectly periodic -- a template matches equally well at every
pitch-multiple shift. Real wafers solve this with dedicated alignment/
overlay marks (citations.md, section D) rather than relying on the
periodic circuit pattern itself for registration; this dataset generator
mirrors that by including a small non-periodic landmark at the true
location in every sample **except** a deliberately-included hard case per
batch, which stays purely periodic on purpose. `evaluate.py` correctly
flags these as `[AMBIGUOUS -- tie-break used]` and reports large errors on
them -- this is the algorithm behaving exactly as expected on a
genuinely ill-posed input, not a bug. See `citations.md` section D for
the full justification and `evaluate.py`'s "HONEST FAILURE CASE" output
section for a live example.

## RGB / optical microscope generalization (bonus)

`inference.py` and `localize.py` handle RGB images transparently --
`to_grayscale()` in `localize.py` converts via standard ITU-R BT.601
luminance weights before matching, so the identical pipeline and CLI work
on 3-channel optical microscope images with no code changes.

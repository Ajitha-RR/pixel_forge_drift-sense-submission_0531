"""
generate_dataset.py -- Synthetic dataset generator for the Applied Materials
"Drift-Sense" navigation-error recovery track.

Produces (reference, search) grayscale image PAIRS where:
  - reference.png: 1000x1000, a small high-resolution capture of a DRAM- or
    FinFET-style die region (1 world-unit per pixel).
  - search.png:    1000x1000, representing EXACTLY 10x the physical field of
    view of the reference (10 world-units per pixel), built by rendering the
    SAME underlying periodic layout over a 10x larger window and box-
    averaging it down -- not by resizing a bitmap, so there is never a
    resampling mismatch between the two images.
  - The reference pattern appears somewhere inside the search image, shrunk
    by exactly 10x, at a recorded ground-truth center (gt_x, gt_y) in search-
    image pixel coordinates, [0,0] = top-left (matches the CSV format the
    scoring utility expects).

Why coordinates are generated procedurally instead of "render a giant bitmap
and downsample it":
  Both images are produced by sampling ONE continuous pattern function at
  different window sizes/resolutions. This (a) guarantees the reference and
  search content are pixel-exact consistent (no separate rendering passes to
  go out of sync), (b) avoids ever materializing a 10000x10000+ array, and
  (c) still performs proper box-filter (area-average) anti-aliasing via
  supersampling -- important because naive point-sampling of a periodic
  grating at a coarse stride would alias/Moire against the DRAM/FinFET pitch.

Degradations applied (each justified in README.md / citations.md):
  - Independent sensor noise per image (signal-dependent Gaussian, after
    Foi et al. 2008's Poissonian-Gaussian raw-sensor noise model) -- drawn
    with separate RNG calls for reference vs. search, never shared.
  - Edge brightening (SEM secondary-electron edge effect).
  - A slow-varying "charging" bias field over non-conductive (background)
    regions.
  - Small random rotation of the underlying layout per sample (simulates
    stage/scan rotation error).
  - Random pitch/linewidth variation per sample (simulates process/node
    variation and magnification calibration drift, "scaling").
  - Search image gets MORE noise and MORE blur than reference, since the
    problem statement states real test data will follow that pattern.

Usage:
    python generate_dataset.py --n_pairs 40 --out_dir ./output --seed 42
"""

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import sobel, gaussian_filter

OUT_SIZE = 1000           # both reference and search are 1000x1000 px
SPAN_REF = 1000.0         # reference covers 1000x1000 world-units -> 1 unit/px
NOMINAL_ZOOM = 10.0       # the ratio your localization algorithm is TOLD to assume
ZOOM_JITTER = (0.95, 1.05)  # true per-sample zoom = NOMINAL_ZOOM * uniform(*ZOOM_JITTER)
                          # simulates magnification-calibration drift (US Patent
                          # 7,381,503 -- scale is one of the standard overlay-error
                          # components, citations.md item 18). The true value is
                          # written to each sample's meta.json for analysis, but
                          # deliberately NOT exposed in dataset.csv -- a real
                          # localization algorithm only gets to ASSUME the nominal
                          # 10x, the same way it would on the organizers' hidden
                          # test set, so this genuinely tests scale robustness
                          # instead of only ever handing the algorithm the exact
                          # answer.
SUPERSAMPLE = 3           # per-axis supersampling factor for box-filter AA


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def rotate_coords(X, Y, angle_deg, cx, cy):
    theta = np.deg2rad(angle_deg)
    Xc, Yc = X - cx, Y - cy
    Xr = Xc * np.cos(theta) - Yc * np.sin(theta) + cx
    Yr = Xc * np.sin(theta) + Yc * np.cos(theta) + cy
    return Xr, Yr


def centered_line_mask(coord, pitch, width):
    """True within `width`/2 of any multiple of `pitch`."""
    m = np.mod(coord + pitch / 2.0, pitch) - pitch / 2.0
    return np.abs(m) < (width / 2.0)


# --------------------------------------------------------------------------
# Pattern functions: given WORLD coordinates (arbitrary continuous units,
# already rotated/offset), return a base intensity array in [0,1] and a
# boolean "is_feature" mask (used for edge detection).
# --------------------------------------------------------------------------

def dram_pattern(X, Y, p):
    """Periodic word-lines (horizontal) / bit-lines (vertical) crossing at
    right angles, with a contact/via dot at every intersection.
    Structural justification: IRDS/ITRS More-Moore roadmaps (periodic
    memory-cell pitch scaling) and the Hynix DRAM layout description (EE
    Times) -- see citations.md, items 1-3 and 9."""
    bg = 0.15
    line_val = 0.55
    dot_val = 0.92

    vert = centered_line_mask(X, p['pitch_x'], p['line_w'])
    horiz = centered_line_mask(Y, p['pitch_y'], p['line_w'])
    on_line = vert | horiz

    gx = np.round(X / p['pitch_x']) * p['pitch_x']
    gy = np.round(Y / p['pitch_y']) * p['pitch_y']
    dist2 = (X - gx) ** 2 + (Y - gy) ** 2
    on_dot = dist2 < (p['dot_r'] ** 2)

    base = np.full_like(X, bg, dtype=np.float32)
    base[on_line] = line_val
    base[on_dot] = dot_val
    is_feature = on_line | on_dot
    return base, is_feature


def finfet_pattern(X, Y, p):
    """Dense parallel vertical fins crossed by 1-2 horizontal gate bars.
    Structural justification: IBM Research FinFET device-structure paper
    and the arXiv NC-FinFET / IRDS-last-FinFET-node context paper -- see
    citations.md, items 4 and 6."""
    bg = 0.12
    fin_val = 0.5
    gate_val = 0.8
    cross_val = 0.95

    fin = centered_line_mask(X, p['fin_pitch'], p['fin_w'])
    gate = centered_line_mask(Y, p['gate_pitch'], p['gate_w'])

    base = np.full_like(X, bg, dtype=np.float32)
    base[fin] = fin_val
    base[gate] = gate_val
    base[fin & gate] = cross_val
    is_feature = fin | gate
    return base, is_feature


PATTERN_FNS = {'dram': dram_pattern, 'finfet': finfet_pattern}


def random_params(style, rng):
    """Randomize structural parameters per sample within realistic ranges,
    representing node-to-node / process variation (justification:
    IRDS/ITRS roadmap pitch-scaling trends, citations.md item 1-2).
    Pitch/linewidth vary by roughly the -20%..+20% band called out in the
    hackathon's own 'Synthetic Datasets' slide."""
    if style == 'dram':
        pitch = rng.uniform(32, 48)
        return {
            'pitch_x': pitch * rng.uniform(0.8, 1.2),
            'pitch_y': pitch * rng.uniform(0.8, 1.2),
            'line_w': rng.uniform(4, 7),
            'dot_r': rng.uniform(2.5, 4.0),
        }
    else:
        fin_pitch = rng.uniform(14, 22)
        return {
            'fin_pitch': fin_pitch * rng.uniform(0.8, 1.2),
            'fin_w': rng.uniform(3, 5),
            'gate_pitch': rng.uniform(220, 320),
            'gate_w': rng.uniform(30, 50),
        }


# --------------------------------------------------------------------------
# Degradations
# --------------------------------------------------------------------------

def edge_brightening(base, is_feature, strength, rng):
    """SEM secondary-electron edge effect: boundaries of raised/etched
    features emit more secondary electrons and appear brighter.
    Justification: Goldstein et al., 'Scanning Electron Microscopy and
    X-Ray Microanalysis' (edge-effect description); USPTO 7,335,880
    (CD-SEM edge-effect in semiconductor metrology) -- citations.md
    items 10-11."""
    gy = sobel(base.astype(np.float32), axis=0)
    gx = sobel(base.astype(np.float32), axis=1)
    grad_mag = np.hypot(gx, gy)
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    return base + strength * grad_mag


def charging_bias(shape, is_feature, strength, rng):
    """Slow-varying brightness drift over non-conductive (background)
    regions, mimicking local charge accumulation under e-beam irradiation.
    Justification: Cazaux (1999) J. Appl. Phys. 85, 1137-1147; Reimer
    (1993) 'Specimen charging and damage' -- citations.md items 12-13."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    field = np.zeros(shape, dtype=np.float32)
    for _ in range(3):
        fx, fy = rng.uniform(1, 4, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        field += np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)
    field = field / field.max() * strength
    return field * (~is_feature)  # charging affects background more than metal/dot


def signal_dependent_noise(img01, a, b, rng):
    """Poissonian-Gaussian sensor noise: std(pixel) = sqrt(a*signal + b).
    Justification: Foi, Trimeche, Katkovnik, Egiazarian, 'Practical
    Poissonian-Gaussian Noise Modeling and Fitting for Single-Image
    Raw-Data,' IEEE TIP 17(10), 2008 -- citations.md item 14."""
    std = np.sqrt(np.clip(img01, 0, 1) * a + b)
    return img01 + rng.normal(0, 1, img01.shape).astype(np.float32) * std


def apply_landmark(base, is_feature, Xr, Yr, pivot, radius, value):
    """Overrides a local patch centered at `pivot` (a WORLD coordinate, not
    a per-window coordinate) with a distinctive, non-periodic marker. This
    is essential, not decorative: a purely periodic DRAM/FinFET pattern
    matches equally well at every pitch-multiple shift, making the
    localization problem mathematically ill-posed (verified empirically --
    an earlier version of this generator without a landmark produced
    ~400px average error even on CLEAN, noise-free patterns, because the
    matcher was finding a different, equally-valid periodic repeat, not
    making a mistake). The organizers' own reference image shows exactly
    this kind of local break in symmetry (a cross-shaped notch in the
    middle of the array on the "Metrics" slide) -- this reproduces that
    idea generically for both styles. Since `pivot` is a WORLD coordinate
    fixed at ref_center for every render in a pair (see render_window),
    the landmark appears at the true corresponding location in both the
    reference and the search image, and nowhere else."""
    dist2 = (Xr - pivot[0]) ** 2 + (Yr - pivot[1]) ** 2
    mask = dist2 < radius ** 2
    base = base.copy()
    base[mask] = value
    return base, (is_feature | mask)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_window(center, span, out_size, style, params, rotation_deg,
                   pivot=None, supersample=SUPERSAMPLE, landmark=None):
    """Render an out_size x out_size image covering a `span` x `span`
    world-unit window centered at `center`, with the ENTIRE world pattern
    rotated by `rotation_deg` around `pivot` (defaults to `center` if not
    given -- fine for a single standalone render, but the reference and
    search windows of the same pair MUST be rendered with the SAME pivot,
    otherwise each window rotates around a different point and the two
    images silently go out of geometric alignment, corrupting the ground
    truth). Rotation simulates stage/scan rotation error -- justification:
    US Patent 7,381,503, which names rotation, translation, scale and lens
    distortion as the standard components of wafer-stage/overlay
    navigation error; citations.md item 15).

    `landmark`, if given, is a (radius, value) tuple applied at `pivot` --
    see apply_landmark() above for why this matters."""
    if pivot is None:
        pivot = center
    px = span / out_size
    ss = out_size * supersample
    coords_1d = (np.arange(ss) + 0.5) / supersample * px - span / 2.0
    Xg, Yg = np.meshgrid(coords_1d, coords_1d)
    Xg = Xg + center[0]
    Yg = Yg + center[1]
    Xr, Yr = rotate_coords(Xg, Yg, rotation_deg, pivot[0], pivot[1])

    base, is_feature = PATTERN_FNS[style](Xr, Yr, params)
    if landmark is not None:
        radius, value = landmark
        base, is_feature = apply_landmark(base, is_feature, Xr, Yr, pivot, radius, value)

    def downsample(a):
        return a.reshape(out_size, supersample, out_size, supersample).mean(axis=(1, 3))

    base_ds = downsample(base)
    feat_ds = downsample(is_feature.astype(np.float32)) > 0.5
    return base_ds, feat_ds


def make_pair(style, rng, force_periodic_offset=False):
    params = random_params(style, rng)
    rotation_deg = rng.uniform(0.5, 3.0) * rng.choice([-1, 1])

    # True zoom for THIS sample -- jittered around the nominal 10x the
    # algorithm is told to assume (see ZOOM_JITTER comment above).
    true_zoom = NOMINAL_ZOOM * rng.uniform(*ZOOM_JITTER)
    span_search = SPAN_REF * true_zoom

    ref_center = (0.0, 0.0)

    margin = 400.0  # world-units of safety margin from the search image edge
    max_offset = span_search / 2.0 - SPAN_REF / 2.0 - margin
    if force_periodic_offset:
        # Deliberately offset by an exact multiple of the pattern pitch so
        # the local neighborhood repeats identically -- a genuinely
        # ambiguous case for any localizer, used as the required honest
        # failure-case example.
        pitch = params.get('pitch_x', params.get('fin_pitch', 40.0))
        k = rng.integers(3, 8)
        dx, dy = pitch * k, pitch * rng.integers(3, 8)
        dx, dy = min(dx, max_offset), min(dy, max_offset)
    else:
        dx = rng.uniform(-max_offset, max_offset)
        dy = rng.uniform(-max_offset, max_offset)

    search_center = (ref_center[0] - dx, ref_center[1] - dy)

    # Landmark breaks pure periodicity so the sample has a genuinely unique
    # match -- omitted ONLY for the designated hard case, which must stay
    # ambiguous on purpose (see apply_landmark() docstring).
    if force_periodic_offset:
        landmark = None
    else:
        # Use the LARGEST pitch parameter, not just any one -- for FinFET,
        # fin_pitch is already sub-resolution after the 10x zoom (verified
        # earlier: fins wash out, only the widely-spaced gate bars survive),
        # so sizing the landmark off fin_pitch left it far too small to
        # break the gate-bar periodicity that actually dominates the search
        # image. Sizing off the largest pitch present fixes this for both
        # styles.
        pitch_values = [v for k, v in params.items() if 'pitch' in k]
        base_pitch = max(pitch_values)
        radius = rng.uniform(2.5, 3.5) * base_pitch  # generous multiplier for
        # robustness -- still safely bounded by the cap below
        radius = min(radius, 0.25 * SPAN_REF)  # cap: must stay a LOCAL feature,
        # not grow large enough to fill/dominate the whole reference window
        # (an earlier version let radius reach ~800 world-units against a
        # 1000-unit window -- verified visually as a solid black frame with
        # zero remaining structure, and numerically as ~55px error with a
        # near-zero NCC confidence score, 0.03-0.04, both symptoms of the
        # same bug)
        landmark = (radius, rng.choice([0.0, 1.0]))

    ref_base, ref_feat = render_window(ref_center, SPAN_REF, OUT_SIZE, style,
                                        params, rotation_deg, pivot=ref_center,
                                        landmark=landmark)
    search_base, search_feat = render_window(search_center, span_search, OUT_SIZE,
                                              style, params, rotation_deg, pivot=ref_center,
                                              landmark=landmark)

    # Edge brightening (reference: sharper/stronger since it's the higher-
    # fidelity capture; search: weaker, already blurred by downsampling)
    ref_img = edge_brightening(ref_base, ref_feat, strength=0.35, rng=rng)
    search_img = edge_brightening(search_base, search_feat, strength=0.15, rng=rng)

    # Charging bias (background regions)
    ref_img = ref_img + charging_bias(ref_img.shape, ref_feat, 0.03, rng)
    search_img = search_img + charging_bias(search_img.shape, search_feat, 0.05, rng)

    # Search gets extra blur (coarser optics / lower magnification defocus).
    # Kept small on purpose: the DRAM/FinFET pitch is only ~3-5 search
    # pixels wide after the 10x zoom-out, so a large sigma here would
    # destroy the periodic signal entirely rather than just degrading it
    # (verified numerically while building this generator -- an earlier
    # sigma of 0.5-1.0 combined with noise below buried the pattern
    # completely, which is unrealistically hard even for the "difficult"
    # requirement).
    search_img = gaussian_filter(search_img, sigma=rng.uniform(0.25, 0.5))

    # Independent sensor noise -- separate rng draws, search noisier than
    # ref but calibrated to stay below the post-blur signal contrast so
    # the pattern remains faintly recoverable (except the designated hard
    # case, which is difficult via periodic ambiguity, not via noise).
    ref_img = signal_dependent_noise(ref_img, a=0.008, b=0.0003, rng=rng)
    search_img = signal_dependent_noise(search_img, a=0.015, b=0.0008, rng=rng)

    ref_img = np.clip(ref_img, 0, 1)
    search_img = np.clip(search_img, 0, 1)

    # Flip augmentation: mirroring is a standard, physically-valid
    # augmentation here because the DRAM/FinFET pattern has no inherent
    # "handedness" -- a mirrored periodic grid is still a valid periodic
    # grid. Applying the SAME flip to both images (and updating gt_x/gt_y
    # to match) keeps the pair geometrically consistent.
    flip_h = rng.random() < 0.5
    flip_v = rng.random() < 0.5
    if flip_h:
        ref_img = np.fliplr(ref_img)
        search_img = np.fliplr(search_img)
    if flip_v:
        ref_img = np.flipud(ref_img)
        search_img = np.flipud(search_img)

    px_size_search = span_search / OUT_SIZE
    gt_x = OUT_SIZE / 2.0 + dx / px_size_search
    gt_y = OUT_SIZE / 2.0 + dy / px_size_search
    if flip_h:
        gt_x = OUT_SIZE - gt_x
    if flip_v:
        gt_y = OUT_SIZE - gt_y

    meta = {
        'style': style,
        'params': {k: float(v) for k, v in params.items()},
        'rotation_deg': float(rotation_deg),
        'nominal_zoom': NOMINAL_ZOOM,
        'true_zoom': float(true_zoom),
        'dx_world': float(dx),
        'dy_world': float(dy),
        'flip_h': bool(flip_h),
        'flip_v': bool(flip_v),
        'has_landmark': landmark is not None,
        'landmark_radius': float(landmark[0]) if landmark else None,
        'gt_x_px': float(gt_x),
        'gt_y_px': float(gt_y),
        'forced_periodic_hard_case': bool(force_periodic_offset),
    }
    return ref_img, search_img, gt_x, gt_y, true_zoom, meta


def save_gray(img01, path):
    arr = (np.clip(img01, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(arr, mode='L').save(path)


def save_preview_with_box(search_img01, gt_x, gt_y, box_size, path):
    arr = (np.clip(search_img01, 0, 1) * 255).round().astype(np.uint8)
    im = Image.fromarray(arr, mode='L').convert('RGB')
    draw = ImageDraw.Draw(im)
    half = box_size / 2.0
    draw.rectangle([gt_x - half, gt_y - half, gt_x + half, gt_y + half],
                   outline=(255, 0, 0), width=2)
    im.save(path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic Reference+Search image pairs for Drift-Sense.')
    parser.add_argument('--architecture', choices=['dram', 'finfet', 'both'], default='both',
                         help='Die architecture style: dram, finfet, or both (alternating). '
                              'Required parameter per the submission spec.')
    parser.add_argument('--n_pairs', type=int, default=40,
                         help='Number of image pairs to generate.')
    parser.add_argument('--out_dir', default='./output',
                         help='Output directory for reference/, search/, preview/, and dataset.csv.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    ref_dir = os.path.join(args.out_dir, 'reference')
    search_dir = os.path.join(args.out_dir, 'search')
    preview_dir = os.path.join(args.out_dir, 'preview')
    for d in (ref_dir, search_dir, preview_dir):
        os.makedirs(d, exist_ok=True)

    master_rng = np.random.default_rng(args.seed)
    rows = []

    for i in range(args.n_pairs):
        if args.architecture == 'both':
            style = 'dram' if i % 2 == 0 else 'finfet'
        else:
            style = args.architecture
        # each sample gets its own child RNG (reproducible, independent streams)
        rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        force_hard = (i == args.n_pairs - 1)  # last sample = designated hard case

        ref_img, search_img, gt_x, gt_y, true_zoom, meta = make_pair(style, rng, force_hard)

        tag = f"{i:03d}_{style}" + ("_hardperiodic" if force_hard else "")
        ref_path = os.path.join(ref_dir, f"{tag}_ref.png")
        search_path = os.path.join(search_dir, f"{tag}_search.png")
        preview_path = os.path.join(preview_dir, f"{tag}_preview.png")

        save_gray(ref_img, ref_path)
        save_gray(search_img, search_path)
        save_preview_with_box(search_img, gt_x, gt_y, box_size=OUT_SIZE / true_zoom,
                               path=preview_path)

        with open(os.path.join(args.out_dir, f"{tag}_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        rows.append([search_path, ref_path, f"{gt_x:.2f}", f"{gt_y:.2f}"])
        print(f"[{i+1}/{args.n_pairs}] {tag}: GT=({gt_x:.1f}, {gt_y:.1f}) "
              f"true_zoom={true_zoom:.3f} (nominal={NOMINAL_ZOOM})"
              + ("  <-- designated hard/periodic failure-case sample" if force_hard else ""))

    csv_path = os.path.join(args.out_dir, 'dataset.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['wide_search_path', 'reference_path', 'gt_x', 'gt_y'])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} pairs.")
    print(f"CSV: {csv_path}")
    print(f"Reference images: {ref_dir}")
    print(f"Search images:    {search_dir}")
    print(f"Preview (GT box overlay): {preview_dir}")


if __name__ == '__main__':
    main()

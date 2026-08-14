# Citations

Every citation below was checked against the source before inclusion, per
the submission requirement to justify every augmentation/noise choice.
Re-verify before final submission. Organized by what each citation
justifies, referencing the specific function/design choice in the code.

## A. Die structure (DRAM / FinFET geometry)

Reused from the hackathon's own "Some Citations to Generate Layouts"
slide, provided specifically "for the in-app citations panel":

1. IRDS 2017 More Moore roadmap -- https://irds.ieee.org/images/files/pdf/2017/2017IRDS_MM.pdf
2. IRDS 2024 More Moore roadmap -- https://irds.ieee.org/images/files/pdf/2024/2024IRDS_MM.pdf
3. ITRS 2015 More Moore roadmap -- https://www.semiconductors.org/wp-content/uploads/2018/06/5_2015-ITRS-2.0-More-Moore.pdf
4. IBM Research, "Opportunities and Challenges of FinFET as a Device Structure Candidate for 14nm Node CMOS Technology" -- https://research.ibm.com/publications/opportunities-and-challenges-of-finfet-as-a-device-structure-candidate-for-14nm-node-cmos-technology
5. Semiconductor Engineering, "7nm Fab Challenges" -- https://semiengineering.com/7nm-fab-challenges/
6. arXiv 2007.14448 (NC-FinFET / IRDS last-FinFET-node context) -- https://arxiv.org/pdf/2007.14448
7. FreePDK15 predictive PDK paper -- https://arxiv.org/pdf/2009.04600
8. TI patent EP0780901A2 (arcuate moats / wavy bitlines) -- https://patents.google.com/patent/EP0780901A2/en
9. EE Times, "Hynix DRAM layout, process integration adapt to change" -- https://www.eetimes.com/hynix-dram-layout-process-integration-adapt-to-change/
10. US Patent 5,554,874 (6T SRAM cell) -- https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5554874
11. US Patent 6,938,226 (7-tracks standard cell library) -- https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6938226
12. imec, "View on logic technology roadmap" -- https://www.imec-int.com/en/articles/view-logic-technology-roadmap

## B. SEM imaging physics (noise, edge brightening, charging)

13. Goodman, J.W. (2007), *Speckle Phenomena in Optics: Theory and
    Applications*, Ch. 2 -- multiplicative sensor noise model.
14. Foi, A., Trimeche, M., Katkovnik, V., Egiazarian, K. (2008),
    "Practical Poissonian-Gaussian Noise Modeling and Fitting for
    Single-Image Raw-Data," IEEE TIP 17(10) -- signal-dependent noise.
15. Goldstein, J.I. et al., *Scanning Electron Microscopy and X-Ray
    Microanalysis*, Plenum Press -- SEM edge-brightening effect
    (`edge_brightening()`).
16. US Patent 7,335,880, "Technique for CD measurement on the basis of
    area fraction determination" -- edge effect specifically in
    semiconductor CD-SEM metrology.
    https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7335880
17. Cazaux, J. (1999), "Some considerations on the secondary electron
    emission from e-irradiated insulators," J. Appl. Phys. 85, 1137-1147
    -- charging bias model (`charging_bias()`).
18. Reimer, L. (1993), "Specimen charging and damage," in Image Formation
    in Low Voltage SEM, Ch. 5, SPIE Press.

## C. Navigation-error components (rotation, scale/zoom drift)

19. US Patent 7,381,503, "Reference wafer calibration reticle" -- names
    translation, rotation, scale/magnification and lens distortion as the
    standard components of wafer-stage/overlay navigation error, directly
    motivating why the dataset generator applies per-sample rotation and
    zoom-ratio jitter (true zoom 9.5x-10.5x vs. a 10x nominal the
    algorithm is told to assume), and why the localization algorithm
    deliberately does NOT assume the zoom ratio is exact.
    https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7381503

## D. Why the dataset needs a disambiguating landmark

A purely, infinitely periodic DRAM/FinFET pattern is mathematically
ill-posed for localization -- a template matches equally well at every
pitch-multiple shift (verified empirically while building this project:
~400px average error even on clean, noise-free patterns, because the
matcher was finding a different, equally-valid periodic repeat, not
making an error). Real wafers solve this with dedicated alignment/overlay
marks precisely because periodic device arrays alone cannot be reliably
registered:

20. US Patent 7,939,224, "Mask with registration marks and method of
    fabricating integrated circuits" -- describes alignment marks (e.g.
    box-in-box) placed specifically to give layers a uniquely identifiable
    registration point, distinct from the surrounding periodic device
    pattern. This directly motivates `apply_landmark()`: every generated
    sample except the deliberately-hard designated case includes a
    non-periodic local marker at the true reference location, exactly
    mirroring why real wafers carry dedicated alignment marks rather than
    relying on the periodic circuit pattern itself for registration.
    https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7939224
21. US Patent 8,143,731, "Integrated alignment and overlay mark" --
    further confirms alignment/overlay marks as standard practice
    distinguishing a unique registration site from the surrounding
    periodic circuit structure.

## E. Localization algorithm (classical, non-CNN/Transformer)

22. Lewis, J.P. (1995), "Fast Normalized Cross-Correlation," Vision
    Interface, pp. 120-123 -- the FFT-accelerated NCC formula implemented
    in `batched_ncc()` (localize.py). Validated against a brute-force
    nested-loop reference implementation before use (max abs difference
    ~3e-15, floating-point exact).
23. Guizar-Sicairos, M., Thurman, S.T., Fienup, J.R. (2008), "Efficient
    Subpixel Image Registration Algorithms," Optics Letters, 33(2),
    156-158 -- the more accurate (FFT-upsampling) alternative to the
    parabolic-fit sub-pixel refinement currently used in
    `subpixel_refine()`; noted as a possible upgrade if tighter-than-1px
    accuracy is ever needed.
24. ITU-R BT.601 luminance weights (0.299, 0.587, 0.114) -- standard
    RGB-to-grayscale conversion (same weights PIL's `Image.convert('L')`
    uses), used in `to_grayscale()` for the RGB/optical-microscope
    generalization.

## F. Why box-filter (area-average) downsampling, not point-sampling

Both the dataset generator's 10x zoom-out and the localization
algorithm's reference-to-template downscale use area-average
resampling (`Image.BOX` / `F.interpolate(..., mode='area')`), matching
how a real lower-magnification detector integrates signal over a larger
physical area per pixel. Naive point-sampling of a periodic grating at a
coarse stride aliases badly against the pattern pitch (Moire artifacts) --
avoided here by construction, not by luck.

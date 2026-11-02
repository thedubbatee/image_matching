Run the comparison with the **full-image sequences**, not the extracted patch
mosaics. From the `feature_detector` directory, create an environment containing one OpenCV wheel:

```bash
python -m venv .venv-verify
.venv-verify/bin/python -m pip install numpy scipy opencv-contrib-python
curl -L --fail -o /tmp/hpatches-sequences-release.zip https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip
unzip /tmp/hpatches-sequences-release.zip -d /tmp/hpatches
.venv-verify/bin/python verification/verify_detectors.py /tmp/hpatches/hpatches-sequences-release
```

The archive is 1,279,545,572 bytes (about 1.28 GB). The complete archive was
downloaded and extracted into `verification/.hpatches-cache/hpatches-sequences-release` for
the full evaluation. See `DETECTOR_RESULTS.md` for the measured results.

Use `--limit 6` for a short run or `--detector dog` to run with standard OpenCV.
The default runs both detectors on all six images in every discovered sequence,
at original resolution. Incomplete sequences and unreadable images are errors.
The JSON report records image counts so a subset run can be identified.

The custom DoG implementation is compared with OpenCV SIFT's DoG detector;
Harris–Laplace is compared with `cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create`.
Custom detector defaults are used. SIFT receives the corresponding custom scale,
contrast and edge parameters; Harris–Laplace uses OpenCV defaults, including
its 5,000-point reference cap. The custom detector keeps at most 500 points.
The implementations have different scale
sampling, threshold semantics and refinement steps, so a score below 90% is a
diagnostic of disagreement, not by itself proof of an implementation error.

For each custom feature, choose the nearest OpenCV feature by position, allowing
reference reuse. Test distance **<= 3 pixels** and scale ratio
**0.8 <= custom_sigma / reference_sigma <= 1.25**. The closest feature is tested
even if a slightly farther feature has a better scale. Exact position ties use
the KD-tree's choice. SIFT sigma is `KeyPoint.size / 2`; OpenCV Harris–Laplace's
integration sigma is `KeyPoint.size / 6`, as established from its source below.
Orientation is ignored. Per-detector accuracy is total correct features divided
by total custom features across images; **>= 90%** passes. Empty output fails.
This measures agreement/precision; it does not measure missed reference features,
one-to-one correspondence, or repeatability under homographies.

`verification/detector_verification.json` contains summaries and, for every custom feature in
detector output order, reference index, distance, scale ratio and correctness.
Exit statuses: 0 = all selected detectors pass, 1 = comparison fails, 2 = setup,
input or runtime error. Run the comparison tests with:

```bash
python -m unittest discover -v
```

Sources:

- [Official HPatches dataset and full sequence downloads](https://github.com/hpatches/hpatches-dataset)
- [Official HPatches descriptor benchmark](https://github.com/hpatches/hpatches-benchmark)
- [OpenCV Harris–Laplace implementation and scale formula](https://github.com/opencv/opencv_contrib/blob/4.x/modules/xfeatures2d/src/harris_lapace_detector.cpp)

The official benchmark evaluates **descriptors** through patch verification,
matching and retrieval. This script implements the requested detector agreement
check separately; it does not invoke or reproduce those official benchmark tasks.
The full-image homographies are unnecessary for this same-image comparison.

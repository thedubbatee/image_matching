Full HPatches detector agreement evaluation, 2026-11-02

Both detectors **failed the requested 90% agreement criterion in the original
run**, which capped the OpenCV Harris–Laplace reference at 500 points. The
comparison script now uses OpenCV's 5,000-point default; the historical figures
below have not yet been recomputed with that setting.

| Detector | Correct features | Total custom features | Agreement | Result |
|---|---:|---:|---:|---|
| DoG | 286,438 | 379,430 | 75.4917% | FAIL |
| Harris–Laplace | 90,166 | 264,471 | 34.0930% | FAIL |

All 116 sequences / 696 images were evaluated at original resolution. A custom
feature passes when its spatially nearest OpenCV feature is within 3 pixels and
the custom/reference sigma ratio is in [0.8, 1.25], inclusive. Reference reuse is
allowed. The detector defaults and evaluation thresholds were unchanged.

| Sequence type | Images | DoG agreement | Harris–Laplace agreement |
|---|---:|---:|---:|
| Illumination (`i_`) | 342 | 75.2336% | 37.7938% |
| Viewpoint (`v_`) | 354 | 75.6162% | 31.5184% |

Position alone passes for 81.67% of DoG features and 44.66% of Harris–Laplace
features. Scale alone passes for 78.50% and 45.94%, respectively. Thus neither
failure is explained solely by the scale check.

Environment: Python 3.14, OpenCV contrib 5.0.0 (wheel 5.0.0.93), NumPy 2.5.4,
SciPy 1.18.1, installed in `/tmp/hpatches-verify-env`. The full evaluation used
four worker processes with one OpenCV thread each and the functions in
`verification/verify_detectors.py`. Runtime was approximately 11.5 minutes after download and
extraction. Both reference detector factories were available. No image was
skipped, resized, or sampled out.

`verification/detector_verification.json` contains all per-feature measurements (about 59 MB).
Its totals were independently recomputed from all 696 unique image records for
each detector. The five comparison unit tests passed in the evaluation
environment. The downloaded dataset and execution log are retained in the
ignored `verification/.hpatches-cache` directory.

To repeat with the standalone script (sequential execution):

```bash
/tmp/hpatches-verify-env/bin/python verification/verify_detectors.py verification/.hpatches-cache/hpatches-sequences-release --output verification/detector_verification.json
```

These results do not confirm implementation correctness under the proposed
criterion. They demonstrate disagreement with the OpenCV references; isolating
bugs from differences in scale sampling, thresholds and refinement requires
further investigation. The official HPatches descriptor benchmark was not run,
because it does not implement this detector agreement test.

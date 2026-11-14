# image_matching
## Benchmarking

There are two evaluations, both on the [HPatches](https://github.com/hpatches/hpatches-dataset) dataset:

| Evaluation | What it tests | Data | Script |
|---|---|---|---|
| **A. Pipeline evaluation** | Full pipelines end to end: Harris-Laplace + SIFT vs DoG + SIFT (vs OpenCV SIFT) | HPatches **image sequences** (696 images, 1.3 GB) | `eval_pipeline.py` |
| **B. Descriptor benchmark** | Our SIFT descriptor alone, on the official HPatches verification / matching / retrieval tasks | HPatches **patches** (4.2 GB) | `hpatches_extract.py` + the official `hpatches-benchmark` toolkit |

The datasets and the toolkit are **not** in this repository. Download them next to it, as below.

### Folder layout

The commands assume this layout (any other location works if you change the paths):

```
~/COMP3821/
├── image_matching/                  # this repository
│   └── src/                         # sift.py, harris.py, dog.py, eval_pipeline.py, ...
├── hpatches-benchmark/              # official toolkit (evaluation B)
│   └── data/
│       ├── hpatches-release/        # patch dataset (evaluation B)
│       └── descriptors/             # descriptor CSVs written by our extractors
└── data/
    └── hpatches-sequences-release/  # image sequences (evaluation A)
```

### Requirements

Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy opencv-python matplotlib pandas joblib dill tqdm tabulate
```

OpenCV must be 4.4 or newer (it needs `cv2.SIFT_create`).

### Files

| File | Purpose |
|---|---|
| `sift.py` | Our SIFT descriptor (orientation assignment and 128-D descriptor) |
| `harris.py`, `dog.py` | Harris-Laplace and Difference-of-Gaussians detectors |
| `detector_helpers.py` | Shared `Keypoint` type and greyscale conversion |
| `matcher.py` | Brute-force L2 matcher with cross-checking |
| `test_comp.py` | Quick check that detectors, SIFT and matcher work together |
| `eval_pipeline.py` | Evaluation A |
| `hpatches_extract.py` | Evaluation B: runs our SIFT on every HPatches patch |
| `hpatches_extract_opencv.py` | Evaluation B: the same with OpenCV SIFT (baseline) |

Before benchmarking, check everything works together:

```bash
cd src
python3 test_comp.py
```

All checks should pass.

---

### A. Pipeline evaluation (Harris-Laplace vs DoG)

**1. Download the image sequences (about 1.3 GB):**

```bash
mkdir -p ~/COMP3821/data && cd ~/COMP3821/data
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xzf hpatches-sequences-release.tar.gz && rm hpatches-sequences-release.tar.gz
ls hpatches-sequences-release | wc -l    # should print 116
```

There are 57 illumination sequences (`i_*`: same viewpoint, different lighting) and 59 viewpoint sequences (`v_*`: planar scenes from different angles). Each folder has `1.ppm` to `6.ppm` and the ground-truth homographies `H_1_2` to `H_1_6`.

**2. Quick test on 4 illumination and 4 viewpoint sequences:**

```bash
cd ~/COMP3821/image_matching/src
time python3 eval_pipeline.py ~/COMP3821/data/hpatches-sequences-release results/pipeline --limit 4
```

Multiply the time by about 15 to estimate the full run.

**3. Full run:**

```bash
python3 eval_pipeline.py ~/COMP3821/data/hpatches-sequences-release results/pipeline
```

Check the **Warning** line under Table 1 in the output. If the pipelines kept different numbers of keypoints, re-run with the `--num-kps` value it suggests, so that no pipeline wins by having more points:

```bash
python3 eval_pipeline.py ~/COMP3821/data/hpatches-sequences-release results/pipeline --num-kps 300
```

**What it does:** every image is downscaled to at most 640 px on its longest side (`--max-side`), with the homographies rescaled to match. Each detector keeps its `--num-kps` strongest keypoints (default 500). Everything else is identical for every pipeline: same SIFT (`sift.py`), same matcher (`matcher.py`, cross-checked). It reports:

- **Repeatability**: % of keypoints found again within 3 px in the other image
- **MMA@t**: % of matches within *t* px of the true position (t = 1 to 10)
- **Correct matches** per image pair, at 3 px
- **Homography accuracy**: % of pairs where the RANSAC-estimated homography maps the image corners within 1 / 3 / 5 px of the truth
- **Runtime** per image

Results are reported for all pairs, illumination only and viewpoint only, and per image pair (1→2 … 1→6).

**Outputs** in `results/pipeline/` (the `TAG` is e.g. `n500_s640`, from the settings used):

| File | Contents |
|---|---|
| `summary_TAG.md` | Result tables |
| `pairs_TAG.csv` | Every metric for every pipeline and image pair |
| `mma_TAG.png` | MMA vs pixel threshold, illumination and viewpoint |
| `mma_by_pair_TAG.png` | MMA@3px vs image pair |
| `cache/` | Per-image keypoints and descriptors; re-runs with the same settings reuse them |

Other options: `--methods harris dog` (skip the OpenCV baseline), `--max-side 0` (full resolution, much slower), `--workers N`. See `python3 eval_pipeline.py --help`.

---

### B. Descriptor benchmark (official HPatches toolkit)

This evaluates the descriptor on its own, on patches that are already detected, scaled and oriented.

**1. Clone the toolkit next to this repository and fix it for current Python:**

The toolkit was written for older pandas/NumPy and crashes without these three fixes:

```bash
cd ~/COMP3821
git clone https://github.com/hpatches/hpatches-benchmark.git
cd hpatches-benchmark
sed -i 's/\.as_matrix()/.to_numpy()/g' python/utils/tasks.py
sed -i 's/np\.trapz(/getattr(np, "trapezoid", getattr(np, "trapz", None))(/g' python/utils/metrics.py
sed -i "s/D\[i-1\] = dist_matrix(d,d_,descr\['distance'\])\[0\]/D[i-1] = dist_matrix(d,d_,descr['distance'])[0, 0]/" python/utils/tasks.py
git diff --stat    # should list python/utils/metrics.py and python/utils/tasks.py
```

**2. Download the patch dataset (about 4.2 GB).** The toolkit's `download.sh` is broken, so download it by hand:

```bash
cd ~/COMP3821/hpatches-benchmark
wget -O data/hpatches-release.zip https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-release.zip
unzip -q data/hpatches-release.zip -d data && rm data/hpatches-release.zip
ls data/hpatches-release | wc -l    # should print 116
```

**3. Extract descriptors.** Each patch is described as one keypoint at the patch centre with angle 0 (the patches are already oriented). Run from `src/`:

```bash
cd ~/COMP3821/image_matching/src
H=~/COMP3821/hpatches-benchmark/data/hpatches-release
D=~/COMP3821/hpatches-benchmark/data/descriptors

python3 hpatches_extract.py $H $D/our-sift
python3 hpatches_extract.py $H $D/our-sift-s613         --sigma 6.13
python3 hpatches_extract.py $H $D/our-sift-cvlike       --sigma 6.13 --blur 1.6
python3 hpatches_extract.py $H $D/our-sift-cvlike-root  --sigma 6.13 --blur 1.6 --rootsift
python3 hpatches_extract_opencv.py $H $D/opencv-sift
python3 hpatches_extract_opencv.py $H $D/opencv-rootsift --rootsift
```

Each command checks itself against `sift.py` first (look for `-> OK`), writes one CSV per patch stack, and can be re-run to resume if interrupted.

| Name | Settings | Purpose |
|---|---|---|
| `our-sift` | σ = 65/12 ≈ 5.42, blur = σ (standard SIFT) | default |
| `our-sift-s613` | σ = 6.13, blur = σ | effect of region size |
| `our-sift-cvlike` | σ = 6.13, blur = 1.6 | same settings as OpenCV in this benchmark |
| `our-sift-cvlike-root` | as above + RootSIFT | effect of RootSIFT |
| `opencv-sift` | OpenCV SIFT, centre keypoint as in the toolkit's own example | baseline |
| `opencv-rootsift` | OpenCV SIFT + RootSIFT | baseline |

σ sets the size of the described region (the 4×4 grid is 12σ wide). Blur is the Gaussian smoothing of the image the gradients are measured on. RootSIFT L1-normalises each descriptor and takes the square root of every value.

**4. Evaluate and print the results:**

```bash
cd ~/COMP3821/hpatches-benchmark/python
for d in our-sift our-sift-s613 our-sift-cvlike our-sift-cvlike-root opencv-sift opencv-rootsift; do
  python3 hpatches_eval.py --descr-name=$d --task=verification --task=matching --task=retrieval
done

python3 hpatches_results.py --descr-name=our-sift --descr-name=our-sift-s613 \
  --descr-name=our-sift-cvlike --descr-name=opencv-sift \
  --descr-name=our-sift-cvlike-root --descr-name=opencv-rootsift \
  --task=verification --task=matching --task=retrieval
```

This uses split `a` (40 test sequences). Results are cached in `hpatches-benchmark/python/results/`. If you re-extract a descriptor under the same name, delete `results/<name>_*` first.

**Expected results** (split `a`, mAP %; verification = mean imbalanced AP, retrieval at 20,000 distractors):

| Descriptor | Verification | Matching | Retrieval |
|---|---|---|---|
| `our-sift` | 66.4 | 16.9 | 22.6 |
| `our-sift-s613` | 64.8 | 15.1 | 20.9 |
| `our-sift-cvlike` | 67.0 | 25.5 | 32.4 |
| `opencv-sift` | 66.9 | 25.3 | 32.2 |
| `our-sift-cvlike-root` | 62.2 | 30.1 | 35.7 |
| `opencv-rootsift` | 61.8 | 30.0 | 35.5 |

With the same settings, our SIFT and OpenCV's differ by less than 0.5 points.

Do not commit `results/`, `data/` or the datasets; add them to `.gitignore`.
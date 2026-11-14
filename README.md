## Setup for testing and evaluation

The repository has two separate kinds of evaluation, both on the [HPatches](https://github.com/hpatches/hpatches-dataset) dataset:

| Section | Purpose | Data | Scripts |
|---|---|---|---|
| [SIFT descriptor: correctness and benchmarking](#sift-descriptor-correctness-and-benchmarking) | Check that our SIFT implementation is correct, and benchmark it against OpenCV's SIFT | HPatches **patches** (4.2 GB) | `test_comp.py`, `hpatches_extract.py`, `hpatches_extract_opencv.py` + the official `hpatches-benchmark` toolkit |
| [Pipeline comparison: Harris-Laplace vs DoG](#pipeline-comparison-harris-laplace-vs-dog) | Compare the two full pipelines (detector + SIFT + matcher) directly | HPatches **image sequences** (696 images, 1.3 GB) | `eval_pipeline.py` |

The datasets and the toolkit are **not** in this repository. Download them next to it, as described in each section.

### Folder layout

The commands assume this layout (any other location works if you change the paths):

```
~/COMP3821/
├── image_matching/                  # this repository
│   └── src/                         # sift.py, harris.py, dog.py, eval_pipeline.py, ...
├── hpatches-benchmark/              # official toolkit (SIFT benchmarking)
│   └── data/
│       ├── hpatches-release/        # patch dataset
│       └── descriptors/             # descriptor CSVs written by our extractors
└── data/
    └── hpatches-sequences-release/  # image sequences (pipeline comparison)
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
| `test_comp.py` | SIFT correctness: checks detectors, SIFT and matcher work together |
| `hpatches_extract.py` | SIFT benchmarking: runs our SIFT on every HPatches patch |
| `hpatches_extract_opencv.py` | SIFT benchmarking: the same with OpenCV SIFT (baseline) |
| `eval_pipeline.py` | Pipeline comparison: Harris-Laplace + SIFT vs DoG + SIFT |

---

## SIFT descriptor: correctness and benchmarking

This section checks that our SIFT implementation is correct. It does **not** compare the detectors.

### 1. Compatibility test

Checks that both detectors' keypoints work with our SIFT and matcher: valid keypoints, valid 128-D descriptors (unit length, clipped, angles in radians), conversion to OpenCV keypoints, and correct matches between an image and the same image rotated by exactly 90°. From `src/`:

```bash
cd ~/COMP3821/image_matching/src
python3 test_comp.py                     # synthetic test image
python3 test_comp.py --image photo.jpg   # your own image
```

All checks should pass (`== Summary: N passed, 0 failed ==`).

### 2. HPatches descriptor benchmark

Evaluates the descriptor on its own with the official HPatches toolkit, on patches that are already detected, scaled and oriented. Our extractors write the descriptors to CSV files (the toolkit's required input format); the toolkit computes all scores, so results are directly comparable with published HPatches numbers.

**2.1. Clone the toolkit next to this repository and fix it for current Python.** The toolkit was written for older pandas/NumPy and crashes without these three fixes:

```bash
cd ~/COMP3821
git clone https://github.com/hpatches/hpatches-benchmark.git
cd hpatches-benchmark
sed -i 's/\.as_matrix()/.to_numpy()/g' python/utils/tasks.py
sed -i 's/np\.trapz(/getattr(np, "trapezoid", getattr(np, "trapz", None))(/g' python/utils/metrics.py
sed -i "s/D\[i-1\] = dist_matrix(d,d_,descr\['distance'\])\[0\]/D[i-1] = dist_matrix(d,d_,descr['distance'])[0, 0]/" python/utils/tasks.py
git diff --stat    # should list python/utils/metrics.py and python/utils/tasks.py
```

**2.2. Download the patch dataset (about 4.2 GB).** The toolkit's `download.sh` is broken, so download it by hand:

```bash
cd ~/COMP3821/hpatches-benchmark
wget -O data/hpatches-release.zip https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-release.zip
unzip -q data/hpatches-release.zip -d data && rm data/hpatches-release.zip
ls data/hpatches-release | wc -l    # should print 116
```

**2.3. Extract descriptors.** Each patch is described as one keypoint at the patch centre with angle 0 (the patches are already oriented). Run from `src/`:

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

Each run writes `<descriptors>/<name>/<sequence>/<type>.csv` (one 128-D descriptor per row, row *i* = patch *i*). `hpatches_extract.py` first checks itself against `sift.py` (look for `-> OK`). Re-running a command resumes where it stopped.

| Name | Settings | Purpose |
|---|---|---|
| `our-sift` | σ = 65/12 ≈ 5.42, blur = σ (standard SIFT) | default |
| `our-sift-s613` | σ = 6.13, blur = σ | effect of region size |
| `our-sift-cvlike` | σ = 6.13, blur = 1.6 | same settings as OpenCV in this benchmark |
| `our-sift-cvlike-root` | as above + RootSIFT | effect of RootSIFT |
| `opencv-sift` | OpenCV SIFT, centre keypoint as in the toolkit's own example | baseline |
| `opencv-rootsift` | OpenCV SIFT + RootSIFT | baseline |

σ sets the size of the described region (the 4×4 grid is 12σ wide). Blur is the Gaussian smoothing of the image the gradients are measured on; `--blur 1.6` reproduces what OpenCV does with the toolkit's keypoint. RootSIFT L1-normalises each descriptor and takes the square root of every value.

**2.4. Evaluate and print the results:**

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

This uses split `a` (40 test sequences). Scores are cached in `hpatches-benchmark/python/results/`; if you re-extract a descriptor under the same name, delete `results/<name>_*` first.

**Expected results** (split `a`, mAP %; verification = mean imbalanced AP, retrieval at 20,000 distractors):

| Descriptor | Verification | Matching | Retrieval |
|---|---|---|---|
| `our-sift` | 66.4 | 16.9 | 22.6 |
| `our-sift-s613` | 64.8 | 15.1 | 20.9 |
| `our-sift-cvlike` | 67.0 | 25.5 | 32.4 |
| `opencv-sift` | 66.9 | 25.3 | 32.2 |
| `our-sift-cvlike-root` | 62.2 | 30.1 | 35.7 |
| `opencv-rootsift` | 61.8 | 30.0 | 35.5 |

With the same settings, our SIFT and OpenCV's differ by less than 0.5 points on every task, which validates our implementation.

---

## Pipeline comparison: Harris-Laplace vs DoG

This section compares the two full pipelines directly. It assumes SIFT has already been validated (previous section). Everything except the detector is identical for both pipelines: same images, same SIFT (`sift.py`), same matcher (`matcher.py`, cross-checked), same number of keypoints. OpenCV's SIFT pipeline is included as a reference point.

### 1. Download the image sequences (about 1.3 GB)

```bash
mkdir -p ~/COMP3821/data && cd ~/COMP3821/data
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
tar -xzf hpatches-sequences-release.tar.gz && rm hpatches-sequences-release.tar.gz
ls hpatches-sequences-release | wc -l    # should print 116
```

There are 57 illumination sequences (`i_*`: same viewpoint, different lighting) and 59 viewpoint sequences (`v_*`: planar scenes from different angles). Each folder has `1.ppm` to `6.ppm` and the ground-truth homographies `H_1_2` to `H_1_6`.

### 2. Run the comparison

Quick test on 4 illumination and 4 viewpoint sequences (multiply the time by about 15 to estimate the full run):

```bash
cd ~/COMP3821/image_matching/src
time python3 eval_pipeline.py ~/COMP3821/data/hpatches-sequences-release results/pipeline --limit 4
```

Full run:

```bash
python3 eval_pipeline.py ~/COMP3821/data/hpatches-sequences-release results/pipeline
```

Check the **Warning** line under Table 1 in the output. If the pipelines kept different numbers of keypoints, re-run with the `--num-kps` value it suggests, so that neither pipeline wins by having more points:

```bash
python3 eval_pipeline.py ~/COMP3821/data/hpatches-sequences-release results/pipeline --num-kps 300
```

Other options: `--methods harris dog` (skip the OpenCV reference), `--max-side 0` (full resolution, much slower), `--workers N`. See `python3 eval_pipeline.py --help`.

### 3. What it measures

Every image is downscaled to at most 640 px on its longest side (`--max-side`), with the homographies rescaled to match. Each detector keeps its `--num-kps` strongest keypoints (default 500). For every image pair (1→2 … 1→6 in each sequence) it reports:

- **Repeatability**: % of keypoints found again within 3 px in the other image (detector only)
- **MMA@t**: % of matches within *t* px of the true position (t = 1 to 10)
- **Correct matches** per image pair, at 3 px
- **Homography accuracy**: % of pairs where the RANSAC-estimated homography maps the image corners within 1 / 3 / 5 px of the truth
- **Runtime** per image

Results are given for all pairs, illumination only and viewpoint only, and per image pair.

### 4. Outputs

In `results/pipeline/` (the `TAG` is e.g. `n500_s640`, from the settings used):

| File | Contents |
|---|---|
| `summary_TAG.md` | Result tables |
| `pairs_TAG.csv` | Every metric for every pipeline and image pair |
| `mma_TAG.png` | MMA vs pixel threshold, illumination and viewpoint |
| `mma_by_pair_TAG.png` | MMA@3px vs image pair |
| `cache/` | Per-image keypoints and descriptors; re-runs with the same settings reuse them |

Do not commit `results/`, `data/` or the datasets; add them to `.gitignore`.
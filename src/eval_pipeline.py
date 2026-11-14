"""End-to-end evaluation of the image matching pipelines on HPatches sequences.

Pipelines compared (everything except the detector is identical):
    harris : HarrisLaplaceDetector -> our SIFT -> our BFMatcher (cross-check)
    dog    : DoGDetector           -> our SIFT -> our BFMatcher (cross-check)
    opencv : OpenCV SIFT detect+compute        -> our BFMatcher (baseline)

Data: the HPatches *image sequences* (hpatches-sequences-release), 116 folders
i_* (illumination) / v_* (viewpoint), each with 1.ppm..6.ppm and the ground-truth
homographies H_1_2..H_1_6 that map image 1 into image n.

Usage:
    python pipeline_eval.py PATH/TO/hpatches-sequences-release results/pipeline
    python pipeline_eval.py ... --num-kps 1000
    python pipeline_eval.py ... --limit 4          # quick test: 4 i_ + 4 v_ sequences

Outputs in OUT_DIR (TAG = n<num-kps>_s<max-side>):
    pairs_TAG.csv          one row per method and image pair (all raw numbers)
    summary_TAG.md         tables ready to paste into a report
    mma_TAG.png            MMA vs pixel threshold (illumination / viewpoint)
    mma_by_pair_TAG.png    MMA@3px vs pair 1->2 ... 1->6
    cache/                 per-image keypoints + descriptors (re-runs reuse them)

Protocol (applied identically to every method):
  * Each image is downscaled so its longest side is at most --max-side (default
    640; our SIFT is Python) and the homographies are rescaled to match. All
    pixel thresholds are in these downscaled pixels.
  * Each detector keeps its --num-kps strongest keypoints (by response), so no
    pipeline wins just by producing more points. SIFT orientation assignment can
    then give a keypoint several descriptors, exactly as in normal use.
  * Repeatability (detector only): a keypoint of image 1 whose projection lies
    inside image n is repeated if image n has a keypoint within 3 px of it;
    the count is divided by min(#image-1 points visible in n, #image-n points
    visible in 1).
  * MMA@t: fraction of matches whose image-1 point, projected by the true
    homography, lands within t px of its matched point (0 if no matches).
  * Homography accuracy: H estimated from the matches with RANSAC (3 px); the
    four image-1 corners are projected with the estimated and true H; a pair
    is correct at t px if the mean corner error is <= t.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one process per sequence uses the cores

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from detector_helpers import Keypoint
from dog import DoGDetector
from harris import HarrisLaplaceDetector
from matcher import BFMatcher
from sift import sift_describe

METHODS = ("harris", "dog", "opencv")
LABELS = {"harris": "Harris-Laplace + SIFT", "dog": "DoG + SIFT", "opencv": "OpenCV SIFT"}
THRESHOLDS = list(range(1, 11))
REP_EPS = 3.0
H_THRESHOLDS = (1, 3, 5)


# --------------------------------------------------------------------------
# Images and homographies
# --------------------------------------------------------------------------

def load_image(path: str, max_side: int):
    """Return (downscaled BGR image, S) where S maps original -> downscaled pixels."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w)) if max_side > 0 else 1.0
    if s < 1.0:
        nw, nh = max(8, round(w * s)), max(8, round(h * s))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    sx, sy = img.shape[1] / w, img.shape[0] / h
    # Pixel centres: x' = sx * (x + 0.5) - 0.5
    S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])
    return img, S


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def inside(pts: np.ndarray, shape) -> np.ndarray:
    h, w = shape[:2]
    return ((pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) &
            (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1))


# --------------------------------------------------------------------------
# Feature extraction (cached per image)
# --------------------------------------------------------------------------

def top_n(kps: list[Keypoint], n: int) -> list[Keypoint]:
    return sorted(kps, key=lambda k: -k.response)[:n]


def extract(method: str, img: np.ndarray, n: int) -> dict:
    t0 = time.perf_counter()
    if method == "opencv":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = cv2.SIFT_create(nfeatures=n).detectAndCompute(gray, None)
        t1 = time.perf_counter()
        desc = np.zeros((0, 128), np.float32) if desc is None else desc.astype(np.float32)
        desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-7)
        xy = np.array([k.pt for k in kps], np.float64).reshape(-1, 2)
        det_xy = np.unique(np.round(xy, 2), axis=0)  # orientations duplicate points
        return dict(det_xy=det_xy, desc_xy=xy, desc=desc,
                    t_detect=t1 - t0, t_describe=0.0, shape=np.array(img.shape[:2]))

    detector = HarrisLaplaceDetector(max_points=None) if method == "harris" else DoGDetector()
    kps = top_n(detector.detect(img), n)
    t1 = time.perf_counter()
    out_kps, desc = sift_describe(img, kps)
    t2 = time.perf_counter()
    return dict(det_xy=np.array([(k.x, k.y) for k in kps], np.float64).reshape(-1, 2),
                desc_xy=np.array([(k.x, k.y) for k in out_kps], np.float64).reshape(-1, 2),
                desc=np.asarray(desc, np.float32).reshape(-1, 128),
                t_detect=t1 - t0, t_describe=t2 - t1, shape=np.array(img.shape[:2]))


def features(cache_dir: str, seq: str, i: int, method: str, img: np.ndarray,
             n: int) -> dict:
    path = os.path.join(cache_dir, method, seq, f"{i}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    f = extract(method, img, n)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **f)
    os.replace(tmp, path)
    return f


# --------------------------------------------------------------------------
# Metrics for one image pair
# --------------------------------------------------------------------------

def repeatability(f1: dict, fn: dict, H: np.ndarray) -> float:
    p1, pn = f1["det_xy"], fn["det_xy"]
    if len(p1) == 0 or len(pn) == 0:
        return 0.0
    p1n = project(H, p1)
    vis1 = inside(p1n, fn["shape"])
    visn = inside(project(np.linalg.inv(H), pn), f1["shape"])
    denom = min(vis1.sum(), visn.sum())
    if denom == 0:
        return 0.0
    d = cdist(p1n[vis1], pn[visn])
    return float(np.count_nonzero(d.min(axis=1) <= REP_EPS) / denom)


def pair_metrics(f1: dict, fn: dict, H: np.ndarray) -> dict:
    row = dict(kps1=len(f1["det_xy"]), kpsn=len(fn["det_xy"]),
               repeatability=repeatability(f1, fn, H))
    matches = BFMatcher().match(f1["desc"], fn["desc"])
    row["matches"] = len(matches)
    if matches:
        q = np.array([m.query_idx for m in matches])
        t = np.array([m.train_idx for m in matches])
        m1, mn = f1["desc_xy"][q], fn["desc_xy"][t]
        err = np.linalg.norm(project(H, m1) - mn, axis=1)
    else:
        m1 = mn = np.zeros((0, 2))
        err = np.zeros(0)
    for th in THRESHOLDS:
        row[f"mma@{th}"] = float(np.mean(err <= th)) if len(err) else 0.0
    row["correct@3"] = int(np.count_nonzero(err <= 3))

    h_err, inliers = float("inf"), 0.0
    if len(m1) >= 4:
        H_est, mask = cv2.findHomography(m1, mn, cv2.RANSAC, 3.0)
        if H_est is not None:
            h, w = f1["shape"]
            corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
            h_err = float(np.linalg.norm(project(H_est, corners) - project(H, corners),
                                         axis=1).mean())
            inliers = float(mask.mean())
    row["h_corner_err"] = h_err
    row["ransac_inlier_ratio"] = inliers
    return row


def process_sequence(seq_dir: str, methods: list[str], n: int, max_side: int,
                     cache_dir: str) -> tuple[list, list]:
    seq = os.path.basename(os.path.normpath(seq_dir))
    imgs, Ss = {}, {}
    for i in range(1, 7):
        imgs[i], Ss[i] = load_image(os.path.join(seq_dir, f"{i}.ppm"), max_side)

    pair_rows, image_rows = [], []
    for method in methods:
        feats = {i: features(cache_dir, seq, i, method, imgs[i], n)
                 for i in range(1, 7)}
        for i in range(1, 7):
            image_rows.append(dict(method=method, seq=seq, image=i,
                                   kps=len(feats[i]["det_xy"]), descs=len(feats[i]["desc"]),
                                   t_detect=float(feats[i]["t_detect"]),
                                   t_describe=float(feats[i]["t_describe"])))
        for k in range(2, 7):
            H = np.loadtxt(os.path.join(seq_dir, f"H_1_{k}"))
            Hs = Ss[k] @ H @ np.linalg.inv(Ss[1])  # homography between downscaled images
            row = dict(method=method, seq=seq, kind=seq[0], pair=k)
            row.update(pair_metrics(feats[1], feats[k], Hs))
            pair_rows.append(row)
    return pair_rows, image_rows


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------

def pct(x) -> str:
    return f"{100 * x:.1f}"


def summarise(rows: list[dict], image_rows: list[dict], methods: list[str],
              n: int, max_side: int, n_seqs: dict) -> str:
    def sel(m, kind=None, pair=None):
        return [r for r in rows if r["method"] == m and (kind is None or r["kind"] == kind)
                and (pair is None or r["pair"] == pair)]

    out = [f"# Pipeline evaluation on HPatches sequences",
           "",
           f"{n_seqs['i']} illumination + {n_seqs['v']} viewpoint sequences, "
           f"{len(rows) // max(len(methods), 1)} image pairs per method. "
           f"Top {n} keypoints per image, images downscaled to <= {max_side} px "
           f"(pixel thresholds are in downscaled pixels). Percentages unless stated.",
           ""]

    head = ("| Pipeline | Subset | Keypoints/img | Repeatability | Matches/pair | "
            "MMA@1 | MMA@3 | MMA@5 | Correct@3/pair | "
            + " | ".join(f"H acc@{t}px" for t in H_THRESHOLDS) + " |")
    out += ["## Table 1. Overall results", "", head,
            "|" + "---|" * (head.count("|") - 1)]
    for m in methods:
        for kind, name in ((None, "All"), ("i", "Illumination"), ("v", "Viewpoint")):
            r = sel(m, kind)
            if not r:
                continue
            mean = lambda key: float(np.mean([x[key] for x in r]))
            hacc = [pct(np.mean([x["h_corner_err"] <= t for x in r])) for t in H_THRESHOLDS]
            out.append(f"| {LABELS[m]} | {name} | {np.mean([x['kps1'] for x in r]):.0f} | "
                       f"{pct(mean('repeatability'))} | {mean('matches'):.0f} | "
                       f"{pct(mean('mma@1'))} | {pct(mean('mma@3'))} | {pct(mean('mma@5'))} | "
                       f"{mean('correct@3'):.0f} | " + " | ".join(hacc) + " |")

    means = {m: np.mean([x["kps1"] for x in sel(m)]) for m in methods if sel(m)}
    if means and max(means.values()) > 1.2 * min(means.values()):
        out += ["", "**Warning:** the pipelines kept different numbers of keypoints per image "
                + ", ".join(f"{LABELS[m]} {v:.0f}" for m, v in means.items())
                + f" (some detectors found fewer than {n}). More keypoints inflate matches and "
                "repeatability, so for a fair comparison re-run with --num-kps at or below "
                f"the smallest of these ({min(means.values()):.0f}), or lower the detector thresholds."]

    for key, title in (("mma@3", "Table 2. MMA@3px by image pair"),
                       ("repeatability", "Table 3. Repeatability by image pair")):
        out += ["", f"## {title}", "",
                "| Pipeline | Subset | 1→2 | 1→3 | 1→4 | 1→5 | 1→6 |",
                "|---|---|---|---|---|---|---|"]
        for m in methods:
            for kind, name in (("i", "Illumination"), ("v", "Viewpoint")):
                if not sel(m, kind):
                    continue
                vals = [pct(np.mean([x[key] for x in sel(m, kind, k)])) for k in range(2, 7)]
                out.append(f"| {LABELS[m]} | {name} | " + " | ".join(vals) + " |")

    out += ["", "## Table 4. Runtime per image (seconds, median)", "",
            "| Pipeline | Detection | Description | Total | Descriptors/img |", "|---|---|---|---|---|"]
    for m in methods:
        r = [x for x in image_rows if x["method"] == m]
        det = np.median([x["t_detect"] for x in r])
        des = np.median([x["t_describe"] for x in r])
        tot = np.median([x["t_detect"] + x["t_describe"] for x in r])
        note = " (combined)" if m == "opencv" else ""
        out.append(f"| {LABELS[m]} | {det:.2f}{note} | {des:.2f} | {tot:.2f} | "
                   f"{np.mean([x['descs'] for x in r]):.0f} |")
    out.append("")
    out.append("Runtimes are for our Python implementations; OpenCV is compiled C++.")
    return "\n".join(out) + "\n"


# Categorical slots 1-3 of the validated default palette (blue, orange, aqua).
COLOURS = {"harris": "#2a78d6", "dog": "#eb6834", "opencv": "#1baf7a"}
MARKERS = {"harris": "o", "dog": "s", "opencv": "^"}


def plot(rows: list[dict], methods: list[str], out_dir: str, tag: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b5b4ae",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False})
    figs = [
        ("mma", "Pixel threshold", THRESHOLDS,
         lambda r, x: np.mean([q[f"mma@{x}"] for q in r])),
        ("mma_by_pair", "Image pair", list(range(2, 7)),
         lambda r, x: np.mean([q["mma@3"] for q in r if q["pair"] == x])),
    ]
    for name, xlabel, xs, fn in figs:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, (kind, title) in zip(axes, (("i", "Illumination"), ("v", "Viewpoint"))):
            ends = []
            for m in methods:
                r = [q for q in rows if q["method"] == m and q["kind"] == kind]
                if not r:
                    continue
                ys = [100 * fn(r, x) for x in xs]
                ax.plot(xs, ys, color=COLOURS[m], marker=MARKERS[m], markersize=6,
                        linewidth=2, label=LABELS[m])
                ends.append([ys[-1], ys[-1], m])
            # End-of-line value labels, pushed apart so close values don't overlap.
            ends.sort(key=lambda e: e[0])
            for j in range(1, len(ends)):
                ends[j][1] = max(ends[j][1], ends[j - 1][1] + 5.5)
            for value, ypos, m in ends:
                ax.annotate(f"{value:.0f}", (xs[-1], ypos), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=9,
                            color="#0b0b0b", annotation_clip=False)
            ax.set_title(title, color="#0b0b0b", loc="left")
            ax.set_xlabel(xlabel)
            ax.set_xticks(xs)
            if name == "mma_by_pair":
                ax.set_xticklabels([f"1→{x}" for x in xs])
            ax.set_ylim(0, 100)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("MMA (%)" if name == "mma" else "MMA@3px (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(os.path.join(out_dir, f"{name}_{tag}.png"), dpi=200)
        plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq_root", help="hpatches-sequences-release folder (i_* and v_* folders)")
    ap.add_argument("out_dir", help="where results, plots and the cache are written")
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--num-kps", type=int, default=500, help="keypoints kept per image (default 500)")
    ap.add_argument("--max-side", type=int, default=640,
                    help="downscale images to this longest side (default 640, 0 = full size)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first K illumination and first K viewpoint sequences")
    args = ap.parse_args()

    seqs = sorted(d for d in os.listdir(args.seq_root)
                  if os.path.isdir(os.path.join(args.seq_root, d)) and d[:2] in ("i_", "v_"))
    if args.limit:
        seqs = ([s for s in seqs if s[0] == "i"][:args.limit] +
                [s for s in seqs if s[0] == "v"][:args.limit])
    if not seqs:
        print(f"No i_*/v_* sequence folders in {args.seq_root}")
        return 2
    n_seqs = {k: sum(s[0] == k for s in seqs) for k in "iv"}
    tag = f"n{args.num_kps}_s{args.max_side}"
    cache_dir = os.path.join(args.out_dir, "cache", tag)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(seqs)} sequences ({n_seqs['i']} i_, {n_seqs['v']} v_), methods {args.methods}, "
          f"top {args.num_kps} keypoints, max side {args.max_side}")

    t0 = time.perf_counter()
    rows, image_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_sequence, os.path.join(args.seq_root, s), args.methods,
                               args.num_kps, args.max_side, cache_dir): s
                   for s in seqs}
        for k, (fut, s) in enumerate(futures.items(), 1):
            pr, ir = fut.result()
            rows += pr
            image_rows += ir
            print(f"[{k:3d}/{len(seqs)}] {s}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    csv_path = os.path.join(args.out_dir, f"pairs_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows, image_rows, args.methods, args.num_kps, args.max_side, n_seqs)
    with open(os.path.join(args.out_dir, f"summary_{tag}.md"), "w") as f:
        f.write(summary)
    plot(rows, args.methods, args.out_dir, tag)
    print("\n" + summary)
    print(f"Wrote {csv_path}, summary_{tag}.md, mma_{tag}.png, mma_by_pair_{tag}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
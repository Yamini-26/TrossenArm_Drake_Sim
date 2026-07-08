#!/usr/bin/env python
import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Camera config

CAMERA_NAMES = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]


# Feature loading

def load_features(features_dir: Path, cam: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-saved CLS and patch .npy arrays for one camera from a features directory.

    Returns (cls_array, patch_array) or raises FileNotFoundError.
    """
    # Try all naming conventions used by the extractor
    candidates_cls = [
        features_dir / f"runA_{cam}_cls.npy",
        features_dir / f"runB_{cam}_cls.npy",
    ]
    candidates_patch = [
        features_dir / f"runA_{cam}_patch.npy",
        features_dir / f"runB_{cam}_patch.npy",
    ]

    cls_path = next((p for p in candidates_cls if p.exists()), None)
    patch_path = next((p for p in candidates_patch if p.exists()), None)

    if cls_path is None:
        raise FileNotFoundError(
            f"No CLS feature file found for camera '{cam}' in {features_dir}.\n"
            f"Looked for: {[str(p) for p in candidates_cls]}"
        )

    cls_arr   = np.load(cls_path).astype(np.float64)    # (N, 768)
    patch_arr = np.load(patch_path).astype(np.float64) if patch_path else None  # (N, 1369, 768)

    print(f"  [{cam}] loaded CLS  {cls_arr.shape}  from {cls_path.name}")
    if patch_arr is not None:
        print(f"  [{cam}] loaded patch {patch_arr.shape} from {patch_path.name}")

    return cls_arr, patch_arr


# Energy Distance

def pairwise_l2_mean(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute mean pairwise Euclidean distance between rows of A and rows of B.

    Uses the ||a-b||² = ||a||² + ||b||² - 2<a,b> identity for efficiency —
    O(n·m·d) via matrix multiply rather than explicit loop.

    A : (n, d)
    B : (m, d)
    Returns scalar: mean of all n*m pairwise distances.
    """
    # ||a||² column vector, ||b||² row vector
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)   # (n, 1)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True)   # (m, 1)

    # Squared distance matrix (n, m) via broadcasting + matrix multiply
    D2 = sq_A + sq_B.T - 2.0 * (A @ B.T)

    # Clip tiny negatives caused by floating-point errors before sqrt
    D2 = np.clip(D2, 0.0, None)

    return float(np.mean(np.sqrt(D2)))


def energy_distance(X: np.ndarray, Y: np.ndarray, n_samples: int = 200, seed: int = 0) -> Dict[str, float]:
    """
    Two-sample energy distance and test statistic T_{n1,n2}.

    Large T → distributions differ.
    E = 0 iff X and Y are identically distributed.

    Parameters
    X, Y       : (N, D) float arrays of L2-normalised CLS features
    n_samples  : subsample size per run for computational efficiency
    seed       : random seed for reproducibility

    Returns
    dict with keys:
        'energy_distance'   : E_{n1,n2} — 0 means identical distributions
        'test_statistic'    : T_{n1,n2} — scale-adjusted version for hypothesis testing
        'cross_term'        : 2·E|Xi-Ym|   (between-run term)
        'within_A'          : E|Xi-Xj|     (within run A spread)
        'within_B'          : E|Yl-Ym|     (within run B spread)
        'n1'                : actual n after subsampling
        'n2'                : actual m after subsampling
    """
    rng = np.random.default_rng(seed)

    # Subsample for tractability (500 frames)
    n1 = min(n_samples, len(X))
    n2 = min(n_samples, len(Y))
    idx_x = rng.choice(len(X), n1, replace=False)
    idx_y = rng.choice(len(Y), n2, replace=False)
    Xs = X[idx_x]
    Ys = Y[idx_y]

    # L2-normalise so that the RBF-like structure is on the unit hypersphere;
    # this makes the energy distance more interpretable (range ~[0, 2])
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    Ys = Ys / (np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-8)

    # Cross term: mean distance between the two runs
    cross = pairwise_l2_mean(Xs, Ys)           # E|Xi - Ym|

    # Within-run terms: internal spread of each distribution
    within_A = pairwise_l2_mean(Xs, Xs)        # E|Xi - Xj|
    within_B = pairwise_l2_mean(Ys, Ys)        # E|Yl - Ym|

    # Energy distance
    E = 2.0 * cross - within_A - within_B

    # Test statistic T_{n1,n2} = (n1*n2 / (n1+n2)) * E
    T = (n1 * n2 / (n1 + n2)) * E

    return {
        "energy_distance": round(float(E), 6),
        "test_statistic":  round(float(T), 4),
        "cross_term":      round(float(cross), 6),     # larger → more different
        "within_A":        round(float(within_A), 6),  # internal spread of run A
        "within_B":        round(float(within_B), 6),  # internal spread of run B
        "n1": int(n1),
        "n2": int(n2),
    }


# Per-frame cosine similarity

def cosine_similarity_per_frame(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """
    CLS-token cosine similarity frame by frame.
    Aligns by minimum length. Returns shape (N,).
    """
    N = min(len(feats_a), len(feats_b))
    A = feats_a[:N]
    B = feats_b[:N]
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (A * B).sum(axis=1)   # (N,)


# Visualisations

def plot_similarity_timeline(similarities: Dict[str, np.ndarray], output_path: Path, title: str = "Per-frame CLS cosine similarity (run A vs run B)"):
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]
    for (cam, sim), color in zip(similarities.items(), colors):
        ax.plot(sim, label=cam, color=color, linewidth=1.5, alpha=0.85)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_energy_breakdown(metrics: Dict[str, Dict],output_path: Path):
    """
    Bar chart showing the three energy distance terms per camera.
    Helps diagnose WHY distributions differ (high cross vs low within, etc.)
    """
    cams   = list(metrics.keys())
    cross  = [metrics[c]["cross_term"]  for c in cams]
    within_a = [metrics[c]["within_A"] for c in cams]
    within_b = [metrics[c]["within_B"] for c in cams]
    energy = [metrics[c]["energy_distance"] for c in cams]

    x = np.arange(len(cams))
    w = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Three terms
    ax1.bar(x - w,   cross,    w, label="Cross E|Xi-Ym|",   color="#E74C3C", alpha=0.85)
    ax1.bar(x,       within_a, w, label="Within A E|Xi-Xj|", color="#3498DB", alpha=0.85)
    ax1.bar(x + w,   within_b, w, label="Within B E|Yl-Ym|", color="#2ECC71", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cams, rotation=15, ha="right")
    ax1.set_ylabel("Mean Euclidean distance (L2-normalised features)")
    ax1.set_title("Energy Distance Components per Camera")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: Net energy distance (= 2*cross - within_a - within_b)
    colors_bar = ["#E74C3C" if e > 0.01 else "#2ECC71" for e in energy]
    ax2.bar(x, energy, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cams, rotation=15, ha="right")
    ax2.set_ylabel("Energy Distance E_{n1,n2}(X,Y)")
    ax2.set_title("Net Energy Distance per Camera\n(0 = identical distributions)")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")

    for i, (e, cam) in enumerate(zip(energy, cams)):
        ax2.text(i, e + 0.0005, f"{e:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_path}")


def plot_pca(feats_a: np.ndarray, feats_b: np.ndarray, cam_name: str, output_path: Path, n_samples: int = 300):
    """2D PCA scatter of CLS features to visualise distribution overlap."""
    rng = np.random.default_rng(42)
    idx_a = rng.choice(len(feats_a), min(n_samples, len(feats_a)), replace=False)
    idx_b = rng.choice(len(feats_b), min(n_samples, len(feats_b)), replace=False)
    A = feats_a[idx_a]
    B = feats_b[idx_b]
    pca  = PCA(n_components=2)
    both = pca.fit_transform(np.vstack([A, B]))
    na   = len(A)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(both[:na, 0], both[:na, 1], s=12, alpha=0.6,
               label="Run A", color="#3498DB")
    ax.scatter(both[na:, 0], both[na:, 1], s=12, alpha=0.6,
               label="Run B", color="#E74C3C")
    ax.set_title(f"{cam_name} — PCA of DINOv2 CLS features")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {output_path}")


# Report

def save_report(metrics: Dict, output_dir: Path):
    report_path = output_dir / "energy_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReport saved → {report_path}")

    # Print the report
    print("SIM-TO-SIM ENERGY STATISTICS COMPARISON")

    for cam, m in metrics.items():
        print(f"\n  Camera : {cam}")
        print(f"  Energy distance  E(X,Y)   : {m['energy_distance']:.6f}")
        print(f"  Test statistic   T(n1,n2) : {m['test_statistic']:.4f}")
        print(f"  Components")
        print(f"    Cross term  2·E|Xi-Ym|  : {m['cross_term']:.6f}  ← between runs")
        print(f"    Within A    E|Xi-Xj|    : {m['within_A']:.6f}  ← spread of run A")
        print(f"    Within B    E|Yl-Ym|    : {m['within_B']:.6f}  ← spread of run B")
        print(f"  Per-frame cosine similarity")
        print(f"    Mean cos sim            : {m['mean_cos_sim']:.4f}")
        print(f"    Min  cos sim            : {m['min_cos_sim']:.4f}")
        print(f"  Frames compared : {m['n_frames']}  (subsampled: n1={m['n1']}, n2={m['n2']})")

    print("INTERPRETATION GUIDE")
    print("""
  Energy distance E(X,Y):
    = 0         distributions are IDENTICAL
    < 0.01      negligible sim gap
    0.01–0.05   small but measurable difference
    0.05–0.15   moderate difference — check which cameras
    > 0.15      significant visual sim gap

  Components:
    cross_term >> within_A/B  →  runs are far apart (bad sim fidelity)
    cross_term ≈ within_A/B   →  runs overlap well  (good sim fidelity)
    within_A >> within_B      →  run A more diverse (or noisier)

  Test statistic T_{n1,n2}:
    Larger T → stronger evidence that distributions differ.
    No fixed threshold without permutation test, but
    T > 10    typically indicates a clear distributional gap.

  Cosine similarity (per-frame, aligned):
    → 1.0       frames are visually identical
    < 0.90      noticeable per-frame visual difference
    < 0.75      significant frame-level divergence
""")


def main():
    parser = argparse.ArgumentParser(description="Sim-to-sim comparison using Energy Statistics (Székely & Rizzo 2013)")
    parser.add_argument("--features_a", required=True,help="Directory containing .npy CLS/patch features for run A (output of dino_feature_extractor.py --save_features)")
    parser.add_argument("--features_b", required=True,help="Directory containing .npy CLS/patch features for run B")
    parser.add_argument("--output", default="energy_comparison",help="Output directory for plots and report")
    parser.add_argument("--n_samples", type=int, default=500,help="Number of frames to subsample per run for energy distance computation (default 500 — stable estimates, ~2s per camera)")
    parser.add_argument("--cameras", nargs="+",default=CAMERA_NAMES,help="Camera names to compare (default: all four)")
    args = parser.parse_args()

    features_a = Path(args.features_a)
    features_b = Path(args.features_b)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run A features : {features_a}")
    print(f"Run B features : {features_b}")
    print(f"Output         : {output_dir}")
    print(f"Cameras        : {args.cameras}")
    print(f"Subsample size : {args.n_samples} frames per run\n")

    all_metrics    = {}
    all_cos_sims   = {}

    for cam in args.cameras:
        print(f"\n── Camera: {cam} ──")

        # Load features
        try:
            cls_a, _ = load_features(features_a, cam)
            cls_b, _ = load_features(features_b, cam)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # 1. Energy distance
        print(f"  Computing energy distance (n_samples={args.n_samples}) ...")
        ed = energy_distance(cls_a, cls_b, n_samples=args.n_samples)
        print(f"    E(X,Y)   = {ed['energy_distance']:.6f}")
        print(f"    T(n1,n2) = {ed['test_statistic']:.4f}")
        print(f"    cross    = {ed['cross_term']:.6f}  | within_A = {ed['within_A']:.6f} | within_B = {ed['within_B']:.6f}")

        # 2. Per-frame cosine similarity
        cos_sim = cosine_similarity_per_frame(cls_a, cls_b)
        all_cos_sims[cam] = cos_sim
        print(f"    cos sim  mean={cos_sim.mean():.4f}  min={cos_sim.min():.4f}  "
              f"(over {len(cos_sim)} aligned frames)")
        cos_sim_path = output_dir / f"cos_sim_{cam}.npy"
        np.save(cos_sim_path, cos_sim)
        print(f"    Saved cos_sim → {cos_sim_path}")

        # 3. PCA plot
        plot_pca(cls_a, cls_b, cam, output_dir / f"pca_{cam}.png")

        # Collect metrics
        all_metrics[cam] = {
            **ed,
            "mean_cos_sim": round(float(cos_sim.mean()), 4),
            "min_cos_sim":  round(float(cos_sim.min()),  4),
            "n_frames":     int(len(cos_sim)),
        }

    # Plots across all cameras
    if all_cos_sims:
        plot_similarity_timeline(
            all_cos_sims,
            output_dir / "cosine_similarity_timeline.png"
        )

    if all_metrics:
        plot_energy_breakdown(
            all_metrics,
            output_dir / "energy_distance_breakdown.png"
        )

    save_report(all_metrics, output_dir)


if __name__ == "__main__":
    main()
    
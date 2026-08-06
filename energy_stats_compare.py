#!/usr/bin/env python
"""
Reads the .npy files produced by dino_feature_extractor.py's --run_dir
--save_features mode: {cam}_cls.npy and {cam}_patch.npy inside
--features_a / --features_b (two separate directories, one per run).

Metrics (--metrics, default: all three)
----------------------------------------
cls    - energy distance + per-frame cosine similarity on the CLS token
         only. Fast, whole-image, original baseline.
patch  - energy distance on mean-pooled patch tokens + per-frame cosine
         similarity averaged over all patches. Whole-image, but ignores
         CLS entirely -- catches things CLS misses/over-smooths.
worst  - energy distance on mean+std-pooled patch tokens (sensitive to how
         *uniform* a frame is, not just its average -- this is what
         actually shifts when an object is missing/misplaced, far more
         than a uniform lighting change does) + per-frame cosine
         similarity aggregated over the worst-matching patches (--agg,
         default p10). Also plots patch-similarity heatmaps for the worst
         frame and a mid-trajectory reference frame per camera, so we can
         see *where* divergence is concentrated vs. spread evenly.

Usage
-----
Run everything (all three metrics), all cameras:
    python energy_stats_compare.py --features_a dino_features/replay_1785878156/ --features_b dino_features/replay_1785878650/ --output energy_comparison/replay_1785878156_vs_1785878650

Just the worst-patch metric with a more aggressive percentile:
    python energy_stats_compare.py --features_a dino_features/replay_1785878156/ --features_b dino_features/replay_1785878650/ --metrics worst --agg p5

Just CLS, one camera:
    python energy_stats_compare.py --features_a dino_features/replay_1785878156/ --features_b dino_features/replay_1785878650/ --metrics cls --cameras cam_high
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

CAMERA_NAMES = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]


#  Feature loading

def load_features(features_dir: Path, cam: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Load {cam}_cls.npy / {cam}_patch.npy from a features directory
    """
    cls_path = features_dir / f"{cam}_cls.npy"
    patch_path = features_dir / f"{cam}_patch.npy"

    cls_arr = np.load(cls_path).astype(np.float64) if cls_path.exists() else None
    patch_arr = np.load(patch_path).astype(np.float64) if patch_path.exists() else None

    if cls_arr is None and patch_arr is None:
        raise FileNotFoundError(
            f"Neither {cls_path.name} nor {patch_path.name} found in {features_dir}"
        )
    if cls_arr is not None:
        print(f"  [{cam}] loaded CLS   {cls_arr.shape}  from {cls_path.name}")
    if patch_arr is not None:
        print(f"  [{cam}] loaded patch {patch_arr.shape} from {patch_path.name}")

    return {"cls": cls_arr, "patch": patch_arr}


#  Energy distance (operates on whatever (N, D) vectors - raw CLS tokens or pooled patch tokens)

def pairwise_l2_mean(A: np.ndarray, B: np.ndarray) -> float:
    """Mean pairwise Euclidean distance between rows of A and rows of B,
    via the ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b> identity."""
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True)
    D2 = np.clip(sq_A + sq_B.T - 2.0 * (A @ B.T), 0.0, None)
    return float(np.mean(np.sqrt(D2)))


def energy_distance(X: np.ndarray, Y: np.ndarray, n_samples: int = 500, seed: int = 0) -> Dict[str, float]:
    """
    Two-sample energy distance and test statistic T_{n1,n2}.
    Large T -> distributions differ. E = 0 iff X and Y are identically
    distributed. X, Y are L2-normalised before distances are computed.
    """
    rng = np.random.default_rng(seed)
    n1, n2 = min(n_samples, len(X)), min(n_samples, len(Y))
    Xs = X[rng.choice(len(X), n1, replace=False)]
    Ys = Y[rng.choice(len(Y), n2, replace=False)]
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8)
    Ys = Ys / (np.linalg.norm(Ys, axis=1, keepdims=True) + 1e-8)

    cross = pairwise_l2_mean(Xs, Ys)       # E|Xi - Ym|
    within_A = pairwise_l2_mean(Xs, Xs)    # E|Xi - Xj|
    within_B = pairwise_l2_mean(Ys, Ys)    # E|Yl - Ym|
    E = 2.0 * cross - within_A - within_B
    T = (n1 * n2 / (n1 + n2)) * E

    return {
        "energy_distance": round(float(E), 6),
        "test_statistic": round(float(T), 4),
        "cross_term": round(float(cross), 6),
        "within_A": round(float(within_A), 6),
        "within_B": round(float(within_B), 6),
        "n1": int(n1), "n2": int(n2),
    }


#  Patch pooling / aggregation

def pool_patch_tokens(patch_arr: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Reduce (N, P, D) patch tokens to (N, D') per-frame vectors, as a
    drop-in replacement for the CLS array in energy_distance().

    mode:
      'mean'    - plain global average pooling over all patches. Cheap
                  drop-in swap for CLS; still fairly lighting-sensitive
                  for the same reason CLS is (a uniform brightness shift
                  moves every patch a bit, and that survives averaging).
      'meanstd' - concatenates per-channel mean AND std over patches
                  ([mean_D, std_D] -> 2D-dim vector). std across patches
                  captures how *uniform* the frame is, which changes far
                  more when an object is missing/misplaced than when only
                  lighting shifts.
    """
    if mode == "mean":
        return patch_arr.mean(axis=1)
    elif mode == "meanstd":
        return np.concatenate([patch_arr.mean(axis=1), patch_arr.std(axis=1)], axis=1)
    raise ValueError(f"Unknown pooling mode: {mode}")


def patch_cosine_map(patch_a: np.ndarray, patch_b: np.ndarray) -> np.ndarray:
    """Per-patch cosine similarity between two aligned frames. Returns (P,)."""
    a = patch_a / (np.linalg.norm(patch_a, axis=1, keepdims=True) + 1e-8)
    b = patch_b / (np.linalg.norm(patch_b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)


AGG_FUNCS = {
    "mean": lambda sims: sims.mean(axis=1),
    "min":  lambda sims: sims.min(axis=1),
    "p5":   lambda sims: np.percentile(sims, 5, axis=1),
    "p10":  lambda sims: np.percentile(sims, 10, axis=1),
    "p25":  lambda sims: np.percentile(sims, 25, axis=1),
}


def cosine_similarity_per_frame_cls(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """CLS-token cosine similarity frame by frame. Aligns by minimum length."""
    N = min(len(feats_a), len(feats_b))
    A = feats_a[:N] / (np.linalg.norm(feats_a[:N], axis=1, keepdims=True) + 1e-8)
    B = feats_b[:N] / (np.linalg.norm(feats_b[:N], axis=1, keepdims=True) + 1e-8)
    return (A * B).sum(axis=1)


def cosine_similarity_per_frame_patch(patch_a: np.ndarray, patch_b: np.ndarray,
                                       agg: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """
    Frame-by-frame patch-level cosine similarity. For each aligned frame
    pair, computes the full per-patch similarity map, then aggregates
    across patches with `agg`.

    Returns:
      agg_sim : (N,) aggregated similarity per frame
      sims    : (N, P) full per-patch similarity maps, kept for heatmaps
    """
    N = min(len(patch_a), len(patch_b))
    A, B = patch_a[:N], patch_b[:N]
    sims = np.empty((N, A.shape[1]), dtype=np.float64)
    for i in range(N):
        sims[i] = patch_cosine_map(A[i], B[i])
    return AGG_FUNCS[agg](sims), sims


def infer_grid_size(num_patches: int) -> int:
    """DINO patch tokens are laid out row-major on a square grid."""
    grid = int(round(np.sqrt(num_patches)))
    if grid * grid != num_patches:
        raise ValueError(
            f"Patch count {num_patches} is not a perfect square -- check the "
            f"input resolution / patch size used at extraction time."
        )
    return grid


#  Plotting

def plot_similarity_timeline(similarities: Dict[str, np.ndarray], output_path: Path, title: str):
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
    print(f"  Saved -> {output_path}")


def plot_energy_breakdown(metrics: Dict[str, Dict], output_path: Path, title_suffix: str = ""):
    """Bar chart of the three energy-distance terms per camera."""
    cams = list(metrics.keys())
    cross = [metrics[c]["cross_term"] for c in cams]
    within_a = [metrics[c]["within_A"] for c in cams]
    within_b = [metrics[c]["within_B"] for c in cams]
    energy = [metrics[c]["energy_distance"] for c in cams]
    x, w = np.arange(len(cams)), 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(x - w, cross, w, label="Cross E|Xi-Ym|", color="#E74C3C", alpha=0.85)
    ax1.bar(x, within_a, w, label="Within A E|Xi-Xj|", color="#3498DB", alpha=0.85)
    ax1.bar(x + w, within_b, w, label="Within B E|Yl-Ym|", color="#2ECC71", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(cams, rotation=15, ha="right")
    ax1.set_ylabel("Mean Euclidean distance (L2-normalised features)")
    ax1.set_title(f"Energy Distance Components per Camera{title_suffix}")
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="y")

    colors_bar = ["#E74C3C" if e > 0.01 else "#2ECC71" for e in energy]
    ax2.bar(x, energy, color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(cams, rotation=15, ha="right")
    ax2.set_ylabel("Energy Distance E_{n1,n2}(X,Y)")
    ax2.set_title(f"Net Energy Distance per Camera{title_suffix}\n(0 = identical distributions)")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis="y")
    for i, e in enumerate(energy):
        ax2.text(i, e + 0.0005, f"{e:.4f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_pca(feats_a: np.ndarray, feats_b: np.ndarray, cam_name: str, feature_label: str,
             output_path: Path, n_samples: int = 300):
    """2D PCA scatter of the given (N, D) features to visualise distribution overlap."""
    rng = np.random.default_rng(42)
    A = feats_a[rng.choice(len(feats_a), min(n_samples, len(feats_a)), replace=False)]
    B = feats_b[rng.choice(len(feats_b), min(n_samples, len(feats_b)), replace=False)]
    pca = PCA(n_components=2)
    both = pca.fit_transform(np.vstack([A, B]))
    na = len(A)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(both[:na, 0], both[:na, 1], s=12, alpha=0.6, label="Run A", color="#3498DB")
    ax.scatter(both[na:, 0], both[na:, 1], s=12, alpha=0.6, label="Run B", color="#E74C3C")
    ax.set_title(f"{cam_name} - PCA of {feature_label}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_patch_similarity_heatmap(sim_map_1d: np.ndarray, grid_size: int, cam_name: str,
                                   frame_idx: int, output_path: Path, tag: str = ""):
    """
    Reshape a single frame's (P,) patch similarity vector into its spatial
    layout and plot as a heatmap. A lighting-only divergence should look
    uniformly medium everywhere; an object-missing/misplaced divergence
    should show a sharp, localized dark patch.
    """
    grid = sim_map_1d.reshape(grid_size, grid_size)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=sim_map_1d.min(), vmax=1.0)
    ax.set_title(f"{cam_name} - patch similarity, frame {frame_idx}{tag}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


#  Metric pipelines

def run_cls_metric(cls_a: np.ndarray, cls_b: np.ndarray, cam: str, args, output_dir: Path) -> Tuple[Dict, np.ndarray]:
    print(f"  [cls] energy distance (n_samples={args.n_samples}) ...")
    ed = energy_distance(cls_a, cls_b, n_samples=args.n_samples)
    cos_sim = cosine_similarity_per_frame_cls(cls_a, cls_b)
    print(f"    E(X,Y)={ed['energy_distance']:.6f}  T={ed['test_statistic']:.4f}  "
          f"cos_sim mean={cos_sim.mean():.4f} min={cos_sim.min():.4f}")

    np.save(output_dir / f"cos_sim_cls_{cam}.npy", cos_sim)
    plot_pca(cls_a, cls_b, cam, "DINO CLS features", output_dir / f"pca_cls_{cam}.png")

    metrics = {**ed, "mean_cos_sim": round(float(cos_sim.mean()), 4),
               "min_cos_sim": round(float(cos_sim.min()), 4), "n_frames": int(len(cos_sim))}
    return metrics, cos_sim


def run_patch_based_metric(metric_name: str, patch_a: np.ndarray, patch_b: np.ndarray, cam: str,
                            args, output_dir: Path, pool_mode: str, agg_mode: str,
                            heatmap_frames: List[str]) -> Tuple[Dict, np.ndarray]:
    print(f"  [{metric_name}] energy distance (pool={pool_mode}, n_samples={args.n_samples}) ...")
    pooled_a = pool_patch_tokens(patch_a, mode=pool_mode)
    pooled_b = pool_patch_tokens(patch_b, mode=pool_mode)
    ed = energy_distance(pooled_a, pooled_b, n_samples=args.n_samples)

    print(f"  [{metric_name}] per-frame cosine (agg={agg_mode}) ...")
    cos_sim, sim_maps = cosine_similarity_per_frame_patch(patch_a, patch_b, agg=agg_mode)
    print(f"    E(X,Y)={ed['energy_distance']:.6f}  T={ed['test_statistic']:.4f}  "
          f"cos_sim mean={cos_sim.mean():.4f} min={cos_sim.min():.4f}")

    np.save(output_dir / f"cos_sim_{metric_name}_{cam}.npy", cos_sim)
    plot_pca(pooled_a, pooled_b, cam, f"{pool_mode}-pooled patch features",
              output_dir / f"pca_{metric_name}_{cam}.png")

    if heatmap_frames:
        grid_size = infer_grid_size(patch_a.shape[1])
        if "worst" in heatmap_frames:
            worst_idx = int(np.argmin(cos_sim))
            plot_patch_similarity_heatmap(
                sim_maps[worst_idx], grid_size, cam, worst_idx,
                output_dir / f"heatmap_{metric_name}_{cam}_worst_frame{worst_idx}.png", tag=" (worst)")
        if "mid" in heatmap_frames:
            mid_idx = len(cos_sim) // 2
            plot_patch_similarity_heatmap(
                sim_maps[mid_idx], grid_size, cam, mid_idx,
                output_dir / f"heatmap_{metric_name}_{cam}_mid_frame{mid_idx}.png", tag=" (mid, reference)")

    metrics = {**ed, "mean_cos_sim": round(float(cos_sim.mean()), 4),
               "min_cos_sim": round(float(cos_sim.min()), 4), "n_frames": int(len(cos_sim)),
               "pool_mode": pool_mode, "agg_mode": agg_mode}
    return metrics, cos_sim


# patch/worst fixed pooling choices; worst's agg comes from --agg at runtime
METRIC_SPECS = {
    "patch": {"pool": "mean", "agg": "mean", "heatmaps": ["mid"]},
    "worst": {"pool": "meanstd", "heatmaps": ["worst", "mid"]},
}


#  Report

INTERPRETATION_GUIDE = """
INTERPRETATION GUIDE

  Energy distance E(X,Y):
    = 0         distributions are IDENTICAL
    < 0.01      negligible sim gap
    0.01-0.05   small but measurable difference
    0.05-0.15   moderate difference -- check which cameras
    > 0.15      significant visual sim gap

  Components:
    cross_term >> within_A/B  ->  runs are far apart (bad sim fidelity)
    cross_term ~= within_A/B  ->  runs overlap well  (good sim fidelity)
    within_A >> within_B      ->  run A more diverse (or noisier)

  Test statistic T_{n1,n2}:
    Larger T -> stronger evidence that distributions differ.
    No fixed threshold without a permutation test, but T > 10 typically
    indicates a clear distributional gap.

  Cosine similarity (per-frame, aligned):
    -> 1.0      frames are visually identical (by this metric)
    < 0.90      noticeable per-frame visual difference
    < 0.75      significant frame-level divergence
"""


def save_report(all_metrics: Dict[str, Dict[str, Dict]], output_dir: Path):
    """all_metrics: {metric_name: {cam: metrics_dict}}"""
    report_path = output_dir / "energy_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nReport saved -> {report_path}")

    print("\nSIM-TO-SIM ENERGY STATISTICS COMPARISON")
    for metric_name, cams in all_metrics.items():
        print(f"\n{'=' * 60}\n{metric_name.upper()} METRIC\n{'=' * 60}")
        for cam, m in cams.items():
            print(f"\n  Camera : {cam}")
            print(f"  Energy distance  E(X,Y)   : {m['energy_distance']:.6f}")
            print(f"  Test statistic   T(n1,n2) : {m['test_statistic']:.4f}")
            print(f"  Components")
            print(f"    Cross term  2*E|Xi-Ym|  : {m['cross_term']:.6f}  <- between runs")
            print(f"    Within A    E|Xi-Xj|    : {m['within_A']:.6f}  <- spread of run A")
            print(f"    Within B    E|Yl-Ym|    : {m['within_B']:.6f}  <- spread of run B")
            print(f"  Per-frame cosine similarity")
            print(f"    Mean cos sim            : {m['mean_cos_sim']:.4f}")
            print(f"    Min  cos sim            : {m['min_cos_sim']:.4f}")
            print(f"  Frames compared : {m['n_frames']}  (subsampled: n1={m['n1']}, n2={m['n2']})")


#  Main

def main():
    parser = argparse.ArgumentParser(description="Sim-to-sim comparison using Energy Statistics (Szekely & Rizzo 2013)")
    parser.add_argument("--features_a", required=True,
                         help="Directory of {cam}_cls.npy / {cam}_patch.npy for run A "
                              "(output of dino_feature_extractor.py --run_dir --save_features)")
    parser.add_argument("--features_b", required=True,
                         help="Directory of {cam}_cls.npy / {cam}_patch.npy for run B")
    parser.add_argument("--output", default="energy_comparison", help="Output directory for plots and report")
    parser.add_argument("--n_samples", type=int, default=500,
                         help="Frames to subsample per run for energy distance (default 500)")
    parser.add_argument("--cameras", nargs="+", default=CAMERA_NAMES, help="Cameras to compare (default: all four)")
    parser.add_argument("--metrics", nargs="+", choices=["cls", "patch", "worst", "all"], default=["all"],
                         help="Which metric(s) to compute (default: all three)")
    parser.add_argument("--agg", default="p10", choices=list(AGG_FUNCS.keys()),
                         help="Per-frame patch aggregation used by the 'worst' metric "
                              "(default p10; try p5 or min for a more localized/aggressive signal)")
    args = parser.parse_args()

    features_a, features_b = Path(args.features_a), Path(args.features_b)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_requested = set(args.metrics)
    if "all" in metrics_requested:
        metrics_requested = {"cls", "patch", "worst"}

    print(f"Run A features : {features_a}")
    print(f"Run B features : {features_b}")
    print(f"Output         : {output_dir}")
    print(f"Cameras        : {args.cameras}")
    print(f"Metrics        : {sorted(metrics_requested)}")
    print(f"Subsample size : {args.n_samples} frames per run\n")

    all_metrics: Dict[str, Dict[str, Dict]] = {m: {} for m in metrics_requested}
    all_cos_sims: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in metrics_requested}

    for cam in args.cameras:
        print(f"\n-- Camera: {cam} --")
        try:
            feats_a = load_features(features_a, cam)
            feats_b = load_features(features_b, cam)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        if "cls" in metrics_requested:
            if feats_a["cls"] is None or feats_b["cls"] is None:
                print(f"  [SKIP cls] no CLS features for {cam}")
            else:
                m, cs = run_cls_metric(feats_a["cls"], feats_b["cls"], cam, args, output_dir)
                all_metrics["cls"][cam] = m
                all_cos_sims["cls"][cam] = cs

        if "patch" in metrics_requested:
            if feats_a["patch"] is None or feats_b["patch"] is None:
                print(f"  [SKIP patch] no patch features for {cam}")
            else:
                spec = METRIC_SPECS["patch"]
                m, cs = run_patch_based_metric("patch", feats_a["patch"], feats_b["patch"], cam, args,
                                                output_dir, spec["pool"], spec["agg"], spec["heatmaps"])
                all_metrics["patch"][cam] = m
                all_cos_sims["patch"][cam] = cs

        if "worst" in metrics_requested:
            if feats_a["patch"] is None or feats_b["patch"] is None:
                print(f"  [SKIP worst] no patch features for {cam}")
            else:
                spec = METRIC_SPECS["worst"]
                m, cs = run_patch_based_metric("worst", feats_a["patch"], feats_b["patch"], cam, args,
                                                output_dir, spec["pool"], args.agg, spec["heatmaps"])
                all_metrics["worst"][cam] = m
                all_cos_sims["worst"][cam] = cs

    for metric_name, cos_sims in all_cos_sims.items():
        if cos_sims:
            plot_similarity_timeline(
                cos_sims, output_dir / f"cosine_similarity_timeline_{metric_name}.png",
                title=f"Per-frame cosine similarity ({metric_name}) -- run A vs run B")

    for metric_name, cams in all_metrics.items():
        if cams:
            plot_energy_breakdown(cams, output_dir / f"energy_distance_breakdown_{metric_name}.png",
                                   title_suffix=f" ({metric_name})")

    save_report(all_metrics, output_dir)


if __name__ == "__main__":
    main()
    
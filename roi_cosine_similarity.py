"""
ROI-weighted real-vs-sim comparison using cosine similarity of raw DINO features.

Pipeline per camera:

  1. roi_seq   = max(real_temporal_diff, sim_temporal_diff)   # (T, grid, grid) – ROI from temporal diff
  2. Load raw DINO features (real_features, sim_features) with shape (T_raw, grid, grid, D).
     Use only the first T frames (where T = len(roi_seq)) for alignment.
  3. For each frame t:
         similarity_map[t] = cosine_similarity(real_features[t], sim_features[t])   # (grid, grid)
         weights_t = roi_seq[t] normalised to sum to 1 over patches
         score_t   = sum_ij( weights_t[i,j] * similarity_map[t,i,j] )
     -> score_seq is a (T,) array: one weighted‑similarity number per frame.
  4. Collapse score_seq into SIX camera‑level numbers: mean, median, max, p95, rms, activity_weighted.
     (activity_weighted uses frame_activity = sum(roi_seq[t]) as before.)
  5. Visualisations: ROI contact sheet, similarity contact sheet, heatmap panel (mean ROI, mean similarity,
     mean weighted similarity), scores over time, aggregation bar chart, and run‑level camera comparison.

Cameras are NOT combined – you get 4 independent camera results.
"""

import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CAMERAS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

AGG_METHODS = ["mean", "median", "max", "p95", "rms", "activity_weighted"]
AGG_LABELS = {
    "mean": "Mean",
    "median": "Median",
    "max": "Max",
    "p95": "95th pct",
    "rms": "RMS",
    "activity_weighted": "Activity-wtd",
}


# --------------------------------------------------------------------------- #
# Core math (unchanged except variable names)
# --------------------------------------------------------------------------- #

def frame_weights(roi_frame: np.ndarray) -> np.ndarray:
    """Normalise a single (grid, grid) ROI frame to weights summing to 1. No masking."""
    total = roi_frame.sum()
    if total > 0:
        return roi_frame / total
    return np.ones_like(roi_frame) / roi_frame.size


def per_frame_scores(roi_seq: np.ndarray, sim_seq: np.ndarray):
    """
    Args:
      roi_seq: (T, grid, grid) ROI values (max of real/sim temporal diffs)
      sim_seq: (T, grid, grid) cosine similarity per patch
    Returns:
      score_seq          : (T,) weighted similarity per frame
      weighted_sim_seq   : (T, grid, grid) weights_t * sim_seq[t] (for heatmap)
    """
    T = roi_seq.shape[0]
    score_seq = np.empty(T, dtype=float)
    weighted_sim_seq = np.empty_like(sim_seq)
    for t in range(T):
        w_t = frame_weights(roi_seq[t])
        ws = w_t * sim_seq[t]
        weighted_sim_seq[t] = ws
        score_seq[t] = float(ws.sum())
    return score_seq, weighted_sim_seq


def aggregate_scores(score_seq: np.ndarray, frame_activity: np.ndarray) -> dict:
    """Collapse a (T,) per-frame score sequence into ONE camera-level result, 6 ways."""
    agg = {
        "mean": float(np.mean(score_seq)),
        "median": float(np.median(score_seq)),
        "max": float(np.max(score_seq)),
        "p95": float(np.percentile(score_seq, 95)),
        "rms": float(np.sqrt(np.mean(score_seq ** 2))),
    }
    total_activity = frame_activity.sum()
    agg["activity_weighted"] = (
        float(np.sum(score_seq * frame_activity) / total_activity)
        if total_activity > 0 else agg["mean"]
    )
    return agg


# --------------------------------------------------------------------------- #
# Plotting helpers – modified titles to reflect similarity
# --------------------------------------------------------------------------- #

def _contact_sheet(seq: np.ndarray, title: str, output_path: Path, max_frames: int = 16):
    T = seq.shape[0]
    n = min(max_frames, T)
    idxs = np.linspace(0, T - 1, n).astype(int)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, idx in zip(axes, idxs):
        ax.imshow(seq[idx], cmap="hot", interpolation="nearest")
        ax.set_title(f"t={idx}", fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_roi_contact_sheet(roi_seq: np.ndarray, cam: str, output_path: Path):
    _contact_sheet(roi_seq, f"{cam}: combined ROI = max(real, sim), per frame", output_path)


def plot_similarity_contact_sheet(sim_seq: np.ndarray, cam: str, output_path: Path):
    _contact_sheet(sim_seq, f"{cam}: cosine similarity (real vs sim) per frame", output_path)


def plot_heatmap_panel(roi_mean_map, sim_mean_map, weighted_sim_mean_map, cam: str, output_path: Path):
    """3 time-averaged maps: ROI, mean similarity, mean weighted similarity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("Mean ROI\n(where activity happens)", roi_mean_map),
        ("Mean cosine similarity\n(real vs sim, unweighted)", sim_mean_map),
        ("Mean weighted similarity\n(= what the score sums; activity AND similarity)", weighted_sim_mean_map),
    ]
    for ax, (title, m) in zip(axes, panels):
        im = ax.imshow(m, cmap="hot", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{cam}: spatial summary (time-averaged over all frames)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_scores_over_time(score_seq: np.ndarray, cam: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(score_seq, linewidth=1.2, color="firebrick")
    ax.set_xlabel("Frame index (N-stride steps)")
    ax.set_ylabel("ROI-weighted cosine similarity score")
    ax.set_title(f"{cam}: per-frame weighted similarity score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_aggregation_bar(cam_aggregates: dict, cam: str, output_path: Path):
    labels = [AGG_LABELS[m] for m in AGG_METHODS]
    vals = [cam_aggregates[m] for m in AGG_METHODS]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color="steelblue")
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=2)
    ax.set_ylabel("Score (cosine similarity)")
    ax.set_title(f"{cam}: aggregation methods compared")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_camera_comparison(scores_by_cam: dict, output_path: Path):
    cams_present = [cam for cam in CAMERAS if cam in scores_by_cam]
    n_methods = len(AGG_METHODS)
    x = np.arange(len(cams_present))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, method in enumerate(AGG_METHODS):
        vals = [scores_by_cam[cam][method] for cam in cams_present]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=AGG_LABELS[method])
        ax.bar_label(bars, fmt="%.3f", fontsize=6, rotation=90, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(cams_present, rotation=15)
    ax.set_ylabel("Score (cosine similarity)")
    ax.set_title("Per-camera scores, all 6 aggregation methods (cameras NOT combined)")
    ax.legend(title="Aggregation", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


# --------------------------------------------------------------------------- #
# Per-camera computation – CHANGED to use raw DINO features
# --------------------------------------------------------------------------- #

def camera_score(real_dir: Path, sim_dir: Path, cam: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load temporal diff sequences for ROI (unchanged)
    real_seq = np.load(real_dir / f"{cam}_temporal_diff_seq.npy")   # (T, grid, grid)
    sim_seq = np.load(sim_dir / f"{cam}_temporal_diff_seq.npy")
    assert real_seq.shape == sim_seq.shape, f"{cam}: frame mismatch {real_seq.shape} vs {sim_seq.shape}"

    roi_seq = np.maximum(real_seq, sim_seq)          # (T, grid, grid)
    T = roi_seq.shape[0]
    grid = roi_seq.shape[1]                          # assuming square grid (grid x grid)
    frame_activity = roi_seq.sum(axis=(1, 2))        # (T,) for activity-weighted agg

    # 2. Load raw PATCH features and slice to match T
    real_patch = np.load(real_dir / f"{cam}_patch.npy")   # shape: (T_raw, N, D) or (T_raw, grid, grid, D)
    sim_patch = np.load(sim_dir / f"{cam}_patch.npy")

    # Reshape if necessary to (T, grid, grid, D)
    if real_patch.ndim == 3:   # (T, N, D)
        N = real_patch.shape[1]
        assert N == grid * grid, f"Number of patches {N} does not match grid² {grid*grid}"
        real_patch = real_patch.reshape(-1, grid, grid, real_patch.shape[-1])
        sim_patch = sim_patch.reshape(-1, grid, grid, sim_patch.shape[-1])
    elif real_patch.ndim == 4:
        # Already (T, grid, grid, D) – just check the spatial dimensions
        assert real_patch.shape[1:3] == (grid, grid), f"Patch spatial shape {real_patch.shape[1:3]} != ({grid},{grid})"
    else:
        raise ValueError(f"Unexpected patch shape: {real_patch.shape}")

    # Slice to first T frames
    real_patch = real_patch[:T]
    sim_patch = sim_patch[:T]
    assert real_patch.shape == sim_patch.shape, f"{cam}: patch shape mismatch"

    # 3. Compute cosine similarity per patch for each frame
    similarity_seq = np.zeros((T, grid, grid), dtype=np.float32)
    eps = 1e-8
    for t in range(T):
        rf = real_patch[t]          # (grid, grid, D)
        sf = sim_patch[t]
        norm_r = np.linalg.norm(rf, axis=-1)
        norm_s = np.linalg.norm(sf, axis=-1)
        dot = np.sum(rf * sf, axis=-1)
        similarity_seq[t] = dot / (norm_r * norm_s + eps)

    # 4. Weighted scoring (same as before)
    score_seq, weighted_sim_seq = per_frame_scores(roi_seq, similarity_seq)

    # 5. Aggregate
    cam_aggregates = aggregate_scores(score_seq, frame_activity)
    agg_str = "  ".join(f"{AGG_LABELS[m]}={cam_aggregates[m]:.4f}" for m in AGG_METHODS)
    print(f"  {cam}: {agg_str}")

    # 6. Save intermediate arrays
    np.save(output_dir / f"{cam}_combined_roi_seq.npy", roi_seq)
    np.save(output_dir / f"{cam}_similarity_seq.npy", similarity_seq)
    np.save(output_dir / f"{cam}_per_frame_scores.npy", score_seq)
    print(f"  Saved -> {output_dir / f'{cam}_combined_roi_seq.npy'} (shape={roi_seq.shape})")
    print(f"  Saved -> {output_dir / f'{cam}_similarity_seq.npy'} (shape={similarity_seq.shape})")
    print(f"  Saved -> {output_dir / f'{cam}_per_frame_scores.npy'}")

    # 7. Generate plots (using similarity maps now)
    plot_roi_contact_sheet(roi_seq, cam, output_dir / f"{cam}_combined_roi.png")
    plot_similarity_contact_sheet(similarity_seq, cam, output_dir / f"{cam}_cosine_similarity.png")

    plot_heatmap_panel(
        roi_seq.mean(axis=0),
        similarity_seq.mean(axis=0),
        weighted_sim_seq.mean(axis=0),
        cam,
        output_dir / f"{cam}_heatmap_panel.png",
    )
    plot_scores_over_time(score_seq, cam, output_dir / f"{cam}_scores_over_time.png")
    plot_aggregation_bar(cam_aggregates, cam, output_dir / f"{cam}_aggregation_comparison.png")

    return cam_aggregates


def compare_run(real_dir: Path, sim_dir: Path, output_dir: Path):
    scores_by_cam = {}

    for cam in CAMERAS:
        try:
            scores_by_cam[cam] = camera_score(real_dir, sim_dir, cam, output_dir)
        except FileNotFoundError as e:
            print(f"  [SKIP] {cam}: {e}")

    if not scores_by_cam:
        raise RuntimeError("No camera data found on either side.")

    plot_camera_comparison(scores_by_cam, output_dir / "camera_comparison.png")

    timestamp = datetime.now().isoformat(timespec="seconds")
    results = {
        "timestamp": timestamp,
        "real_dir": str(real_dir),
        "sim_dir": str(sim_dir),
        "score_definition": "per-frame: sum(weights_t * cosine_sim(real_features[t], sim_features[t])), "
                            "weights_t = normalised roi_seq[t] (full, unmasked). Per-camera number collapses "
                            "the per-frame sequence 6 ways: mean/median/max/p95/rms/activity_weighted. "
                            "Cameras are NOT combined into a single number.",
        "per_camera_aggregates": scores_by_cam,
    }
    with open(output_dir / "results_per_camera.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {output_dir / 'results_per_camera.json'}")

    return scores_by_cam


if __name__ == "__main__":
    scores_by_cam = compare_run(
        Path("dino_features/temporal_frames/real_stride_30"),
        Path("dino_features/temporal_frames/sim_gn_stride_30"),
        Path("dino_features/temporal_frames/real_vs_sim_gn_30_roi_weighted_cosine"),
    )

    print("\nPer-camera scores, all methods:")
    for cam, agg in scores_by_cam.items():
        print(f"  {cam}:")
        for method in AGG_METHODS:
            print(f"    {AGG_LABELS[method]}: {agg[method]:.4f}")
            
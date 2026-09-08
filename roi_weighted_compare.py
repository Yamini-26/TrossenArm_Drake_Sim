"""
ROI-weighted real-vs-sim comparison — per-camera only, full (unmasked) ROI weighting.

Pipeline per camera:

  1. roi_seq   = max(real_seq, sim_seq)     # (T, grid, grid), per frame, full sequence
  2. delta_seq = |real_seq - sim_seq|       # (T, grid, grid), per frame, full sequence

  3. For each frame t:
         weights_t = roi_seq[t] normalized to sum to 1 over patches (no masking)
         score_t   = sum_ij( weights_t[i,j] * delta_seq[t,i,j] )
     -> score_seq is a (T,) array: one weighted-mismatch number per frame.

  4. score_seq is collapsed into ONE camera-level result SIX different ways, ALL
     saved side by side (no single number is picked as "the" answer):
         mean, median, max, p95 (95th percentile), rms, activity_weighted
     "activity_weighted" weights each frame's score by that frame's own raw
     activity (sum of roi_seq[t]), so near-static frames barely count.

  This stops at the per-camera level. Cameras are NOT combined into a single
  run-level number here — you get 4 independent camera results, each with 6
  aggregate numbers, and you decide what to do with the 4 of them.

Visualizations, per camera:
  - roi_contact_sheet   : sampled frames of the raw combined ROI over time
  - delta_contact_sheet : sampled frames of the raw |real-sim| delta over time
  - heatmap_panel       : 3 time-averaged maps side by side — mean ROI (where
                           activity happens), mean delta (where real/sim differ
                           on average), and mean weighted-delta (where activity
                           AND disagreement overlap — this is what the score
                           is actually built from)
  - scores_over_time    : the raw (T,) per-frame score line
  - aggregation_bar      : the 6 collapsed numbers side by side, for this camera

Plus a single run-level chart, camera x method, purely for eyeballing the 4
cameras next to each other — this does NOT compute or save any combined
cross-camera number.
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
# Core math
# --------------------------------------------------------------------------- #

def frame_weights(roi_frame: np.ndarray) -> np.ndarray:
    """Normalize a single (grid, grid) ROI frame to weights summing to 1. No masking."""
    total = roi_frame.sum()
    if total > 0:
        return roi_frame / total
    return np.ones_like(roi_frame) / roi_frame.size


def per_frame_scores(roi_seq: np.ndarray, delta_seq: np.ndarray):
    """
    Returns:
      score_seq          : (T,) weighted-delta score per frame
      weighted_delta_seq : (T, grid, grid) weights_t * delta_seq[t], per frame (for heatmap)
    """
    T = roi_seq.shape[0]
    score_seq = np.empty(T, dtype=float)
    weighted_delta_seq = np.empty_like(delta_seq)
    for t in range(T):
        w_t = frame_weights(roi_seq[t])
        wd = w_t * delta_seq[t]
        weighted_delta_seq[t] = wd
        score_seq[t] = float(wd.sum())
    return score_seq, weighted_delta_seq


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
# Plotting helpers
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


def plot_delta_contact_sheet(delta_seq: np.ndarray, cam: str, output_path: Path):
    _contact_sheet(delta_seq, f"{cam}: |real - sim| delta, per frame", output_path)


def plot_heatmap_panel(roi_mean_map, delta_mean_map, weighted_mean_map, cam: str, output_path: Path):
    """3 time-averaged maps: where activity is, where real/sim differ, and where the
    score is actually coming from (activity-weighted disagreement)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("Mean ROI\n(where activity happens)", roi_mean_map),
        ("Mean |real - sim| delta\n(where they differ, unweighted)", delta_mean_map),
        ("Mean weighted delta\n(= what the score sums; activity AND disagreement)", weighted_mean_map),
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
    ax.set_ylabel("ROI-weighted |real - sim| score")
    ax.set_title(f"{cam}: per-frame weighted mismatch score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_aggregation_bar(cam_aggregates: dict, cam: str, output_path: Path):
    """Single bar chart: the 6 collapsed numbers for this one camera."""
    labels = [AGG_LABELS[m] for m in AGG_METHODS]
    vals = [cam_aggregates[m] for m in AGG_METHODS]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color="steelblue")
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=2)
    ax.set_ylabel("Score")
    ax.set_title(f"{cam}: aggregation methods compared")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


def plot_camera_comparison(scores_by_cam: dict, output_path: Path):
    """Run-level (view-only): camera x method grouped bars. No combination is computed here."""
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
    ax.set_ylabel("Score")
    ax.set_title("Per-camera scores, all 6 aggregation methods (cameras NOT combined)")
    ax.legend(title="Aggregation", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")


# --------------------------------------------------------------------------- #
# Per-camera computation
# --------------------------------------------------------------------------- #

def camera_score(real_dir: Path, sim_dir: Path, cam: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    real_seq = np.load(real_dir / f"{cam}_temporal_diff_seq.npy")
    sim_seq = np.load(sim_dir / f"{cam}_temporal_diff_seq.npy")
    assert real_seq.shape == sim_seq.shape, f"{cam}: frame mismatch {real_seq.shape} vs {sim_seq.shape}"

    roi_seq = np.maximum(real_seq, sim_seq)     # (T, grid, grid), full sequence
    delta_seq = np.abs(real_seq - sim_seq)      # (T, grid, grid), full sequence

    np.save(output_dir / f"{cam}_combined_roi_seq.npy", roi_seq)
    np.save(output_dir / f"{cam}_delta_seq.npy", delta_seq)
    print(f"  Saved -> {output_dir / f'{cam}_combined_roi_seq.npy'}  (shape={roi_seq.shape})")
    print(f"  Saved -> {output_dir / f'{cam}_delta_seq.npy'}  (shape={delta_seq.shape})")

    plot_roi_contact_sheet(roi_seq, cam, output_dir / f"{cam}_combined_roi.png")
    plot_delta_contact_sheet(delta_seq, cam, output_dir / f"{cam}_abs_diff.png")

    frame_activity = roi_seq.sum(axis=(1, 2))   # (T,) raw activity per frame, for activity_weighted agg

    score_seq, weighted_delta_seq = per_frame_scores(roi_seq, delta_seq)
    cam_aggregates = aggregate_scores(score_seq, frame_activity)
    agg_str = "  ".join(f"{AGG_LABELS[m]}={cam_aggregates[m]:.4f}" for m in AGG_METHODS)
    print(f"  {cam}: {agg_str}")

    np.save(output_dir / f"{cam}_per_frame_scores.npy", score_seq)
    print(f"  Saved -> {output_dir / f'{cam}_per_frame_scores.npy'}")

    plot_heatmap_panel(
        roi_seq.mean(axis=0), delta_seq.mean(axis=0), weighted_delta_seq.mean(axis=0),
        cam, output_dir / f"{cam}_heatmap_panel.png",
    )
    plot_scores_over_time(score_seq, cam, output_dir / f"{cam}_scores_over_time.png")
    plot_aggregation_bar(cam_aggregates, cam, output_dir / f"{cam}_aggregation_comparison.png")

    return cam_aggregates


def compare_run(real_dir: Path, sim_dir: Path, output_dir: Path):
    scores_by_cam = {}

    for cam in CAMERAS:
        try:
            scores_by_cam[cam] = camera_score(real_dir, sim_dir, cam, output_dir)
        except FileNotFoundError:
            print(f"  [SKIP] {cam}: no diff_seq found on one side")

    if not scores_by_cam:
        raise RuntimeError("No camera diff_seq files found on either side.")

    plot_camera_comparison(scores_by_cam, output_dir / "camera_comparison.png")

    timestamp = datetime.now().isoformat(timespec="seconds")
    results = {
        "timestamp": timestamp,
        "real_dir": str(real_dir),
        "sim_dir": str(sim_dir),
        "score_definition": "per-frame: sum(weights_t * |real_seq[t]-sim_seq[t]|), weights_t = "
                             "normalized roi_seq[t] (full, unmasked). Per-camera number collapses "
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
        Path("dino_features/temporal_frames/sim_dn_stride_30"),
        Path("dino_features/temporal_frames/real_vs_sim_dn_30_roi_weighted"),
    )

    print("\nPer-camera scores, all methods:")
    for cam, agg in scores_by_cam.items():
        print(f"  {cam}:")
        for method in AGG_METHODS:
            print(f"    {AGG_LABELS[method]}: {agg[method]:.4f}")

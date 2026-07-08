#!/usr/bin/env python
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# Colours
BLUE   = "#3B82F6"
RED    = "#EF4444"
GREEN  = "#10B981"
AMBER  = "#F59E0B"
PURPLE = "#8B5CF6"
GRAY   = "#9CA3AF"

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def load_state_scalars(state_report_path: Path) -> dict:
    """Load mean RMSE (mm) from state report."""
    with open(state_report_path, "r") as f:
        data = json.load(f)
    # The report has 'combined_rmse' -> 'mean_rmse_mm'
    mean_rmse = data.get("combined_rmse", {}).get("mean_rmse_mm", None)
    max_rmse  = data.get("combined_rmse", {}).get("max_rmse_mm", None)
    return {"mean_rmse_mm": mean_rmse, "max_rmse_mm": max_rmse}


def load_state_timeseries(state_npz_path: Path) -> dict:
    """Load RMSE per timestep from trajectory_states.npz."""
    data = np.load(state_npz_path)
    # If you saved the RMSE in the NPZ, you can load it directly.
    # Otherwise, we need to recompute it from EE and cube positions.
    # For simplicity, assume the NPZ contains 'times', 'rmse_per_t_mm'.
    # If not, we compute it here (requires ee_pos, obj_pos).
    if "rmse_per_t_mm" in data:
        times = data["times"]
        rmse = data["rmse_per_t_mm"]
    else:
        # Fallback: recompute from EE and cube positions
        # (this assumes you have ee_pos and obj_pos in the npz)
        ee_pos = data["ee_pos"]      # (T, 3) metres
        obj_pos = data["obj_pos"]    # (T, 3) metres
        dee = ee_pos  # Difference not computed here
        dobj = obj_pos
        # Actually we need the difference between two sims; this NPZ only has one.
        # So we can't recompute RMSE here if we only have one NPZ.
        # We'll rely on the report scalar.
        print("  [WARN] rmse_per_t_mm not found in NPZ. "
              "Cannot plot time series without aligning two NPZs.")
        return None
    return {"times": times, "rmse": rmse}


def load_energy_scalars(energy_report_path: Path) -> dict:
    """Load energy distance per camera from energy report."""
    with open(energy_report_path, "r") as f:
        data = json.load(f)
    # data is {cam_name: {energy_distance: ..., ...}}
    return {cam: info.get("energy_distance", None) for cam, info in data.items()}


def load_cos_sim(energy_output_dir: Path, cameras: list) -> dict:
    """Load per-frame cosine similarity .npy files."""
    cos_dict = {}
    for cam in cameras:
        fpath = energy_output_dir / f"cos_sim_{cam}.npy"
        if fpath.exists():
            cos_dict[cam] = np.load(fpath)
        else:
            print(f"  [WARN] {fpath} not found – skipping time series for {cam}")
    return cos_dict


def plot_grouped_bars(state_rmse_dict: dict, energy_dist_dict: dict, output_dir: Path):
    """Grouped bar chart for direct per‑camera comparison."""
    cams = sorted(set(state_rmse_dict.keys()) & set(energy_dist_dict.keys()))
    if not cams:
        return

    x = np.arange(len(cams))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, [state_rmse_dict[c] for c in cams], w,
                   label="State RMSE (mm)", color=BLUE, alpha=0.8)
    bars2 = ax.bar(x + w/2, [energy_dist_dict[c] for c in cams], w,
                   label="Energy Distance", color=RED, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(cams)
    ax.set_ylabel("Metric value")
    ax.set_title("Per‑Camera Comparison: Physics vs Visual Gap")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    _save(fig, output_dir / "grouped_bars_state_vs_energy.png")


def plot_timeseries_overlay(state_timeseries: dict, cos_sim_dict: dict,
                            output_dir: Path):
    """
    Time‑series overlay: State RMSE (left axis) vs Cosine Similarity (right axis).
    One subplot per camera.
    """
    if state_timeseries is None or not cos_sim_dict:
        print("  [SKIP] Time‑series overlay: missing data.")
        return

    times = state_timeseries["times"]
    rmse  = state_timeseries["rmse"]   # (T,) mm

    # Align cameras that have cos_sim
    cams = sorted(cos_sim_dict.keys())
    n = len(cams)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, cam in zip(axes, cams):
        cos_sim = cos_sim_dict[cam]
        # Trim to same length as RMSE (if shorter)
        N = min(len(times), len(cos_sim))
        t = times[:N]
        rmse_trim = rmse[:N]
        cos_trim = cos_sim[:N]

        # Left axis: RMSE
        ax1 = ax
        color1 = BLUE
        ax1.plot(t, rmse_trim, color=color1, lw=1.5, label="State RMSE")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("RMSE (mm)", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(alpha=0.3)

        # Right axis: Cosine similarity
        ax2 = ax.twinx()
        color2 = RED
        ax2.plot(t, cos_trim, color=color2, lw=1.5, label="Cosine Similarity")
        ax2.set_ylabel("Cosine Similarity", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0.6, 1.05)  # typical range for CLS features

        # Title
        ax1.set_title(f"{cam} — State RMSE vs Cosine Similarity over Time")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, output_dir / "timeseries_state_vs_cos.png")

def plot_global_bars(state_rmse_mean: float, energy_dist_dict: dict,
                     mean_cos_dict: dict, output_dir: Path):
    """
    Side-by-side bar chart showing:
      - Global mean State RMSE (same for all cameras)
      - Mean Cosine Similarity per camera
      - Energy Distance per camera
    """
    cams = sorted(energy_dist_dict.keys())
    if not cams:
        return

    x = np.arange(len(cams))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar 1: Global State RMSE (repeated for each camera)
    bars1 = ax.bar(x - w, [state_rmse_mean] * len(cams), w,
                   label=f"State RMSE (mean = {state_rmse_mean:.2f} mm)", color=BLUE, alpha=0.8)

    # Bar 2: Mean Cosine Similarity per camera
    mean_cos_vals = [mean_cos_dict.get(c, 0) for c in cams]
    bars2 = ax.bar(x, mean_cos_vals, w,
                   label="Mean Cosine Similarity", color=GREEN, alpha=0.8)

    # Bar 3: Energy Distance per camera
    ed_vals = [energy_dist_dict[c] for c in cams]
    # Scale ED to roughly match the visual range (e.g., multiply by 100 for percentage-like display)
    # ed_scaled = [v * 100 for v in ed_vals]
    bars3 = ax.bar(x + w, ed_vals, w,
                   label="Energy Distance × 100", color=RED, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(cams)
    ax.set_ylabel("Metric value")
    ax.set_title("Global Comparison: Physics Gap vs Visual Gap per Camera")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar, ed in zip(bars3, ed_vals):
        height = bar.get_height()
        ax.annotate(f"{ed:.4f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, output_dir / "global_comparison_bars.png")


def main():
    parser = argparse.ArgumentParser(
        description="Correlate state‑space (physics) RMSE with visual energy distance / cosine similarity."
    )
    parser.add_argument("--state_dir", required=True,
                        help="Directory containing state_comparison_report.json "
                             "and (optionally) trajectory_states.npz")
    parser.add_argument("--energy_dir", required=True,
                        help="Directory containing energy_comparison_report.json "
                             "and cos_sim_<cam>.npy files (from compare_energy.py)")
    parser.add_argument("--output", default="state_vs_energy_output",
                        help="Output directory for correlation plots")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    energy_dir = Path(args.energy_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("STATE vs ENERGY CORRELATION")
    print(f"  State dir  : {state_dir}")
    print(f"  Energy dir : {energy_dir}")
    print(f"  Output     : {output_dir}\n")

    # Load state report (scalar RMSE)
    state_report = state_dir / "state_comparison_report.json"
    if not state_report.exists():
        print(f"  [ERROR] state_comparison_report.json not found in {state_dir}")
        return

    state_scalars = load_state_scalars(state_report)
    # state_scalars has only mean/max, not per‑camera.
    # Actually, state RMSE is a single scalar (global), not per‑camera.
    # We need to assign the same RMSE to all cameras for bar/scatter plots.
    # This is an important limitation: state RMSE is global; energy is per‑camera.
    print(f"  Global State RMSE (mean): {state_scalars['mean_rmse_mm']:.2f} mm")

    # Load energy report (per‑camera energy distance)
    energy_report = energy_dir / "energy_comparison_report.json"
    if not energy_report.exists():
        print(f"  [ERROR] energy_comparison_report.json not found in {energy_dir}")
        return

    energy_scalars = load_energy_scalars(energy_report)
    print(f"  Loaded energy distances for cameras: {list(energy_scalars.keys())}")

    # Build per‑camera state RMSE (same scalar for all cameras)
    state_per_cam = {cam: state_scalars["mean_rmse_mm"] for cam in energy_scalars.keys()}

    # Load time‑series RMSE (if available)
    state_npz = state_dir / "trajectory_states.npz"
    state_ts = None
    if state_npz.exists():
        state_ts = load_state_timeseries(state_npz)
        if state_ts is None:
            print("  [WARN] Could not load time‑series RMSE from NPZ.")
    else:
        print("  [WARN] trajectory_states.npz not found – time‑series overlay will be skipped.")

    # Load cosine similarity files
    cos_sim_dict = load_cos_sim(energy_dir, list(energy_scalars.keys()))

    # Generate plots

    # Grouped bar chart (per camera)
    plot_grouped_bars(state_per_cam, energy_scalars, output_dir)

    # Time‑series overlay (if both exist)
    if state_ts is not None and cos_sim_dict:
        plot_timeseries_overlay(state_ts, cos_sim_dict, output_dir)

    #  Add a line plot of global RMSE over time on its own
    if state_ts is not None:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(state_ts["times"], state_ts["rmse"], color=BLUE, lw=2)
        ax.fill_between(state_ts["times"], state_ts["rmse"], alpha=0.2, color=BLUE)
        ax.axhline(state_scalars["mean_rmse_mm"], color=RED, linestyle="--",
                   label=f"mean = {state_scalars['mean_rmse_mm']:.2f} mm")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Combined RMSE (mm)")
        ax.set_title("Global State RMSE over Time")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, output_dir / "state_rmse_timeseries.png")
    
    # Load mean cosine similarity per camera from the energy report
    with open(energy_report, "r") as f:
        energy_data = json.load(f)
    
    mean_cos_dict = {
        cam: info.get("mean_cos_sim", 0) 
        for cam, info in energy_data.items()
    }

    # Generate the global comparison plot
    plot_global_bars(
        state_rmse_mean=state_scalars["mean_rmse_mm"],
        energy_dist_dict=energy_scalars,
        mean_cos_dict=mean_cos_dict,
        output_dir=output_dir
    )

    # Print interpretation
    print("\n" + "="*60)
    print("CORRELATION INTERPRETATION")
    print("="*60)
    print(f"  Global mean state RMSE : {state_scalars['mean_rmse_mm']:.2f} mm")
    print(f"  Global max state RMSE  : {state_scalars['max_rmse_mm']:.2f} mm")
    print("\n  Energy distances per camera:")
    for cam, ed in energy_scalars.items():
        print(f"    {cam:20s} : {ed:.6f}")

    print("\n  How to interpret:")
    print("  - If energy distance is HIGH but state RMSE is LOW  → differences are mainly visual (lighting, textures).")
    print("  - If BOTH are HIGH                                 → real physical divergence (motion + visual).")
    print("  - If state RMSE is HIGH but energy distance is LOW → physics changed but camera can't see it.")
    print("  - Time‑series overlay shows if spikes in RMSE correspond to drops in cosine similarity.")


if __name__ == "__main__":
    main()

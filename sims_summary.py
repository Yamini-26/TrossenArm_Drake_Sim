#!/usr/bin/env python

# Usage:
# python sims_summary.py --dirs \\
# baseline_vs_drop_1785155063 \\
# baseline_vs_fast_1785155063 \\
# baseline_vs_heavy_1785155063 \\
# baseline_vs_shifted_1785155063 \\
# baseline_vs_weak_1785155063 \\
# --output state_vs_cosine_output/summary_across_sims

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re


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


# Discovery / loading

# Matches e.g. "baseline_vs_drop_1785155063" -> variant group "drop"
_VARIANT_RE = re.compile(r"^baseline_vs_(.+?)(?:_\d+)?$")


def discover_comparison_dirs(root: Path) -> list:
    """Find every immediate subdirectory of `root` starting with 'baseline_vs_'."""
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith("baseline_vs_")
    )


def variant_name(comp_dir: Path) -> str:
    """
    Extract a clean variant label from a directory named like
    'baseline_vs_<variant>_<timestamp>' (timestamp optional), e.g.
    'baseline_vs_drop_1785155063' -> 'drop'.
    Falls back to the raw directory name if it doesn't match the pattern.
    """
    name = comp_dir.name
    m = _VARIANT_RE.match(name)
    return m.group(1) if m else name


def load_state_rmse(comp_dir: Path) -> dict:
    """Load {mean_rmse_mm, max_rmse_mm} from state_comparison_report.json."""
    report_path = Path("state_comparison") / comp_dir / "state_comparison_report.json"
    if not report_path.exists():
        print(f"  [WARN] missing {report_path}")
        return None
    with open(report_path, "r") as f:
        data = json.load(f)
    rmse = data.get("combined_rmse", {})
    mean_rmse = rmse.get("mean_rmse_mm")
    max_rmse  = rmse.get("max_rmse_mm")
    if mean_rmse is None:
        print(f"  [WARN] {report_path} has no combined_rmse.mean_rmse_mm")
        return None
    return {"mean_rmse_mm": mean_rmse, "max_rmse_mm": max_rmse}


def load_energy(comp_dir: Path) -> dict:
    """Load {camera_name: energy_distance} from energy_comparison_report.json."""
    report_path = Path("energy_comparison") / comp_dir / "energy_comparison_report.json"
    if not report_path.exists():
        print(f"  [WARN] missing {report_path}")
        return {}
    with open(report_path, "r") as f:
        data = json.load(f)
    return {
        cam: info.get("energy_distance")
        for cam, info in data.items()
        if info.get("energy_distance") is not None
    }


def load_state_timeseries(comp_dir: Path):
    """
    Load per-timestep combined RMSE (mm) for a variant, if available.

    Looks for, in order:
      1. state_comparison/<comp_dir>/rmse_per_t_mm.npy   (plain array; sample index used as time)
      2. state_comparison/<comp_dir>/trajectory_states.npz with a 'rmse_per_t_mm' key

    NOTE: the sim runner's trajectory_states.npz (as currently written) only stores
    ee_pos / obj_pos / q / times per single run — not a ready-made per-timestep RMSE
    against baseline. If your state_comparison step writes that array under a
    different filename/key, update the paths below to match.
    """
    base = Path("state_comparison") / comp_dir
    npy_path = base / "rmse_per_t_mm.npy"
    npz_path = base / "trajectory_states.npz"

    if npy_path.exists():
        rmse = np.load(npy_path)
        times = np.arange(len(rmse))
        return {"times": times, "rmse": rmse}

    if npz_path.exists():
        data = np.load(npz_path)
        if "rmse_per_t_mm" in data:
            return {"times": data["times"], "rmse": data["rmse_per_t_mm"]}

    print(f"  [WARN] no per-timestep RMSE found for {comp_dir} "
          f"(looked for {npy_path}, and {npz_path} with key 'rmse_per_t_mm')")
    return None


def load_cos_sim_timeseries(comp_dir: Path, cameras: list) -> dict:
    """Load per-frame cosine similarity .npy files for a variant."""
    base = Path("energy_comparison") / comp_dir
    cos_dict = {}
    for cam in cameras:
        fpath = base / f"cos_sim_{cam}.npy"
        if fpath.exists():
            cos_dict[cam] = np.load(fpath)
        else:
            print(f"  [WARN] {fpath} not found – skipping cos-sim time series for {cam}")
    return cos_dict


# Plots

def plot_rmse_bar(summary: dict, output_dir: Path):
    """Bar chart: mean + max combined RMSE (mm), one bar pair per variant."""
    variants = [v for v in summary if summary[v]["state"] is not None]
    if not variants:
        print("  [SKIP] No state RMSE data available for any variant")
        return

    means = [summary[v]["state"]["mean_rmse_mm"] for v in variants]
    maxs  = [summary[v]["state"]["max_rmse_mm"]  for v in variants]

    x = np.arange(len(variants))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 1.4), 6))
    bars1 = ax.bar(x - w/2, means, w, label="Mean RMSE", color=BLUE, alpha=0.85)
    bars2 = ax.bar(x + w/2, maxs,  w, label="Max RMSE",  color=RED,  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel("Combined RMSE (mm)")
    ax.set_title("Combined State RMSE per Variant (vs baseline)\n"
                 "sqrt(||ΔEE||² + ||Δobj||²), unweighted")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, output_dir / "summary_rmse_per_variant.png")


def plot_energy_grouped(summary: dict, output_dir: Path):
    """Grouped bar chart: energy distance per variant, one group of bars per camera."""
    variants = [v for v in summary if summary[v]["energy"]]
    if not variants:
        print("  [SKIP] No energy distance data available for any variant")
        return

    all_cams = sorted({cam for v in variants for cam in summary[v]["energy"].keys()})
    if not all_cams:
        print("  [SKIP] No cameras found across energy reports")
        return

    x = np.arange(len(variants))
    n_cams = len(all_cams)
    w = 0.8 / n_cams
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_cams, 1)))

    fig, ax = plt.subplots(figsize=(max(10, len(variants) * 1.6), 6))
    for i, cam in enumerate(all_cams):
        vals = [summary[v]["energy"].get(cam, np.nan) for v in variants]
        offset = (i - (n_cams - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=cam, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel("Energy Distance  (0 = identical distributions)")
    ax.set_title("Energy Distance per Variant, Grouped by Camera (vs baseline)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, output_dir / "summary_energy_per_variant.png")


def _downsample_to_frames(times: np.ndarray, rmse: np.ndarray, n_frames: int):
    """
    Resample a per-timestep array (times, rmse) down to n_frames evenly-spaced
    samples across the FULL trajectory, so it aligns with a per-frame quantity
    (e.g. cosine similarity) that was computed at a much lower rate than the
    underlying simulation.

    Using array[:n_frames] instead of this would just take the first n_frames
    raw simulation steps -- a tiny sliver at the very start of the trajectory --
    rather than samples spread across its full duration.
    """
    if len(rmse) == n_frames:
        return times, rmse
    idx = np.linspace(0, len(rmse) - 1, num=n_frames).round().astype(int)
    return times[idx], rmse[idx]


def plot_combined_timeseries(ts_summary: dict, output_dir: Path):
    """
    One figure PER CAMERA, with every variant overlaid as its own colored line:
      - top subplot:    RMSE(t)  (mm)          for every variant
      - bottom subplot: cosine similarity(t)   for every variant, same colors

    ts_summary: {variant: {"state_ts": {"times":..., "rmse":...} or None,
                            "cos_sim": {cam: array}}}
    """
    variants = [v for v, d in ts_summary.items() if d["state_ts"] is not None and d["cos_sim"]]
    if not variants:
        print("  [SKIP] Combined timeseries: no variant has both state and cos-sim time series")
        return

    all_cams = sorted({cam for v in variants for cam in ts_summary[v]["cos_sim"].keys()})
    if not all_cams:
        print("  [SKIP] Combined timeseries: no cameras found")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 1)))
    color_map = dict(zip(variants, colors))

    for cam in all_cams:
        fig, (ax_rmse, ax_cos) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        for v in variants:
            cos_sim_dict = ts_summary[v]["cos_sim"]
            if cam not in cos_sim_dict:
                continue
            state_ts = ts_summary[v]["state_ts"]
            times, rmse = state_ts["times"], state_ts["rmse"]
            cos_sim = cos_sim_dict[cam]
            t, rmse_trim = _downsample_to_frames(times, rmse, len(cos_sim))
            cos_trim = cos_sim
            color = color_map[v]

            ax_rmse.plot(t, rmse_trim, color=color, lw=1.6, label=v)
            ax_cos.plot(t, cos_trim, color=color, lw=1.6, label=v)

        ax_rmse.set_ylabel("State RMSE (mm)")
        ax_rmse.set_title(f"{cam} — State RMSE over Time, All Variants (vs baseline)")
        ax_rmse.grid(alpha=0.3)
        ax_rmse.legend(fontsize=8, ncol=2)

        ax_cos.set_xlabel("Time (s)")
        ax_cos.set_ylabel("Cosine Similarity")
        ax_cos.set_ylim(0.6, 1.05)
        ax_cos.set_title(f"{cam} — Cosine Similarity over Time, All Variants (vs baseline)")
        ax_cos.grid(alpha=0.3)
        ax_cos.legend(fontsize=8, ncol=2)

        fig.tight_layout()
        _save(fig, output_dir / f"summary_timeseries_{cam}.png")


def plot_combined_timeseries_avg_cos(ts_summary: dict, output_dir: Path):
    """
    ONE figure (not per-camera): every variant overlaid, where the cosine
    similarity trace is the mean across all cameras available for that variant.
      - top subplot:    RMSE(t)                                   for every variant
      - bottom subplot: mean cosine similarity(t) (avg over cams)  for every variant

    ts_summary: {variant: {"state_ts": {"times":..., "rmse":...} or None,
                            "cos_sim": {cam: array}}}
    """
    variants = [v for v, d in ts_summary.items() if d["state_ts"] is not None and d["cos_sim"]]
    if not variants:
        print("  [SKIP] Averaged cos-sim timeseries: no variant has both state and cos-sim time series")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 1)))
    color_map = dict(zip(variants, colors))

    fig, (ax_rmse, ax_cos) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for v in variants:
        state_ts = ts_summary[v]["state_ts"]
        times, rmse = state_ts["times"], state_ts["rmse"]

        cos_sim_dict = ts_summary[v]["cos_sim"]
        cam_arrays = list(cos_sim_dict.values())
        # Trim every camera's series to the shortest one so they stack cleanly,
        # then average across cameras to get a single cos-sim(t) curve.
        min_cam_len = min(len(a) for a in cam_arrays)
        stacked = np.stack([a[:min_cam_len] for a in cam_arrays], axis=0)
        avg_cos = stacked.mean(axis=0)

        N = min(min_cam_len, len(times), len(rmse))
        t, rmse_trim = _downsample_to_frames(times, rmse, N)
        cos_trim = avg_cos[:N]
        color = color_map[v]

        ax_rmse.plot(t, rmse_trim, color=color, lw=1.6, label=v)
        ax_cos.plot(t, cos_trim, color=color, lw=1.6, label=v)

    ax_rmse.set_ylabel("State RMSE (mm)")
    ax_rmse.set_title("State RMSE over Time, All Variants (vs baseline)")
    ax_rmse.grid(alpha=0.3)
    ax_rmse.legend(fontsize=8, ncol=2)

    ax_cos.set_xlabel("Time (s)")
    ax_cos.set_ylabel("Mean Cosine Similarity (avg over cameras)")
    ax_cos.set_ylim(0.6, 1.05)
    ax_cos.set_title("Cosine Similarity over Time — Averaged Across Cameras, All Variants (vs baseline)")
    ax_cos.grid(alpha=0.3)
    ax_cos.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    _save(fig, output_dir / "summary_timeseries_avg_cos.png")


# Report

def save_json_summary(summary: dict, output_dir: Path):
    out = output_dir / "summary_across_sims.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → {out}")
    return out


def print_table(summary: dict):
    print("\nSUMMARY ACROSS SIMS (vs baseline)")
    for v, info in summary.items():
        state = info["state"]
        if state is not None:
            print(f"  {v:25s} mean_rmse={state['mean_rmse_mm']:.2f} mm  "
                  f"max_rmse={state['max_rmse_mm']:.2f} mm")
        else:
            print(f"  {v:25s} [no state RMSE data]")
        for cam, e in info["energy"].items():
            print(f"      {cam:20s} energy={e:.4f}")
        if not info["energy"]:
            print("      [no energy distance data]")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate RMSE and energy distance across multiple "
                     "<variant>_vs_baseline comparison directories."
    )
    parser.add_argument("--root", default="comparisons",
                        help="Directory containing '*_vs_baseline' subfolders "
                             "to auto-discover (default: comparisons)")
    parser.add_argument("--dirs", nargs="+", default=None,
                        help="Explicit list of comparison directories. "
                             "Overrides --root auto-discovery if given.")
    parser.add_argument("--output", default="summary_across_sims",
                        help="Output directory for plots and JSON summary")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dirs:
        comp_dirs = [Path(d) for d in args.dirs]
    else:
        comp_dirs = discover_comparison_dirs(Path(args.root))

    if not comp_dirs:
        print(f"No comparison directories found "
              f"(root='{args.root}', dirs={args.dirs}). Nothing to summarize.")
        return

    print("SUMMARIZE ACROSS SIMS")
    print(f"  Found {len(comp_dirs)} comparison directories:")
    for d in comp_dirs:
        print(f"    {d}")

    summary = {}
    ts_summary = {}
    for d in comp_dirs:
        vname = variant_name(d)
        print(f"\n[{vname}]")
        state        = load_state_rmse(d)
        energy       = load_energy(d)
        summary[vname] = {
            "state":      state,
            "energy":     energy,
            "source_dir": str(d),
        }

        state_ts = load_state_timeseries(d)
        cos_sim  = load_cos_sim_timeseries(d, list(energy.keys()))
        ts_summary[vname] = {"state_ts": state_ts, "cos_sim": cos_sim}

    print("\nGenerating plots ...")
    plot_rmse_bar(summary, output_dir)
    plot_energy_grouped(summary, output_dir)
    plot_combined_timeseries(ts_summary, output_dir)
    plot_combined_timeseries_avg_cos(ts_summary, output_dir)

    save_json_summary(summary, output_dir)
    print_table(summary)

    print(f"\nAll outputs in: {output_dir}/")
    print("  summary_rmse_per_variant.png       ← combined RMSE, all variants")
    print("  summary_energy_per_variant.png     ← energy distance, grouped by camera")
    print("  summary_timeseries_<camera>.png    ← RMSE(t) + cos-sim(t), all variants overlaid, per camera")
    print("  summary_timeseries_avg_cos.png     ← RMSE(t) + mean cos-sim(t) averaged over cameras, all variants overlaid")
    print("  summary_across_sims.json           ← raw numbers behind all plots")


if __name__ == "__main__":
    main()

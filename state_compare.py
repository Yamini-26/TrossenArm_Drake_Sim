#!/usr/bin/env python
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# Colours
BLUE   = "#3B82F6"
RED    = "#EF4444"
GREEN  = "#10B981"
AMBER  = "#F59E0B"
PURPLE = "#8B5CF6"
GRAY   = "#9CA3AF"
AXIS_C = ["#EF4444", "#10B981", "#3B82F6"]   # X / Y / Z

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})


# Loading and alignment

def load_trajectory(traj_dir: Path) -> dict:
    npz = traj_dir / "trajectory_states.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"trajectory_states.npz not found in {traj_dir}\n"
            f"Make sure simulate.py saves the log using the snippet in its docstring."
        )
    d = np.load(npz)
    print(f"  Loaded {len(d['times'])} timesteps from '{traj_dir.name}/'")
    return {k: d[k] for k in d.files}


def align(a: dict, b: dict) -> tuple:
    """Trim both to the same number of timesteps (shorter)."""
    N = min(len(a["times"]), len(b["times"]))
    print(f"  Aligned to {N} timesteps  (A={len(a['times'])}, B={len(b['times'])})")
    return (
        {k: v[:N] for k, v in a.items()},
        {k: v[:N] for k, v in b.items()},
    )


# Comparison — point-by-point

def compare(a: dict, b: dict) -> dict:
    """
    Point-by-point state comparison at every aligned timestep.

    Joint angles q:
        dq        (T, nq)  signed error per joint (rad)
        dq_abs    (T, nq)  absolute error per joint (rad)
        dq_norm   (T,)     Euclidean norm across all joints (rad)

    End-effector position (metres → mm for display):
        dee       (T, 3)   signed per-axis error (mm)
        dee_dist  (T,)     Euclidean distance (mm)

    Object position (metres → mm for display):
        dobj      (T, 3)   signed per-axis error (mm)
        dobj_dist (T,)     Euclidean distance (mm)

    Quaternion angular distance (cube orientation):
        quat_angle (T,)    geodesic angle (radians)

    RMSE — combined single scalar per timestep:
        Both dee and dobj are already in metres so they are directly
        comparable. No joint conversion needed because dee already
        captures the physical consequence of joint errors in Cartesian
        space via forward kinematics.

        rmse(t) = sqrt(dee_dist_m(t)² + dobj_dist_m(t)² )

        where dee_dist_m and dobj_dist_m are in METRES
        The scalar summary is then reported in mm for readability.

    Joints are also classified by type so the plots can separate:
        - arm joints     (non-zero movement, meaningful to plot)
        - fixed joints   (zero error throughout, skip)
        - floating base  (cube quaternion + xyz, separate group)
    """
    print(f"  q shape: A={a['q'].shape}  B={b['q'].shape}")
    nq = min(a["q"].shape[1], b["q"].shape[1])

    # The cube's floating base occupies the last 7 positions of q:
    # [qw, qx, qy, qz, x, y, z]. Everything before that is arm joints.
    cube_start = nq - 7
    arm_end    = cube_start

    # Arm joint errors — plain subtraction (real angles only)
    dq_arm  = a["q"][:, :arm_end] - b["q"][:, :arm_end]   # (T, n_arm) signed
    dq_abs  = np.abs(dq_arm)                               # (T, n_arm)
    dq_norm = np.linalg.norm(dq_arm, axis=1)               # (T,)

    # EE error
    dee       = (a["ee_pos"] - b["ee_pos"]) * 1000    # (T, 3) mm
    dee_dist  = np.linalg.norm(dee, axis=1)         # (T,)   mm

    # Object (cube) position error
    dobj        = (a["obj_pos"] - b["obj_pos"]) * 1000    # (T, 3) mm
    dobj_dist   = np.linalg.norm(dobj, axis=1)          # (T,)   mm

    # Cube orientation error — geodesic angular distance (radians)
    # no component-wise subtraction of the quaternion!
    quat_indices = slice(cube_start, cube_start + 4)  # qw, qx, qy, qz
    qA_quat = a["q"][:, quat_indices]   # (T, 4)
    qB_quat = b["q"][:, quat_indices]   # (T, 4)
 
    dot = np.sum(qA_quat * qB_quat, axis=1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)   # abs() handles quaternion double-cover
    quat_angle = 2.0 * np.arccos(dot)       # (T,) radians
 
    mean_quat_angle = float(quat_angle.mean())
    max_quat_angle  = float(quat_angle.max())

    # Combined RMSE
    rmse_per_t = np.sqrt(dee_dist ** 2 + dobj_dist ** 2) # (T,)  mm

    # Scalar summaries
    mean_rmse = float(rmse_per_t.mean())
    max_rmse  = float(rmse_per_t.max())

    # Classify joints
    # A joint is "fixed" if its error is zero at every timestep.
    per_joint_mean = dq_abs.mean(axis=0) if arm_end > 0 else np.array([])  # (nq,)
    fixed_mask     = per_joint_mean < 1e-9         # True = never moves
    
    arm_joints     = [j for j in range(arm_end) if not fixed_mask[j]]
    worst_joint    = int(np.argmax(per_joint_mean)) if arm_end > 0 else None

    return {
        "t":            a["times"],
        "nq":           nq,
        "arm_end":      arm_end,
        # arm joints
        "dq_abs":       dq_abs,
        "dq_norm":      dq_norm,
        # EE
        "dee":          dee,
        "dee_dist":     dee_dist,
        # object
        "dobj":         dobj,
        "dobj_dist":    dobj_dist,
        # quaternion
        "quat_angle":          quat_angle,       # (T,) radians
        "mean_quat_angle_rad": mean_quat_angle,
        "max_quat_angle_rad":  max_quat_angle,
        # RMSE (unweighted)
        "rmse_per_t_mm": rmse_per_t,
        "mean_rmse_mm":  mean_rmse,
        "max_rmse_mm":   max_rmse,
        # joint classification
        "arm_joints":   arm_joints,
        "fixed_mask":   fixed_mask,
        # summary scalars
        "mean_joint_err_rad": float(dq_norm.mean()) if arm_end > 0 else 0.0,
        "max_joint_err_rad":  float(dq_norm.max())  if arm_end > 0 else 0.0,
        "mean_ee_mm":         float(dee_dist.mean()),
        "max_ee_mm":          float(dee_dist.max()),
        "mean_obj_mm":        float(dobj_dist.mean()),
        "max_obj_mm":         float(dobj_dist.max()),
        "worst_joint":        worst_joint,
    }


# Plots

def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# Plot 0 : Summary dashboard
 
def plot_dashboard(a, b, c, output_dir: Path):
    t = c["t"]
    T = len(t)
 
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.38,
                            height_ratios=[0.75, 1.2, 1.2])
 
    # Score cards (now includes combined RMSE)
    cards = [
        ("Mean EE error",      f"{c['mean_ee_mm']:.2f}",      "mm",          GREEN),
        ("Mean cube error",    f"{c['mean_obj_mm']:.2f}",      "mm",          AMBER),
        ("Combined RMSE mean", f"{c['mean_rmse_mm']:.2f}",     "mm",          PURPLE),
        ("Combined RMSE max",  f"{c['max_rmse_mm']:.2f}",      "mm",          RED),
    ]
    for col, (label, val, unit, color) in enumerate(cards):
        ax = fig.add_subplot(gs[0, col])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                                    facecolor=color, alpha=0.10,
                                    edgecolor=color, linewidth=2,
                                    transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 0.78, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.45, val,   ha="center", va="center",
                fontsize=19, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.18, unit,  ha="center", va="center",
                fontsize=8, color=GRAY, transform=ax.transAxes)
 
    # Combined RMSE over time
    ax = fig.add_subplot(gs[1, :2])
    ax.plot(t, c["rmse_per_t_mm"], color=PURPLE, lw=1.8, label="Combined RMSE")
    ax.fill_between(t, c["rmse_per_t_mm"], alpha=0.2, color=PURPLE)
    ax.axhline(c["mean_rmse_mm"], color=RED, lw=1.1, linestyle="--",
               label=f"mean RMSE = {c['mean_rmse_mm']:.2f} mm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (mm)")
    ax.set_title(f"Combined RMSE over time\n"
                 f"sqrt(||ΔEE||² + ||Δobj||²)  [both in metres, unweighted]")
    ax.legend(fontsize=8)
 
    # EE distance
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(t, c["dee_dist"], color=GREEN, lw=1.5)
    ax.fill_between(t, c["dee_dist"], alpha=0.2, color=GREEN)
    ax.axhline(c["mean_ee_mm"], color=RED, lw=1.1, linestyle="--",
               label=f"mean = {c['mean_ee_mm']:.2f} mm")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("||EE_A − EE_B|| (mm)")
    ax.set_title("End-effector position error magnitude"); ax.legend(fontsize=8)
 
    # Cube distance
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(t, c["dobj_dist"], color=AMBER, lw=1.5)
    ax.fill_between(t, c["dobj_dist"], alpha=0.2, color=AMBER)
    ax.axhline(c["mean_obj_mm"], color=RED, lw=1.1, linestyle="--",
               label=f"mean = {c['mean_obj_mm']:.2f} mm")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("||obj_A − obj_B|| (mm)")
    ax.set_title("Cube position error magnitude"); ax.legend(fontsize=8)
 
    # Cube Z height both sims
    ax = fig.add_subplot(gs[2, 2:])
    ax.plot(t, a["obj_pos"][:T, 2] * 1000, color=BLUE, lw=1.5, label="Sim A")
    ax.plot(t, b["obj_pos"][:T, 2] * 1000, color=RED,  lw=1.5, label="Sim B",
            linestyle="--")
    ax.fill_between(t, a["obj_pos"][:T, 2] * 1000, b["obj_pos"][:T, 2] * 1000,
                    alpha=0.15, color=AMBER, label="difference")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cube Z height (mm)")
    ax.set_title("Cube lift height — Sim A vs Sim B"); ax.legend(fontsize=8)
 
    fig.suptitle(
        "State-to-State Trajectory Comparison — Sim A vs Sim B\n"
        "(pure state-space · lighting-independent · combined RMSE in mm)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    _save(fig, output_dir / "00_summary_dashboard.png")
 
 
# Plot 1 : Arm joint trajectories
 
def _plot_joint_group(t, a_q, b_q, dq_abs, joint_indices, group_name,
                      worst_joint, output_path: Path):
    """
    Plot one group of arm joints.
    Left column: both sim trajectories overlaid.
    Right column: absolute error.
    Skips joints with zero error (fixed joints already filtered upstream).
    """
    n = len(joint_indices)
    if n == 0:
        print(f"  [SKIP] No joints to plot for group '{group_name}'")
        return
 
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n),
                              sharex=True, gridspec_kw={"wspace": 0.35})
    if n == 1:
        axes = np.array([axes])   # keep 2D indexing
 
    for row, j in enumerate(joint_indices):
        # Left: trajectories
        ax = axes[row, 0]
        ax.plot(t, a_q[:, j], color=BLUE, lw=1.4, label="Sim A")
        ax.plot(t, b_q[:, j], color=RED,  lw=1.4, label="Sim B", linestyle="--")
        ax.fill_between(t, a_q[:, j], b_q[:, j], alpha=0.15, color=AMBER)
        ax.set_ylabel(f"J{j} (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
        if j == worst_joint:
            ax.set_ylabel(f"J{j} ★ (rad)", fontsize=8, color=RED)
        if row == 0:
            ax.set_title(f"Joint angle — {group_name}", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
 
        # Right: absolute error
        ax = axes[row, 1]
        ax.plot(t, dq_abs[:, j], color=PURPLE, lw=1.2)
        ax.fill_between(t, dq_abs[:, j], alpha=0.25, color=PURPLE)
        ax.set_ylabel(f"|ΔJ{j}| (rad)", fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title("|Error| per joint", fontsize=9)
 
    axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=9)
 
    fig.suptitle(f"Joint Trajectories — {group_name}\n"
                 f"(★ = most-divergent joint overall: J{worst_joint})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, output_path)
 
 
def plot_joints(a, b, c, output_dir: Path):
    """
    Plots only the arm joints (the joints your controller actually drives).
    The cube's floating base is intentionally NOT plotted here — its
    position is covered by 04_cube_comparison.png and its orientation by
    07_quaternion_error.png, since a raw quaternion-component subtraction
    is not a meaningful error metric.
    Fixed joints (zero error) are silently skipped.
    """
    t   = c["t"]
    arm_end = c["arm_end"]
    a_q = a["q"][:, :arm_end]
    b_q = b["q"][:, :arm_end]
 
    _plot_joint_group(
        t, a_q, b_q, c["dq_abs"],
        joint_indices = c["arm_joints"],
        group_name    = "Arm joints",
        worst_joint   = c["worst_joint"],
        output_path   = output_dir / "01_joint_trajectories_arm.png",
    )
 
 
# Plot 2 : Per-joint error heatmap
 
def plot_heatmap(a, b, c, output_dir: Path):
    t = c["t"]
    arm_end = c["arm_end"]
 
    # Only show arm joints that actually have non-zero error
    active = [j for j in range(arm_end) if not c["fixed_mask"][j]]
    if not active:
        print("  [SKIP] No active arm joints to plot in heatmap")
        return
    diff   = c["dq_abs"][:, active].T    # (n_active, T)
    labels = [f"J{j}" + (" ★" if j == c["worst_joint"] else "")
              for j in active]
 
    fig, ax = plt.subplots(figsize=(14, max(4, len(active) * 0.45)))
    im = ax.imshow(diff, aspect="auto", cmap="YlOrRd", origin="lower",
                   extent=[t[0], t[-1], -0.5, len(active) - 0.5],
                   interpolation="bilinear")
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Joint", fontsize=10)
    ax.set_yticks(range(len(active)))
    ax.set_yticklabels(labels, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("|q_A − q_B| (rad)", fontsize=9)
 
    peak_t = t[np.argmax(c["dq_norm"])]
    ax.axvline(peak_t, color=BLUE, lw=1.2, linestyle="--", alpha=0.8,
               label=f"Peak joint error at t={peak_t:.1f}s")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Per-Joint Error Heatmap — |Sim A − Sim B| over time\n"
                 "(arm joints only, fixed/zero-error joints hidden  |  ★ = worst joint)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "02_joint_error_heatmap.png")
 
 
# Plot 3 : End-effector comparison
 
def plot_ee(a, b, c, output_dir: Path):
    t = c["t"]
    T = len(t)
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
 
    # EE Z height
    ax = axes[0, 0]
    ax.plot(t, a["ee_pos"][:T, 2] * 1000, color=BLUE, lw=1.4, label="Sim A")
    ax.plot(t, b["ee_pos"][:T, 2] * 1000, color=RED,  lw=1.4, label="Sim B", linestyle="--")
    ax.fill_between(t, a["ee_pos"][:T, 2] * 1000, b["ee_pos"][:T, 2] * 1000, alpha=0.15, color=AMBER)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE height (mm)")
    ax.set_title("EE height over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # Per-axis EE error
    ax = axes[0, 1]
    for i, (col, lbl) in enumerate(zip(AXIS_C, ["X", "Y", "Z"])):
        ax.plot(t, c["dee"][:, i], color=col, lw=1.2, label=f"{lbl}")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EE error (mm)")
    ax.set_title("EE error per axis (A − B)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # EE total distance (norm)
    ax = axes[1, 0]
    ax.plot(t, c["dee_dist"], color=GREEN, lw=1.5)
    ax.fill_between(t, c["dee_dist"], alpha=0.2, color=GREEN)
    ax.axhline(c["mean_ee_mm"], color=RED,   lw=1.1, linestyle="--", label=f"mean = {c['mean_ee_mm']:.2f} mm")
    ax.axhline(c["max_ee_mm"],  color=AMBER, lw=1.0, linestyle=":",  label=f"max  = {c['max_ee_mm']:.2f} mm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||EE_A − EE_B|| (mm)")
    ax.set_title("EE total position error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # Histogram of EE total error
    ax = axes[1, 1]
    ax.hist(c["dee_dist"], bins=30, color=GREEN, alpha=0.7, edgecolor='black')
    ax.axvline(c["mean_ee_mm"], color=RED, linestyle="--", label=f"mean: {c['mean_ee_mm']:.2f} mm")
    ax.set_xlabel("EE error (mm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of EE position error")
    ax.legend()
    ax.grid(alpha=0.3)
 
    fig.suptitle(f"End-Effector Comparison — Sim A vs Sim B\n"
                 f"mean = {c['mean_ee_mm']:.2f} mm  |  max = {c['max_ee_mm']:.2f} mm",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, output_dir / "03_ee_comparison.png")
 
# Plot 4 : Object (cube) comparison
 
def plot_object(a, b, c, output_dir: Path):
    t = c["t"]
    T = len(t)
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
 
    # Cube Z height
    ax = axes[0, 0]
    ax.plot(t, a["obj_pos"][:T, 2] * 1000, color=BLUE, lw=1.4, label="Sim A")
    ax.plot(t, b["obj_pos"][:T, 2] * 1000, color=RED,  lw=1.4, label="Sim B", linestyle="--")
    ax.fill_between(t, a["obj_pos"][:T, 2] * 1000, b["obj_pos"][:T, 2] * 1000, alpha=0.15, color=AMBER)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cube Z height (mm)")
    ax.set_title("Cube height over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # Per-axis cube error
    ax = axes[0, 1]
    for i, (col, lbl) in enumerate(zip(AXIS_C, ["X", "Y", "Z"])):
        ax.plot(t, c["dobj"][:, i], color=col, lw=1.2, label=f"{lbl}")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cube error (mm)")
    ax.set_title("Cube error per axis (A − B)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # Cube total distance (norm)
    ax = axes[1, 0]
    ax.plot(t, c["dobj_dist"], color=AMBER, lw=1.5)
    ax.fill_between(t, c["dobj_dist"], alpha=0.2, color=AMBER)
    ax.axhline(c["mean_obj_mm"], color=RED,   lw=1.1, linestyle="--", label=f"mean = {c['mean_obj_mm']:.2f} mm")
    ax.axhline(c["max_obj_mm"],  color=AMBER, lw=1.0, linestyle=":",  label=f"max  = {c['max_obj_mm']:.2f} mm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||obj_A − obj_B|| (mm)")
    ax.set_title("Cube total position error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    # Histogram of cube total error
    ax = axes[1, 1]
    ax.hist(c["dobj_dist"], bins=30, color=AMBER, alpha=0.7, edgecolor='black')
    ax.axvline(c["mean_obj_mm"], color=RED, linestyle="--", label=f"mean: {c['mean_obj_mm']:.2f} mm")
    ax.set_xlabel("Cube error (mm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of cube position error")
    ax.legend()
    ax.grid(alpha=0.3)
 
    fig.suptitle(f"Cube (Object) Comparison — Sim A vs Sim B\n"
                 f"mean = {c['mean_obj_mm']:.2f} mm  |  max = {c['max_obj_mm']:.2f} mm",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, output_dir / "04_cube_comparison.png")
 
# Plot 5 : Per-joint bar — mean error, active arm joints only
 
def plot_per_joint_bar(c, output_dir: Path):
    arm_end  = c["arm_end"]
    mean_err = c["dq_abs"].mean(axis=0)   # (n_arm,)
 
    # Only show joints with non-zero error
    active = [j for j in range(arm_end) if not c["fixed_mask"][j]]
    if not active:
        print("  [SKIP] No active arm joints to plot in per-joint bar")
        return
    vals   = mean_err[active]
    labels = [f"J{j}" + (" ★" if j == c["worst_joint"] else "") for j in active]
    colors = [RED if j == c["worst_joint"] else BLUE for j in active]
 
    fig, ax = plt.subplots(figsize=(max(8, len(active) * 0.7), 5))
    bars = ax.bar(range(len(active)), vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(active)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("|q_A − q_B| (rad)", fontsize=10)
    ax.set_title(f"Per-Joint Mean Error — Sim A vs Sim B\n"
                 f"(arm joints only  |  ★ = worst)",
                 fontsize=11, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
 
    # Legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=BLUE,  label="Arm joint"),
        Patch(color=RED,   label="Worst joint ★"),
    ], fontsize=8)
 
    fig.tight_layout()
    _save(fig, output_dir / "05_per_joint_bar.png")
 
 
# Plot 6 : Combined RMSE and its components (unweighted)
 
def plot_combined_rmse(c, output_dir: Path):
    """
    Single panel: combined RMSE alongside its two components
    (EE error and cube position error). No weighting is applied — both
    terms are already in mm and combined in quadrature:
 
        rmse(t) = sqrt(dee_dist(t)^2 + dobj_dist(t)^2)
    """
    t = c["t"]
 
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, c["rmse_per_t_mm"], color=PURPLE, lw=2.0, label="Combined RMSE", zorder=3)
    ax.plot(t, c["dee_dist"],  color=GREEN, lw=1.3, linestyle="--", label="EE component")
    ax.plot(t, c["dobj_dist"], color=AMBER, lw=1.3, linestyle="--", label="Cube component")
    ax.axhline(c["mean_rmse_mm"], color=RED, lw=1.1, linestyle=":",
               label=f"mean RMSE = {c['mean_rmse_mm']:.2f} mm")
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Distance (mm)", fontsize=10)
    ax.set_title("Combined RMSE and its components\n"
                 "rmse(t) = sqrt(||ΔEE||² + ||Δobj||²)   (unweighted)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
 
    fig.suptitle("Combined RMSE Analysis\n"
                 "(ground-truth state-space metric — compare with energy distance)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, output_dir / "06_combined_rmse.png")
 
 
# Plot 7 : Cube orientation (quaternion) error over time
 
def plot_quaternion_error(c, output_dir: Path):
    t = c["t"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, c["quat_angle"] * 180 / np.pi, color=PURPLE, lw=1.5, label="Angular distance")
    ax.fill_between(t, c["quat_angle"] * 180 / np.pi, alpha=0.2, color=PURPLE)
    ax.axhline(c["mean_quat_angle_rad"] * 180 / np.pi, color=RED, linestyle="--",
               label=f"mean = {c['mean_quat_angle_rad']*180/np.pi:.2f}°")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular error (degrees)")
    ax.set_title("Cube Orientation Error (Sim A vs Sim B)\n"
                 "(geodesic angle between quaternions — never a linear subtraction)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "07_quaternion_error.png")
 
 
# Report
 
def save_report(c, output_dir: Path):
    report = {
        "combined_rmse": {
            "formula":       "sqrt(dee_dist_mm^2 + dobj_dist_mm^2)  (unweighted)",
            "mean_rmse_mm":  round(c["mean_rmse_mm"], 4),
            "max_rmse_mm":   round(c["max_rmse_mm"],  4),
            "note":          "All distances in mm. Compare mean_rmse_mm with energy distance.",
        },
        "end_effector_position": {
            "mean_error_mm": round(c["mean_ee_mm"], 4),
            "max_error_mm":  round(c["max_ee_mm"],  4),
        },
        "cube_position": {
            "mean_error_mm": round(c["mean_obj_mm"], 4),
            "max_error_mm":  round(c["max_obj_mm"],  4),
        },
        "cube_orientation": {
            "mean_error_rad": round(c["mean_quat_angle_rad"], 6),
            "max_error_rad":  round(c["max_quat_angle_rad"],  6),
            "note": "Geodesic angular distance between quaternions (radians), not a subtraction.",
        },
        "arm_joint_q": {
            "mean_error_rad":     round(c["mean_joint_err_rad"], 6),
            "max_error_rad":      round(c["max_joint_err_rad"],  6),
            "worst_joint_index":  c["worst_joint"],
            "arm_joints":         c["arm_joints"],
            "per_joint_mean_rad": [round(float(v), 6)
                                   for v in c["dq_abs"].mean(axis=0)],
            "note": "Cube's floating-base (quaternion + xyz) is excluded from this "
                    "joint-angle comparison; see cube_position / cube_orientation instead.",
        },
    }
    out = output_dir / "state_comparison_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
 
    print("\nSTATE-TO-STATE COMPARISON REPORT")
    print(f"  Combined RMSE  mean / max : "
          f"{c['mean_rmse_mm']:.2f} / {c['max_rmse_mm']:.2f} mm")
    print(f"  EE position    mean / max : "
          f"{c['mean_ee_mm']:.2f} / {c['max_ee_mm']:.2f} mm")
    print(f"  Cube position  mean / max : "
          f"{c['mean_obj_mm']:.2f} / {c['max_obj_mm']:.2f} mm")
    print(f"  Cube orientation mean / max : "
          f"{c['mean_quat_angle_rad']:.4f} / {c['max_quat_angle_rad']:.4f} rad")
    print(f"  Arm joint q    mean / max : "
          f"{c['mean_joint_err_rad']:.4f} / {c['max_joint_err_rad']:.4f} rad")
    print(f"  Worst joint               : J{c['worst_joint']}")
    print(f"  Arm joints                : {c['arm_joints']}")
    print(f"""
  HOW TO COMPARE WITH ENERGY DISTANCE
  Energy distance (DINOv2):  appearance-space gap  = lighting + motion
  State RMSE (this file):    physics-space gap     = motion only
 
  If energy_distance is HIGH and state RMSE is LOW:
      → sims look different but move the same → lighting is the cause
 
  If BOTH are HIGH:
      → sims look different AND move differently → real physical gap
 
  If state RMSE is HIGH and energy_distance is LOW:
      → sims move differently but camera cannot see it
        (e.g. internal joint difference not visible from outside)
""")
    print(f"  Full report → {out}")
 
 
# Main
 
def main():
    parser = argparse.ArgumentParser(
        description="Point-by-point state trajectory comparison: Sim A vs Sim B"
    )
    parser.add_argument("--traj_a",  required=True,
                        help="save_dir of Sim A (contains trajectory_states.npz)")
    parser.add_argument("--traj_b",  required=True,
                        help="save_dir of Sim B")
    parser.add_argument("--output",  default="state_comparison_output",
                        help="Output directory for plots and report")
    args = parser.parse_args()
 
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print("STATE-TO-STATE TRAJECTORY COMPARATOR")
    print(f"\nSim A  : {args.traj_a}")
    print(f"Sim B  : {args.traj_b}")
    print(f"Output : {output_dir}\n")
 
    print("Loading ...")
    a_raw = load_trajectory(Path(args.traj_a))
    b_raw = load_trajectory(Path(args.traj_b))
 
    print("\nAligning ...")
    a, b = align(a_raw, b_raw)
 
    print("\nComputing point-by-point state differences ...")
    c = compare(a, b)
 
    # Save RMSE time series for energy correlation
    rmse_npz_path = output_dir / "trajectory_states.npz"
    np.savez_compressed(
        rmse_npz_path,
        times=c["t"],
        rmse_per_t_mm=c["rmse_per_t_mm"]
    )
    print(f"  Saved RMSE timeseries → {rmse_npz_path}")
 
    print("\nGenerating plots ...")
    plot_dashboard(a, b, c, output_dir)
    plot_joints(a, b, c, output_dir)
    plot_heatmap(a, b, c, output_dir)
    plot_ee(a, b, c, output_dir)
    plot_object(a, b, c, output_dir)
    plot_per_joint_bar(c, output_dir)
    plot_combined_rmse(c, output_dir)
    plot_quaternion_error(c, output_dir)
 
    save_report(c, output_dir)
 
    print(f"\nAll outputs in: {output_dir}/")
    print("  00_summary_dashboard.png     ← start here")
    print("  01_joint_trajectories_arm.png ← arm joints only")
    print("  02_joint_error_heatmap.png   ← arm joints × time")
    print("  03_ee_comparison.png")
    print("  04_cube_comparison.png")
    print("  05_per_joint_bar.png")
    print("  06_combined_rmse.png         ← compare with energy distance")
    print("  07_quaternion_error.png      ← cube orientation, geodesic angle")
    print("  state_comparison_report.json")


if __name__ == "__main__":
    main()

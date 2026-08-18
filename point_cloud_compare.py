#!/usr/bin/env python
"""
Point cloud comparison between real (teleop) and sim (Drake replay) frames.

Uses calibrated camera intrinsics + extrinsics to unproject depth into a
COMMON world/robot-base frame for each of the 4 cameras, then merges all 4
cameras' points into one combined cloud per label (e.g. "grab" for real,
"grab_neutral" / "grab_warm" for sim). Real-vs-sim comparison is then done
in 3D directly, which sidesteps pixel misalignment and unit/normalization
issues that the RGB/depth-colorization + DINO route has.

Why this instead of colorized-depth + DINO:
  - Drake's simulated depth is a geometric ray-cast; it should not change
    between your "neutral" and "warm" lighting sim variants (lighting only
    affects the RGB render). So Chamfer distance here should be near-zero
    for lighting-only changes and clearly larger when the cube is actually
    misplaced -- which is the exact contrast you're trying to prove.
  - Comparing in 3D avoids the per-image depth-colormap normalization
    problem in the original depth_comparison.py script (each frame's
    colorization is scaled independently, destroying absolute-depth
    comparability).

Calibration file format (JSON), one entry per camera name matching your
depth_comparison.py cam folder names (cam_high, cam_low, cam_left_wrist,
cam_right_wrist, ...):

{
  "cam_high": {
    "fx": 605.2, "fy": 605.4, "cx": 320.1, "cy": 240.6,
    "width": 640, "height": 480,
    "cam_to_world": [[r11, r12, r13, tx],
                      [r21, r22, r23, ty],
                      [r31, r32, r33, tz],
                      [0,    0,    0,   1]]
  },
  "cam_low": { ... },
  ...
}

Conventions:
  - Camera frame is standard OpenCV/pinhole: x-right, y-down, z-forward.
  - "cam_to_world" maps a point FROM camera frame TO your world/robot-base
    frame (i.e. p_world = cam_to_world @ [p_cam; 1]). If your calibration
    stores the inverse (world_to_cam / extrinsic matrix in the classic CV
    sense), pass --invert_extrinsics and this script will invert it for you.
  - Real depth is read in millimeters (matches depth_comparison.py), sim
    depth in meters. Both get converted to meters before unprojection.

If you confirm sim cameras are posed identically to the real rig, pass the
SAME calib file to both --real_calib and --sim_calib. If they might differ,
pass them separately.

Usage:
    python point_cloud_compare.py \
        --group_dir simulation_frames/test_frames_real_depth_1 \
        --real_root data/pick_place_depth_3 \
        --sim_replays simulation_frames/replay_1786335633 simulation_frames/replay_1786335800 \
                      simulation_frames/replay_XXXXXXXXXX simulation_frames/replay_YYYYYYYYYY \
        --real_calib calib/real_cams.json \
        --sim_calib calib/sim_cams.json \
        --out_root pointcloud_comparison \
        --near_m 0.15 --far_m 0.9 \
        --roi_center 0.45 0.0 0.05 --roi_half_extent 0.08 0.08 0.08

Output:
    pointcloud_comparison/<group>_output/clouds/<label>.ply   (viewable in
        MeshLab / CloudCompare / open3d)
    pointcloud_comparison/<group>_output/metrics.json
    pointcloud_comparison/<group>_output/chamfer_bar.png
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the file-discovery / parquet-lookup helpers from your existing script
# so cam-folder / frame-number / stride logic stays identical and in sync.
from depth_comparison import (
    frame_number,
    label_from_filename,
    find_real_parquet,
    infer_stride,
    find_sim_depth_npy,
    load_real_depth_frame,
)

try:
    import open3d as o3d
    HAVE_O3D = True
except ImportError:
    HAVE_O3D = False


# ---------- calibration ----------

def load_calib(path: Path, invert: bool) -> Dict[str, dict]:
    with open(path) as f:
        raw = json.load(f)
    calib = {}
    for cam, c in raw.items():
        K = np.array([[c["fx"], 0, c["cx"]],
                      [0, c["fy"], c["cy"]],
                      [0, 0, 1]], dtype=np.float64)
        T = np.array(c["cam_to_world"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"cam_to_world for '{cam}' must be 4x4, got {T.shape}")
        if invert:
            T = np.linalg.inv(T)
        calib[cam] = {"K": K, "T": T, "width": c.get("width"), "height": c.get("height")}
    return calib


# ---------- unprojection ----------

def unproject(depth_m: np.ndarray, K: np.ndarray, T_cam_to_world: np.ndarray,
              near_m: float, far_m: float,
              rgb: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """depth_m: (H, W) depth in meters. Returns (N, 3) world-frame points
    [+ (N, 3) uint8 colors if rgb given], filtered to [near_m, far_m]."""
    h, w = depth_m.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    valid = (depth_m > near_m) & (depth_m < far_m) & np.isfinite(depth_m)
    if not valid.any():
        return np.zeros((0, 3)), (np.zeros((0, 3), dtype=np.uint8) if rgb is not None else None)

    vs, us = np.nonzero(valid)
    z = depth_m[vs, us]
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)

    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)  # (N, 4)
    pts_world = (T_cam_to_world @ pts_h.T).T[:, :3]

    colors = rgb[vs, us] if rgb is not None else None
    return pts_world, colors


# ---------- comparison metrics ----------

def chamfer_distance(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Symmetric nearest-neighbor distance between two point sets."""
    if len(a) == 0 or len(b) == 0:
        return {"chamfer_mean": float("nan"), "a_to_b_mean": float("nan"), "b_to_a_mean": float("nan")}
    tree_b = cKDTree(b)
    tree_a = cKDTree(a)
    d_a_to_b, _ = tree_b.query(a, k=1)
    d_b_to_a, _ = tree_a.query(b, k=1)
    return {
        "chamfer_mean": float(d_a_to_b.mean() + d_b_to_a.mean()),
        "a_to_b_mean": float(d_a_to_b.mean()),
        "b_to_a_mean": float(d_b_to_a.mean()),
        "a_to_b_median": float(np.median(d_a_to_b)),
        "b_to_a_median": float(np.median(d_b_to_a)),
    }


def roi_occupancy(points: np.ndarray, center: np.ndarray, half_extent: np.ndarray) -> Dict[str, float]:
    """Count / density of points inside an axis-aligned box in world frame --
    point this at the expected cube/goal location to directly measure
    'is there stuff here or not' independent of the rest of the scene."""
    if len(points) == 0:
        return {"n_points_in_roi": 0, "fraction_in_roi": 0.0}
    lo, hi = center - half_extent, center + half_extent
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    return {
        "n_points_in_roi": int(inside.sum()),
        "fraction_in_roi": float(inside.mean()),
    }


# ---------- ply writing (no open3d required) ----------

def write_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None):
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is not None:
            for p, c in zip(points, colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        else:
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")


# ---------- main per-group processing ----------

def build_real_cloud(cam_dirs: List[Path], real_root: Path, parquet_path: Path,
                      calib: Dict[str, dict], near_m: float, far_m: float,
                      real_depth_col_template: str, want_rgb: bool
                      ) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Returns {label: (points, colors)} merged across all cams for each real_* label."""
    stride_cache = {}
    per_label_pts: Dict[str, List[np.ndarray]] = {}
    per_label_col: Dict[str, List[np.ndarray]] = {}

    for cam_dir in cam_dirs:
        cam = cam_dir.name
        if cam not in calib:
            print(f"  [warn] no calibration for '{cam}', skipping this camera for real cloud")
            continue
        for real_png in sorted(cam_dir.glob("real_*_frame_*.png")):
            label = label_from_filename(real_png, "real_")
            fnum = frame_number(real_png)
            if cam not in stride_cache:
                stride_cache[cam] = infer_stride(parquet_path, real_root, cam)
            row_idx = (fnum - 1) * stride_cache[cam]

            depth_mm = load_real_depth_frame(parquet_path, cam, row_idx, real_depth_col_template)
            depth_m = depth_mm.astype(np.float64) / 1000.0
            rgb = np.array(Image.open(real_png).convert("RGB")) if want_rgb else None

            pts, cols = unproject(depth_m, calib[cam]["K"], calib[cam]["T"], near_m, far_m, rgb)
            per_label_pts.setdefault(label, []).append(pts)
            if want_rgb:
                per_label_col.setdefault(label, []).append(cols)
            print(f"  [real:{cam}] {label} (frame {fnum}) -> {len(pts)} pts in [{near_m},{far_m}]m")

    out = {}
    for label, parts in per_label_pts.items():
        pts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))
        cols = np.concatenate(per_label_col[label], axis=0) if want_rgb and per_label_col.get(label) else None
        out[label] = (pts, cols)
    return out


def build_sim_clouds(cam_dirs: List[Path], sim_replay_dirs: List[Path],
                      calib: Dict[str, dict], near_m: float, far_m: float,
                      want_rgb: bool, label_filter=None) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Returns {label: (points, colors)} merged across all cams for each sim_* label."""
    per_label_pts: Dict[str, List[np.ndarray]] = {}
    per_label_col: Dict[str, List[np.ndarray]] = {}

    for cam_dir in cam_dirs:
        cam = cam_dir.name
        if cam not in calib:
            print(f"  [warn] no calibration for '{cam}', skipping this camera for sim clouds")
            continue
        for sim_png in sorted(cam_dir.glob("sim_*_frame_*.png")):
            label = label_from_filename(sim_png, "sim_").strip('_')
            print(f"  [debug] PNG: {sim_png.name} -> label='{label}', filter={label_filter}")

            if label_filter is not None and label not in label_filter:
                continue
            fnum = frame_number(sim_png)

            matches = find_sim_depth_npy(sim_replay_dirs, cam, fnum)
            if len(matches) != 1:
                tag = "no match" if not matches else "ambiguous"
                print(f"  [warn:{tag}] {cam} frame {fnum} ({sim_png.name}), skipping")
                continue

            depth_m = np.load(matches[0]).astype(np.float64)
            rgb = np.array(Image.open(sim_png).convert("RGB")) if want_rgb else None

            pts, cols = unproject(depth_m, calib[cam]["K"], calib[cam]["T"], near_m, far_m, rgb)
            per_label_pts.setdefault(label, []).append(pts)
            if want_rgb:
                per_label_col.setdefault(label, []).append(cols)
            print(f"  [sim:{cam}] {label} (frame {fnum}) -> {len(pts)} pts in [{near_m},{far_m}]m")

    out = {}
    for label, parts in per_label_pts.items():
        pts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))
        cols = np.concatenate(per_label_col[label], axis=0) if want_rgb and per_label_col.get(label) else None
        out[label] = (pts, cols)
    return out


def process_group(args):
    group_dir = Path(args.group_dir)
    real_root = Path(args.real_root)
    sim_replay_dirs = [Path(p) for p in args.sim_replays]
    out_root = Path(args.out_root) / f"{group_dir.name}_output"
    clouds_dir = out_root / "clouds"
    clouds_dir.mkdir(parents=True, exist_ok=True)

    real_calib = load_calib(Path(args.real_calib), args.invert_extrinsics)
    sim_calib = load_calib(Path(args.sim_calib), args.invert_extrinsics) if args.sim_calib else real_calib
    if args.sim_calib is None:
        print("[info] using --real_calib for sim cameras too (assumed identical poses)")

    parquet_path = find_real_parquet(real_root)
    cam_dirs = sorted(d for d in group_dir.iterdir() if d.is_dir())

    print("\n=== Building real point clouds ===")
    real_clouds = build_real_cloud(cam_dirs, real_root, parquet_path, real_calib,
                                    args.near_m, args.far_m,
                                    args.real_depth_col_template, args.save_rgb_ply)

    print("\n=== Building sim point clouds ===")
    sim_clouds = build_sim_clouds(cam_dirs, sim_replay_dirs, sim_calib,
                                   args.near_m, args.far_m, args.save_rgb_ply, args.label_filter)

    for label, (pts, cols) in real_clouds.items():
        write_ply(clouds_dir / f"real_{label}.ply", pts, cols)
    for label, (pts, cols) in sim_clouds.items():
        write_ply(clouds_dir / f"sim_{label}.ply", pts, cols)
    print(f"\nWrote {len(real_clouds)} real + {len(sim_clouds)} sim point clouds -> {clouds_dir}")

    # --- metrics: every real label vs every sim label ---
    roi_center = np.array(args.roi_center) if args.roi_center else None
    roi_half = np.array(args.roi_half_extent) if args.roi_half_extent else None

    metrics = {}
    for r_label, (r_pts, _) in real_clouds.items():
        for s_label, (s_pts, _) in sim_clouds.items():
            key = f"real_{r_label}__vs__sim_{s_label}"
            m = chamfer_distance(r_pts, s_pts)
            if roi_center is not None:
                m["real_roi"] = roi_occupancy(r_pts, roi_center, roi_half)
                m["sim_roi"] = roi_occupancy(s_pts, roi_center, roi_half)
                m["roi_point_count_diff"] = m["real_roi"]["n_points_in_roi"] - m["sim_roi"]["n_points_in_roi"]
            metrics[key] = m
            print(f"  {key}: chamfer_mean={m['chamfer_mean']:.4f}")

    with open(out_root / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {out_root / 'metrics.json'}")

    # bar chart of chamfer distance, sorted worst-to-best, so lighting vs
    # misplacement variants are easy to eyeball against each other
    if metrics:
        keys = sorted(metrics.keys(), key=lambda k: metrics[k]["chamfer_mean"])
        vals = [metrics[k]["chamfer_mean"] for k in keys]
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(keys)), 5))
        ax.bar(range(len(keys)), vals, color="steelblue")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Chamfer distance (m, lower = closer)")
        ax.set_title("Real vs sim point cloud distance")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_root / "chamfer_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved chart -> {out_root / 'chamfer_bar.png'}")

    if args.visualize:
        viz_dir = clouds_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        for label, (pts, cols) in real_clouds.items():
            if len(pts) > 0:
                plot_point_cloud(pts, cols, viz_dir / f"real_{label}_3d.png", title=f"Real {label}")
        for label, (pts, cols) in sim_clouds.items():
            if len(pts) > 0:
                plot_point_cloud(pts, cols, viz_dir / f"sim_{label}_3d.png", title=f"Sim {label}")
        print(f"Saved 3D visualizations to {viz_dir}")


def plot_point_cloud(points: np.ndarray, colors: Optional[np.ndarray], out_path: Path, title: str = "",
                     max_points: int = 10000):
    if len(points) == 0:
        return
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx] if colors is not None else None
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if colors is not None:
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors/255.0, s=1)
    else:
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group_dir", required=True)
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--sim_replays", nargs="+", required=True)
    ap.add_argument("--real_calib", required=True, help="JSON calib file for real cameras")
    ap.add_argument("--sim_calib", default=None,
                     help="JSON calib file for sim cameras. Omit to reuse --real_calib "
                          "(only valid if sim cams are posed identically to the real rig).")
    ap.add_argument("--invert_extrinsics", action="store_true",
                     help="pass this if your calib stores world_to_cam instead of cam_to_world")
    ap.add_argument("--label_filter", nargs="+", default=None, help="Only process these sim labels, e.g. grab_neutral drop_neutral")
    ap.add_argument("--out_root", default="pointcloud_comparison")
    ap.add_argument("--real_depth_col_template", default="observation.images.{cam}.depth")
    ap.add_argument("--near_m", type=float, default=0.15,
                     help="single near-field threshold (meters) applied to BOTH real and sim, "
                          "so the crop is finally consistent between them")
    ap.add_argument("--far_m", type=float, default=0.9)
    ap.add_argument("--roi_center", type=float, nargs=3, default=None,
                     help="x y z (world frame, meters) of expected cube/goal location, for the "
                          "ROI occupancy metric. Omit to skip this metric.")
    ap.add_argument("--roi_half_extent", type=float, nargs=3, default=[0.08, 0.08, 0.08],
                     help="half-size of the ROI box in meters, per axis")
    ap.add_argument("--save_rgb_ply", action="store_true",
                     help="color the point clouds using the RGB frames (slower, bigger files, "
                          "but much easier to visually inspect in MeshLab/CloudCompare)")
    ap.add_argument("--visualize", action="store_true",
                help="Generate 3D scatter plots of the point clouds (subsampled) for inspection")
    args = ap.parse_args()
    process_group(args)


if __name__ == "__main__":
    main()
    
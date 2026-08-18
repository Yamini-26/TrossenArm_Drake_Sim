#!/usr/bin/env python
"""
Depth comparison preprocessing, driven entirely by frame numbers already
embedded in your hand-picked filenames -- no manual row-index lookup needed.

Expected input layout (auto-discovers every cam subfolder, no need to name
them individually):

  <group_dir>/
      cam_high/
          real_grab_frame_000050.png
          sim_grab_neutral_frame_00033.png
          sim_grab_warm_frame_00033.png
          sim_drop_neutral_frame_00041.png
          sim_drop_warm_frame_00041.png
      cam_low/
          ...
      cam_right_wrist/
          ...

For each cam folder, every real_*_frame_XXXXXX.png is paired against every
sim_*_frame_XXXXX.png in that same cam folder (so one real frame can be
compared against several sim variants).

The RGB itself is read straight from your renamed files (no need to touch
the original source images) -- only the *depth* has to be looked up:
  - real depth: located via --real_root (the episode's parquet), with the
    parquet row index inferred automatically from the frame number and the
    ratio of parquet rows to extracted frame pngs.
  - sim depth: located by searching only the replay_<timestamp> folders you
    pass via --sim_replays for a depth/frame_XXXXX.npy with a matching frame
    number and cam. If the same frame number exists in more than one of the
    replay folders you passed, that file is reported as ambiguous and
    skipped (rather than silently guessing).

Usage:
    python depth_comparison.py \
        --group_dir simulation_frames/test_frames_real_depth_1 \
        --real_root data/pick_place_depth_3 \
        --sim_replays simulation_frames/replay_1786335633 simulation_frames/replay_1786335800 \
        --out_root depth_comparison

Output:
    depth_comparison/test_frames_real_depth_1_output/masked_rgb/<cam>/real_<label>.png, sim_<label>.png
    depth_comparison/test_frames_real_depth_1_output/depth_vis/<cam>/real_<label>.png, sim_<label>.png
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

FRAME_NUM_RE = re.compile(r"_frame_0*(\d+)\.png$")


def print_schema(parquet_path: Path):
    pf = pq.ParquetFile(parquet_path)
    print(f"Columns in {parquet_path}:")
    for n in pf.schema_arrow.names:
        print(f"  {n}")


# ---------- core depth helpers (unchanged from the original script) ----------

def load_real_depth_frame(parquet_path, cam, row_idx, col_template, h=480, w=640) -> np.ndarray:
    col = col_template.format(cam=cam)
    pf = pq.ParquetFile(parquet_path)
    schema_names = pf.schema_arrow.names
    if col not in schema_names:
        candidates = [n for n in schema_names if cam in n or "depth" in n.lower()]
        raise KeyError(
            f"Column '{col}' not found in {parquet_path}.\n"
            f"  Columns containing '{cam}' or 'depth' in this file:\n"
            + "".join(f"    {n}\n" for n in candidates)
            + f"  Pass the correct pattern via --real_depth_col_template, using '{{cam}}' "
              f"as a placeholder for the cam name, e.g.:\n"
              f"    --real_depth_col_template 'observation.images.{{cam}}.depth'"
        )
    table = pf.read(columns=[col])
    val = table.column(col)[row_idx].as_py()
    return np.array(val, dtype=np.int32).reshape(h, w)


def near_field_mask(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    return (depth > near) & (depth < far)


def colorize_depth(depth: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    d = depth.astype(np.float32).copy()
    if mask is not None:
        d[~mask] = 0
    finite = d > 0
    vis = np.zeros_like(d)
    if finite.any():
        d_min, d_max = d[finite].min(), d[finite].max()
        vis[finite] = (d[finite] - d_min) / max(d_max - d_min, 1e-6)
    colored = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def apply_mask(rgb: np.ndarray, mask: np.ndarray, fill=(0, 0, 0)) -> np.ndarray:
    out = rgb.copy()
    out[~mask] = fill
    return out


# ---------- lookup helpers ----------

def frame_number(png: Path) -> int:
    m = FRAME_NUM_RE.search(png.name)
    if not m:
        raise ValueError(f"couldn't find _frame_XXXXXX in {png.name}")
    return int(m.group(1))


def label_from_filename(png: Path, prefix: str) -> str:
    """real_grab_frame_000050.png -> 'grab'; sim_grab_neutral_frame_00033.png -> 'grab_neutral'."""
    stem = png.name[:-len(".png")]
    stem = stem[len(prefix):]                      # strip 'real_' / 'sim_'
    stem = FRAME_NUM_RE.sub("", "_" + stem + ".png")  # reuse regex to strip trailing _frame_XXXXXX
    return stem


def find_real_parquet(real_root: Path) -> Path:
    matches = sorted(real_root.glob("data/chunk-*/episode_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no parquet found under {real_root}/data/chunk-*/episode_*.parquet")
    if len(matches) > 1:
        print(f"  [warn] multiple parquet files under {real_root}, using {matches[0]}")
    return matches[0]


def infer_stride(parquet_path: Path, real_root: Path, cam: str) -> int:
    n_rows = pq.ParquetFile(parquet_path).metadata.num_rows
    frames_dir = real_root / "frames" / cam
    n_frames = len(list(frames_dir.glob("frame_*.png"))) if frames_dir.is_dir() else 0
    if n_frames == 0:
        print(f"  [warn] couldn't count frames under {frames_dir}, assuming stride 1")
        return 1
    return max(round(n_rows / n_frames), 1)


def find_sim_depth_npy(sim_replay_dirs, cam: str, frame_num: int):
    """Search only the given replay_<timestamp> dirs for depth/frame_<frame_num>.npy under this cam."""
    candidates = []
    for replay_dir in sim_replay_dirs:
        depth_dir = replay_dir / cam / "depth"
        if not depth_dir.is_dir():
            continue
        for npy in depth_dir.glob("frame_*.npy"):
            m = re.search(r"frame_0*(\d+)\.npy$", npy.name)
            if m and int(m.group(1)) == frame_num:
                candidates.append(npy)
    return candidates


# ---------- processing ----------

def process_group(args):
    group_dir = Path(args.group_dir)
    real_root = Path(args.real_root)
    sim_replay_dirs = [Path(p) for p in args.sim_replays]
    for d in sim_replay_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"--sim_replays path does not exist or isn't a directory: {d}")
    out_root = Path(args.out_root) / f"{group_dir.name}_output"

    parquet_path = find_real_parquet(real_root)
    stride_cache = {}
    real_depth_cache = {}  # (cam, real_label) -> True once written

    skipped = []

    cams = sorted(d for d in group_dir.iterdir() if d.is_dir())
    if not cams:
        print(f"[warn] no cam subfolders found under {group_dir}")
        return

    for cam_dir in cams:
        cam = cam_dir.name
        reals = sorted(cam_dir.glob("real_*_frame_*.png"))
        sims = sorted(cam_dir.glob("sim_*_frame_*.png"))
        if not reals:
            print(f"  [warn] no real_*_frame_*.png found in {cam_dir}")
            continue
        if not sims:
            print(f"  [warn] no sim_*_frame_*.png found in {cam_dir}")
            continue

        masked_dir = out_root / "masked_rgb" / cam
        depth_dir = out_root / "depth_vis" / cam
        masked_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # --- real frames: mask + colorize once each ---
        real_masks = {}
        for real_png in reals:
            label = label_from_filename(real_png, "real_")
            fnum = frame_number(real_png)

            if cam not in stride_cache:
                stride_cache[cam] = infer_stride(parquet_path, real_root, cam)
            row_idx = (fnum - 1) * stride_cache[cam]

            depth = load_real_depth_frame(parquet_path, cam, row_idx, args.real_depth_col_template)
            mask = near_field_mask(depth, args.real_near_mm, args.real_far_mm)
            rgb = np.array(Image.open(real_png).convert("RGB"))

            Image.fromarray(apply_mask(rgb, mask)).save(masked_dir / f"real_{label}.png")
            Image.fromarray(colorize_depth(depth, mask)).save(depth_dir / f"real_{label}.png")
            print(f"[{cam}] real_{label} (frame {fnum}, row {row_idx}): mask frac={mask.mean():.3f}")

        # --- sim frames: mask + colorize each, looked up by frame number ---
        for sim_png in sims:
            label = label_from_filename(sim_png, "sim_")
            fnum = frame_number(sim_png)

            matches = find_sim_depth_npy(sim_replay_dirs, cam, fnum)
            if len(matches) == 0:
                print(f"  [warn] no depth npy found for {cam} frame {fnum} ({sim_png.name}), skipping")
                skipped.append(sim_png)
                continue
            if len(matches) > 1:
                print(f"  [warn] AMBIGUOUS: frame {fnum} for {cam} found in multiple replay dirs, skipping:")
                for m in matches:
                    print(f"      {m}")
                skipped.append(sim_png)
                continue

            depth = np.load(matches[0])
            mask = near_field_mask(depth, args.sim_near_m, args.sim_far_m)
            rgb = np.array(Image.open(sim_png).convert("RGB"))

            Image.fromarray(apply_mask(rgb, mask)).save(masked_dir / f"sim_{label}.png")
            Image.fromarray(colorize_depth(depth, mask)).save(depth_dir / f"sim_{label}.png")
            print(f"[{cam}] sim_{label} (frame {fnum}, {matches[0].parent.parent.name}): mask frac={mask.mean():.3f}")

    print(f"\nDone. Output written to {out_root}")
    if skipped:
        print(f"[warn] {len(skipped)} sim frame(s) skipped -- see warnings above.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group_dir",
                     help="e.g. simulation_frames/test_frames_real_depth_1 (not needed with --print_schema_only)")
    ap.add_argument("--real_root", required=True,
                     help="e.g. data/pick_place_depth_3 (contains frames/, data/, meta/, videos/)")
    ap.add_argument("--sim_replays", nargs="+",
                     help="one or more specific replay_<timestamp> dirs to search (required unless "
                          "--print_schema_only), e.g. "
                          "simulation_frames/replay_1786335633 simulation_frames/replay_1786335800 ...")
    ap.add_argument("--out_root", default="depth_comparison")
    ap.add_argument("--real_depth_col_template", default="observation.images.{cam}.depth",
                     help="parquet column name pattern for real depth, with {cam} as a placeholder. "
                          "If this is wrong you'll get a KeyError listing the actual column names "
                          "found in your parquet file so you can correct it.")
    ap.add_argument("--real_near_mm", type=float, default=1)
    ap.add_argument("--real_far_mm", type=float, default=10000)
    ap.add_argument("--sim_near_m", type=float, default=0.15)
    ap.add_argument("--sim_far_m", type=float, default=0.9)
    ap.add_argument("--print_schema_only", action="store_true",
                     help="just print the parquet's column names (found via --real_root) and exit, "
                          "useful for figuring out --real_depth_col_template")
    args = ap.parse_args()

    if args.print_schema_only:
        print_schema(find_real_parquet(Path(args.real_root)))
        return

    if not args.group_dir:
        ap.error("--group_dir is required unless --print_schema_only is set")
    if not args.sim_replays:
        ap.error("--sim_replays is required unless --print_schema_only is set")

    process_group(args)


if __name__ == "__main__":
    main()
    
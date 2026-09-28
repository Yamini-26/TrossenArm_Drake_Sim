#!/usr/bin/env python
"""Combine per-camera click_roi_compare.py summaries into one overall number.

click_roi_compare.py's summary.json numbers aren't on a common scale across
cameras: each camera has its own click set, its own w direction, and its own
absolute magnitude -- a close-up wrist camera and a wide static overview
don't score the same way even at equal "actual" mismatch. Before combining
across cameras, each metric is min-max normalized against that CAMERA'S OWN
observed range, [0, max_i]. 0 is the true floor (these are absolute-value
mismatch scores, so they can't go below 0); max_i is that camera's own worst
frame, already saved in summary.json. Dividing by max_i turns each metric
into "how bad is this, as a fraction of this camera's own worst moment" -- a
scale every camera shares, regardless of their raw magnitude differences.

Note that "max" itself always normalizes to exactly 1.0 for every camera (a
value divided by itself), so it carries no cross-camera information after
normalization and is left out of the combined report; only the plain mean,
the top-k-time mean, and the excess-over-baseline mean get combined.

Two combinations are reported, over whichever cameras have a summary.json:
  plain mean    -- every present camera weighted equally.
  weighted mean -- cam_right_wrist weighted highest (it's the direct view of
                   the arm actually doing the pick-and-place), cam_high /
                   cam_low next (wide overviews, weaker but still relevant
                   signal), cam_left_wrist last (watching the idle arm in a
                   right-arm task, the least informative view). See WEIGHTS
                   below to change the tiers.

Usage:
    python combine_cam_scores.py --sim-name replay_drop_mid
"""

import argparse
import json
from pathlib import Path

CAMERAS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

WEIGHTS = {
    "cam_right_wrist": 0.5,
    "cam_high": 0.2,
    "cam_low": 0.2,
    "cam_left_wrist": 0.1,
}


def load_camera_summary(sim_analysis_dir: Path, cam: str):
    path = sim_analysis_dir / cam / "summary.json"
    if not path.is_file():
        print(f"  [WARN] {cam}: no summary.json at {path}, skipping")
        return None
    data = json.loads(path.read_text())
    # Key names embed TIME_FRACTION / BASELINE_PERCENTILE (e.g.
    # "top-15%-time mean (severity)"), so match by prefix rather than an
    # exact key in case those constants change between runs.
    mean_key = next(k for k in data if k.startswith("mean"))
    max_key = next(k for k in data if k.startswith("max"))
    top_key = next(k for k in data if k.startswith("top-"))
    excess_key = next(k for k in data if k.startswith("excess-over-"))
    return {
        "mean": data[mean_key],
        "max": data[max_key],
        "top": data[top_key],
        "excess": data[excess_key],
    }


def normalize(raw: dict) -> dict:
    """Min-max normalize mean/top/excess against this camera's own [0, max]."""
    scale = raw["max"] if raw["max"] > 0 else 1.0
    return {
        "mean": raw["mean"] / scale,
        "top": raw["top"] / scale,
        "excess": raw["excess"] / scale,
    }


def combine(values: dict, weights=None) -> dict:
    """values: {cam: {metric: normalized_value}}. weights: {cam: weight}, or
    None for a plain mean. Weights are renormalized over the cameras
    actually present, so a missing camera doesn't just zero out its share."""
    cams = list(values.keys())
    if weights is None:
        weights = {c: 1.0 for c in cams}
    total_weight = sum(weights.get(c, 0.0) for c in cams)

    return {
        metric: sum(weights.get(c, 0.0) * values[c][metric] for c in cams) / total_weight
        for metric in ("mean", "top", "excess")
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-name", type=str, required=True,
                    help="Basename of the sim run used in click_roi_compare.py's "
                         "--sim-dir (e.g. 'replay_drop_mid'), used to locate "
                         "click_analysis/real_vs_<sim-name>/<cam>/summary.json.")
    return p.parse_args()


def main():
    args = parse_args()
    sim_analysis_dir = Path(f"click_analysis/real_vs_{args.sim_name}")

    raw, normalized = {}, {}
    for cam in CAMERAS:
        loaded = load_camera_summary(sim_analysis_dir, cam)
        if loaded is None:
            continue
        raw[cam] = loaded
        normalized[cam] = normalize(loaded)

    if not normalized:
        raise SystemExit(f"No camera summaries found under {sim_analysis_dir}")

    print("\nPer-camera normalized scores (fraction of that camera's own worst frame):")
    for cam, vals in normalized.items():
        print(f"  {cam:16s} mean={vals['mean']:.4f}  top-k-time={vals['top']:.4f}  "
              f"excess={vals['excess']:.4f}")

    plain = combine(normalized, weights=None)
    weighted = combine(normalized, weights=WEIGHTS)

    print(f"\nCombined -- plain mean over {len(normalized)} camera(s):")
    for metric, value in plain.items():
        print(f"  {metric}: {value:.4f}")

    used_weights = {c: WEIGHTS[c] for c in normalized}
    print(f"\nCombined -- weighted ({used_weights}):")
    for metric, value in weighted.items():
        print(f"  {metric}: {value:.4f}")

    result = {
        "cameras_used": list(normalized.keys()),
        "per_camera_normalized": normalized,
        "combined_plain_mean": plain,
        "combined_weighted_mean": weighted,
        "weights_used": used_weights,
    }
    out_path = sim_analysis_dir / "combined_summary.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

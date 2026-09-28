"""Measure sim-to-real mismatch along a direction in DINOv3 feature space that
you pick out yourself, by clicking on the objects you actually care about.

Comparing whole patch embeddings drowns the objects in the background they sit
on. Clicking a few of them instead singles out one direction in feature space,
and the whole trajectory is then compared only along that direction.

Run with --pick to (re)label; otherwise the last labels are reused. To try a
different DINOv3 model, edit MODEL_NAME in dinov3.py and rerun -- the patch
grid and everything downstream follows automatically.

--cam and --sim-dir pick which camera and which sim run to compare against
the real episode; the output folder name follows --sim-dir automatically
(e.g. --sim-dir simulation_frames/replay_drop_mid -> click_analysis/real_vs_replay_drop_mid/<cam>),
so different sim variants never overwrite each other's clicks/plots.

Re-deriving w from fresh clicks on every scenario means every comparison uses
a slightly different, possibly domain- or scenario-biased direction, which
makes scenario-to-scenario comparisons meaningless. --save-w / --load-w fix
that: calibrate once, by clicking on a scenario where the object is clearly
and correctly visible in BOTH sim and real (e.g. a successful pick), with
--save-w pointing at a .pt file; then for every other scenario, skip
clicking entirely and pass --load-w that same file, so every comparison is
read along the exact same frozen direction. (Copying clicks.json between
--sim-dir runs does NOT do this -- "sim"/"real" labels there get resolved
against whatever --sim-dir/--real-dir the CURRENT run points at, silently
re-deriving w from the new scenario's own images instead of the original
calibration ones.)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from dinov3 import on_grid, patch_at, patch_features


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", type=str, default="cam_high",
                    help="Camera subfolder name shared by both frame directories.")
    p.add_argument("--sim-dir", type=str, default="simulation_frames/replay_drop_mid",
                    help="Sim run directory (contains a <cam> frames subfolder). "
                         "Its basename also names the output folder.")
    p.add_argument("--real-dir", type=str, default="real_data/pick_place_depth_3/frames",
                    help="Real episode frames directory (contains a <cam> frames subfolder).")
    p.add_argument("--pick", action="store_true",
                    help="(Re)label by clicking, even if clicks.json already exists.")
    p.add_argument("--save-w", type=str, default=None,
                    help="After deriving w from clicks, save it to this .pt path so "
                         "other scenarios can reuse the exact same direction via "
                         "--load-w instead of re-clicking (and re-deriving w) each time.")
    p.add_argument("--load-w", type=str, default=None,
                    help="Skip clicking/deriving w entirely and load a direction "
                         "previously written by --save-w. Use this for every "
                         "scenario except the one clean baseline you calibrate on.")
    return p.parse_args()


args = parse_args()
if args.save_w and args.load_w:
    raise SystemExit("--save-w and --load-w are mutually exclusive: --save-w derives "
                      "a new w from clicks and stores it; --load-w skips deriving one "
                      "entirely and reuses a previously saved file.")

CAM = args.cam
SIM_DIR = Path(args.sim_dir)
REAL_DIR = Path(args.real_dir)

OUTPUT_DIR = Path(f"click_analysis/real_vs_{SIM_DIR.name}/{CAM}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODES = {
    "sim":  SIM_DIR / CAM,
    "real": REAL_DIR / CAM,
}
FRAMES = [1, 40, 45, 70]  # the frames you label on, and the ones drawn as maps
TRAJECTORY = range(0, 76)  # the frames swept for the mismatch-over-time curves
OBJECT_FRACTION = 0.1  # the top of the directional score counted as "the object"
TIME_FRACTION = 0.15  # the worst fraction of frames counted by the top-k-time summary
BASELINE_PERCENTILE = 25  # the percentile of object_mismatch treated as "noise floor"
CLICKS = OUTPUT_DIR / "clicks.json"


def load(label, frame):
    return Image.open(EPISODES[label] / f"frame_{frame:05d}.png").convert("RGB")


panels = [(label, frame) for label in EPISODES for frame in FRAMES]
images = {panel: load(*panel) for panel in panels}  # still needed below, for the plots


def collect_clicks():
    """Show the representative frames and let the user click the objects on them."""
    picked, markers = [], []
    figure, axes = plt.subplots(len(EPISODES), len(FRAMES), layout="constrained",
                                figsize=(3 * len(FRAMES), 2.6 * len(EPISODES)))
    figure.suptitle("click the objects of interest, right-click to undo, "
                    "then close the window")
    owner = {}
    for axis, panel in zip(axes.flat, panels):
        axis.imshow(images[panel])
        axis.set_title(f"{panel[0]} frame {panel[1]}", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        owner[axis] = panel

    def on_click(event):
        if event.inaxes not in owner or event.xdata is None:
            return
        if event.button == 3:  # right click takes back the last click on this panel
            for index in reversed(range(len(picked))):
                if picked[index][0] == owner[event.inaxes]:
                    picked.pop(index)
                    markers.pop(index).remove()
                    break
        else:
            picked.append((owner[event.inaxes], patch_at(event.xdata, event.ydata)))
            markers.append(event.inaxes.plot(event.xdata, event.ydata, "o",
                                             color="lime", markeredgecolor="black")[0])
        event.canvas.draw_idle()

    figure.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    return picked


if args.load_w:
    w = torch.load(args.load_w)
    print(f"Loaded w <- {args.load_w} (skipping clicks entirely)")
else:
    features = {panel: patch_features(image) for panel, image in images.items()}

    if args.pick or not CLICKS.exists():
        clicks = collect_clicks()
        if not clicks:
            raise SystemExit("no clicks, nothing to compare along")
        CLICKS.write_text(json.dumps([[label, frame, patch]
                                      for (label, frame), patch in clicks]))
    else:
        clicks = [((label, frame), patch)
                  for label, frame, patch in json.loads(CLICKS.read_text())]

    print(f"{len(clicks)} clicks over {len({panel for panel, _ in clicks})} frames")

    # The direction is the clicked patches minus the average patch, not the clicked
    # patches themselves: DINOv3 features share a large common component describing
    # the scene as a whole, and taking it out is what leaves a direction that is
    # about the objects rather than about everything they are sitting on.
    clicked = torch.stack([features[panel][patch] for panel, patch in clicks]).float()
    average = torch.cat([features[panel] for panel in panels]).float().mean(0)
    w = clicked.mean(0) - average
    w = w / w.norm()

    if args.save_w:
        save_path = Path(args.save_w)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(w, save_path)
        print(f"Saved w -> {save_path}")

# Sweep both episodes in lockstep, keeping only per-patch scores along w
maps, mismatch, object_mismatch, baseline = {}, [], [], []

for frame in TRAJECTORY:
    z = {label: patch_features(load(label, frame)).float() for label in EPISODES}
    score = {label: z[label] @ w for label in EPISODES}
    difference = (score["sim"] - score["real"]).abs()

    # The object is wherever either episode scores highest along the user's
    # direction, which is the only place the mismatch is meant to be read.
    salience = torch.maximum(score["sim"], score["real"])
    count = max(int(OBJECT_FRACTION * salience.numel()), 1)
    on_object = difference[salience.topk(count).indices]

    mismatch.append(difference.mean().item())
    object_mismatch.append(on_object.mean().item())

    # What comparing the whole embedding gives you, for reference
    similarity = torch.nn.functional.cosine_similarity(z["sim"], z["real"], dim=-1)
    baseline.append(1 - similarity.mean().item())

    if frame in FRAMES:
        maps[frame] = score

# ---- boiling object_mismatch down to one number -----------------------------
#
# object_mismatch[t] is already the whole story: exactly where and how much
# real and sim disagree, frame by frame. Two ways to collapse it to a single
# number, kept side by side because they answer different questions and a
# plain mean answers neither well -- it lets a long well-matched stretch
# dilute a short severe one down to nothing.
#
# top-k-time mean: sort the per-frame scores, average only the worst
# TIME_FRACTION of them -- the same trick OBJECT_FRACTION already applies
# along space (only average the most salient patches), just applied along
# time instead:
#     d(t) = object_mismatch[t],  t = 1..T
#     d_(1) >= d_(2) >= ... >= d_(T)          (sorted descending)
#     k = max(round(TIME_FRACTION * T), 1)
#     M_topk = (1/k) * sum_{i=1..k} d_(i)
# Answers "how bad does it get" -- a severity metric. Sensitive to peak
# height, blind to exactly how long the worst period lasts: a 2-frame spike
# and a 20-frame plateau of the same height score identically as long as
# both clear the worst-k cut.
#
# excess-over-baseline mean: first estimate a per-episode noise floor (the
# BASELINE_PERCENTILE-th percentile of d(t) itself -- "how much mismatch is
# there even when nothing is really wrong"), subtract it off, clip negative
# values to zero, then average THAT excess over the WHOLE trajectory rather
# than just the worst slice:
#     floor  = percentile_{BASELINE_PERCENTILE}( d(t) )
#     e(t)   = max(0, d(t) - floor)
#     M_excess = (1/T) * sum_{t=1..T} e(t)
# Answers "how much total divergence accumulated" -- a duration-weighted
# metric. A real event that drags on for many frames pushes this up more
# than an equally tall 1-frame blip, because it adds more nonzero terms to
# the sum; a trajectory that matches throughout scores near zero regardless
# of length, because floor absorbs the baseline domain-gap noise.
d = torch.tensor(object_mismatch)
T = d.numel()

k = max(round(TIME_FRACTION * T), 1)
m_topk = d.topk(k).values.mean().item()

floor = torch.quantile(d, BASELINE_PERCENTILE / 100).item()
m_excess = (d - floor).clamp(min=0).mean().item()

summary = {
    "mean (plain average, for reference)": d.mean().item(),
    "max (single worst frame)": d.max().item(),
    f"top-{TIME_FRACTION:.0%}-time mean (severity)": m_topk,
    f"excess-over-p{BASELINE_PERCENTILE} mean (duration-weighted)": m_excess,
}
print("\n--- single-number summaries (object_mismatch) ---")
for name, value in summary.items():
    print(f"  {name}: {value:.4f}")
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

# ---- the per-frame maps ----------------------------------------------------

figure, axes = plt.subplots(5, len(FRAMES), layout="constrained",
                            figsize=(3 * len(FRAMES), 11))
figure.suptitle("features projected onto the clicked direction")
for axis, frame in zip(axes[0], FRAMES):
    axis.set_title(f"frame {frame}", fontsize=10)
for axis in axes.flat:
    axis.set_xticks([])
    axis.set_yticks([])


def row(index, label, panels_or_fields, cmap=None, limits=None):
    for axis, field in zip(axes[index], panels_or_fields):
        drawn = axis.imshow(field, cmap=cmap,
                            vmin=None if limits is None else limits[0],
                            vmax=None if limits is None else limits[1])
    axes[index][0].set_ylabel(label, fontsize=10)
    if cmap is not None:
        figure.colorbar(drawn, ax=axes[index], fraction=0.02)


# Both episodes' score maps share one scale, and the difference is symmetric
# about zero, so the panels can be read against each other
scores = [value for frame in FRAMES for value in maps[frame].values()]
low = min(score.min().item() for score in scores)
high = max(score.max().item() for score in scores)
gap = max(abs(maps[frame]["sim"] - maps[frame]["real"]).max().item() for frame in FRAMES)

row(0, "sim", [images[("sim", frame)] for frame in FRAMES])
row(1, "z_sim . w", [on_grid(maps[frame]["sim"]) for frame in FRAMES],
    cmap="inferno", limits=(low, high))
row(2, "real", [images[("real", frame)] for frame in FRAMES])
row(3, "z_real . w", [on_grid(maps[frame]["real"]) for frame in FRAMES],
    cmap="inferno", limits=(low, high))
row(4, "difference", [on_grid(maps[frame]["sim"] - maps[frame]["real"])
                      for frame in FRAMES], cmap="coolwarm", limits=(-gap, gap))
figure.savefig(OUTPUT_DIR / "feature_projection.png", dpi=150)

# ---- mismatch over the whole trajectory ------------------------------------

figure, (top, bottom) = plt.subplots(2, 1, sharex=True, layout="constrained",
                                     figsize=(8, 6))
figure.suptitle("sim-to-real mismatch over the episode")

top.plot(TRAJECTORY, baseline, color="tab:grey")
top.set_ylabel("1 - cos(z_sim, z_real)")
top.set_title("whole embedding, every patch", fontsize=10)

bottom.plot(TRAJECTORY, object_mismatch, color="tab:red", label="on the object")
bottom.plot(TRAJECTORY, mismatch, color="tab:red", alpha=0.3, label="every patch")
bottom.axhline(m_topk, color="tab:purple", linestyle="--", linewidth=1,
               label=f"top-{TIME_FRACTION:.0%}-time mean = {m_topk:.4f}")
bottom.axhline(floor, color="tab:grey", linestyle=":", linewidth=1,
               label=f"p{BASELINE_PERCENTILE} floor = {floor:.4f}")
bottom.set_ylabel("|z_sim . w - z_real . w|")
bottom.set_xlabel("frame")
bottom.set_title(f"along the clicked direction  "
                  f"(excess-over-baseline mean = {m_excess:.4f})", fontsize=10)
bottom.legend(fontsize=9)
figure.savefig(OUTPUT_DIR / "mismatch_over_episode.png", dpi=150)

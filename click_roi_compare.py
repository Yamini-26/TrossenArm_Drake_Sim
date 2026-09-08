"""Measure sim-to-real mismatch along a direction in DINOv3 feature space."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

CAM = "cam_high"
OUTPUT_DIR = Path(f"click_analysis/real_vs_gn/{CAM}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REAL_PATCH = Path(f"dino_features/temporal/real/{CAM}_patch.npy")
SIM_PATCH  = Path(f"dino_features/temporal/sim_grab_neutral/{CAM}_patch.npy")    # (T, N, D1)

EPISODES = {
    "sim":  Path(f"data/pick_place_depth_3/frames/{CAM}/"),
    "real": Path(f"simulation_frames/replay_1786561243/{CAM}"),
}
FRAMES          = [1, 40, 45, 70]
TRAJECTORY      = range(0, 76)
OBJECT_FRACTION = 0.1
CLICKS          = OUTPUT_DIR / "clicks.json"

# ---- grid shape — must match what dino_feature_extractor used ---------------
GRID         = (32, 32)    # DINOv3 vitb16 512×512  →  32×32 patches
INPUT_HEIGHT = 512
INPUT_WIDTH  = 512

def patch_at(x, y):
    row = min(int(y / INPUT_HEIGHT * GRID[0]), GRID[0] - 1)
    col = min(int(x / INPUT_WIDTH  * GRID[1]), GRID[1] - 1)
    return row * GRID[1] + col

def on_grid(values):
    return values.float().reshape(*GRID).cpu().numpy()

# ---- load pre-saved features once at startup --------------------------------
print("Loading features ...")
_real = np.load(REAL_PATCH)   # (T, N, D)
_sim  = np.load(SIM_PATCH)    # (T, N, D)

def load(label, frame):
    return Image.open(EPISODES[label] / f"frame_{frame:05d}.png").convert("RGB")

def get_features(label, frame):
    arr = _real[frame] if label == "real" else _sim[frame]
    return torch.from_numpy(arr).float()

# ---- identical to original from here down -----------------------------------
panels   = [(label, frame) for label in EPISODES for frame in FRAMES]
images   = {panel: load(*panel) for panel in panels}
features = {panel: get_features(*panel) for panel in panels}

def collect_clicks():
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
        if event.button == 3:
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

if "--pick" in sys.argv or not CLICKS.exists():
    clicks = collect_clicks()
    if not clicks:
        raise SystemExit("no clicks, nothing to compare along")
    CLICKS.write_text(json.dumps([[label, frame, patch]
                                  for (label, frame), patch in clicks]))
else:
    clicks = [((label, frame), patch)
              for label, frame, patch in json.loads(CLICKS.read_text())]

print(f"{len(clicks)} clicks over {len({panel for panel, _ in clicks})} frames")

clicked = torch.stack([features[panel][patch] for panel, patch in clicks]).float()
average = torch.cat([features[panel] for panel in panels]).float().mean(0)
w = clicked.mean(0) - average
w = w / w.norm()

maps, mismatch, object_mismatch, baseline = {}, [], [], []
for frame in TRAJECTORY:
    z     = {label: get_features(label, frame) for label in EPISODES}  # only line that changed
    score = {label: z[label] @ w for label in EPISODES}
    difference = (score["sim"] - score["real"]).abs()
    salience   = torch.maximum(score["sim"], score["real"])
    count      = max(int(OBJECT_FRACTION * salience.numel()), 1)
    on_object  = difference[salience.topk(count).indices]
    mismatch.append(difference.mean().item())
    object_mismatch.append(on_object.mean().item())
    similarity = torch.nn.functional.cosine_similarity(z["sim"], z["real"], dim=-1)
    baseline.append(1 - similarity.mean().item())
    if frame in FRAMES:
        maps[frame] = score

# plots — identical to original
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

scores = [value for frame in FRAMES for value in maps[frame].values()]
low    = min(score.min().item() for score in scores)
high   = max(score.max().item() for score in scores)
gap    = max(abs(maps[frame]["sim"] - maps[frame]["real"]).max().item() for frame in FRAMES)

row(0, "sim",        [images[("sim",  frame)] for frame in FRAMES])
row(1, "z_sim . w",  [on_grid(maps[frame]["sim"])  for frame in FRAMES], cmap="inferno", limits=(low, high))
row(2, "real",       [images[("real", frame)] for frame in FRAMES])
row(3, "z_real . w", [on_grid(maps[frame]["real"]) for frame in FRAMES], cmap="inferno", limits=(low, high))
row(4, "difference", [on_grid(maps[frame]["sim"] - maps[frame]["real"]) for frame in FRAMES],
    cmap="coolwarm", limits=(-gap, gap))
figure.savefig(OUTPUT_DIR / "feature_projection.png", dpi=150)

figure, (top, bottom) = plt.subplots(2, 1, sharex=True, layout="constrained", figsize=(8, 6))
figure.suptitle("sim-to-real mismatch over the episode")
top.plot(TRAJECTORY, baseline, color="tab:grey")
top.set_ylabel("1 - cos(z_sim, z_real)")
top.set_title("whole embedding, every patch", fontsize=10)
bottom.plot(TRAJECTORY, object_mismatch, color="tab:red", label="on the object")
bottom.plot(TRAJECTORY, mismatch,        color="tab:red", alpha=0.3, label="every patch")
bottom.set_ylabel("|z_sim . w - z_real . w|")
bottom.set_xlabel("frame")
bottom.set_title("along the clicked direction", fontsize=10)
bottom.legend(fontsize=9)
figure.savefig(OUTPUT_DIR / "mismatch_over_episode.png", dpi=150)

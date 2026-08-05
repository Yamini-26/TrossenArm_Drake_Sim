#!/usr/bin/env python
"""
Unified DINOv2 / DINOv3 feature comparison tool.

Combines the DINOv2 and DINOv3 models usage into one, driven
by --model_version. Also generalizes the "pairs" input: instead of a fixed
frame/action vocabulary, it just looks for real_*.png / sim_*.png files in a
folder (any number of each, any labels) and compares them.

Three analysis modes, each independently selectable via --analysis:
  cls    - whole-image similarity using the CLS token only
  patch  - whole-image similarity using raw patch tokens (no CLS), plus
           the full spatial similarity heatmap
  worst  - same patch-level similarity, but surfaces + visualizes the
           K worst-matching (lowest similarity) patches per pair, so you
           can see *where* real and sim diverge most

Usage
-----
Compare a folder of real_*/sim_* frames with DINOv2, all analyses:
    python dino_feature_extractor.py --input_dir simulation_frames/test_frames_real/cam_right_wrist --output dino_features/test_frames_real/cam_right_wrist/ --model_version v2

Just CLS-token similarity with DINOv3:
    python dino_feature_extractor.py --input_dir simulation_frames/test_frames_real/cam_low/ --output dino_features/test_frames_real/cam_low/ --model_version v3 --analysis cls

Patch + worst-patch only, keep the top 20 worst patches per pair:
    python dino_feature_extractor.py --input_dir simulation_frames/test_frames_real/cam_high/ --output dino_features/test_frames_real/cam_high/ --model_version v2 --analysis worst --top_k_worst 20

Legacy run/camera-directory mode (single run, no real/sim comparison, just
per-camera self-similarity heatmaps e.g. PCA/norm/cluster over patches):
    python dino_feature_extractor.py --run_dir simulation_frames/replay_1785878156/ --output dino_features/replay_1785878156_v3/ --model_version v3 --save_features --generate_self_heatmaps --heatmap_mode pca
"""

import argparse
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
 
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")
CAMERA_NAMES = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
 
 
#  Model configs (differs between v2/v3)
 
@dataclass
class ModelConfig:
    version: str
    name: str
    input_size: int
    patch_size: int
    dim: int
    hub_id: Optional[str] = None
    hf_id: Optional[str] = None
 
    @property
    def grid_size(self) -> int:
        return self.input_size // self.patch_size
 
 
MODEL_CONFIGS = {
    "v2": ModelConfig(version="v2", name="dinov2_vitb14", input_size=518,
                       patch_size=14, dim=768, hub_id="dinov2_vitb14"),
    "v3": ModelConfig(version="v3", name="dinov3_vitb16", input_size=512,
                       patch_size=16, dim=768,
                       hf_id="facebook/dinov3-vitb16-pretrain-lvd1689m"),
}
 
 
def load_model(version: str, device: str) -> Tuple[torch.nn.Module, ModelConfig]:
    cfg = MODEL_CONFIGS[version]
    print(f"Loading {cfg.name} ({version}) ...")
    if version == "v2":
        model = torch.hub.load("facebookresearch/dinov2", cfg.hub_id,
                                trust_repo=True, force_reload=False)
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(cfg.hf_id)
    model.eval().to(device)
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, cfg
 
 
def make_transform(cfg: ModelConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(cfg.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
 
 
@torch.no_grad()
def forward_batch(model, cfg: ModelConfig, batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (cls (B, dim), patch (B, N, dim)) regardless of v2/v3 API differences."""
    if cfg.version == "v2":
        out = model.forward_features(batch)
        cls, patch = out["x_norm_clstoken"], out["x_norm_patchtokens"]
    else:
        num_register_tokens = getattr(model.config, "num_register_tokens", 0)
        out = model(pixel_values=batch)
        hidden = out.last_hidden_state
        cls = hidden[:, 0]
        patch = hidden[:, 1 + num_register_tokens:]
    return cls.cpu().float().numpy(), patch.cpu().float().numpy()
 
 
@torch.no_grad()
def extract_features(model, cfg: ModelConfig, files: List[Path], transform,
                      device: str, batch_size: int = 8) -> Dict:
    """Extract CLS + patch features for an explicit list of image paths."""
    if not files:
        raise FileNotFoundError("No image files provided")
    print(f"  Extracting {len(files)} frame(s) with {cfg.name} ...")
 
    cls_list, patch_list = [], []
    n_batches = (len(files) - 1) // batch_size + 1
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        imgs = [transform(Image.open(f).convert("RGB")) for f in batch_files]
        batch = torch.stack(imgs).to(device)
        cls, patch = forward_batch(model, cfg, batch)
        cls_list.append(cls)
        patch_list.append(patch)
        print(f"    batch {i // batch_size + 1} / {n_batches}")
 
    return {
        "cls": np.concatenate(cls_list, axis=0),     # (N, dim)
        "patch": np.concatenate(patch_list, axis=0), # (N, n_patches, dim)
        "files": [f.name for f in files],
    }
 
 
#  Generalized real_*/sim_* input discovery
 
# real_grab.png / sim_grab.png / sim_grab_light.png / sim_drop_dim_v2.png ...
# source is required; "action" is the first token after the source and is
# used to group same-content images together; anything after that is an
# open-ended variant tag and does NOT need to match a fixed vocabulary.
FILE_LABEL_RE = re.compile(r"^(?P<source>real|sim)_(?P<action>[a-zA-Z0-9]+)(?:_(?P<variant>.+))?$")
 
 
def discover_images(input_dir: Path) -> Tuple[List[Path], Dict[str, dict]]:
    """
    Scan input_dir for real_*/sim_* images (any extension in IMAGE_EXTS).
    Returns (sorted file list, {filename: {source, action, variant}}).
    New images -- more real_*, more sim_*, new action labels -- just work;
    nothing about the vocabulary is hardcoded beyond the real_/sim_ prefix.
    """
    candidates = []
    for pattern in IMAGE_EXTS:
        candidates.extend(input_dir.glob(pattern))
    candidates = sorted(set(candidates))
 
    files, meta = [], {}
    for f in candidates:
        m = FILE_LABEL_RE.match(f.stem)
        if not m:
            print(f"  [WARN] '{f.name}' doesn't match real_<label>/sim_<label>, skipping")
            continue
        meta[f.name] = {
            "source": m.group("source"),
            "action": m.group("action"),
            "variant": m.group("variant"),
        }
        files.append(f)
 
    if not files:
        raise FileNotFoundError(f"No real_*/sim_* images found in {input_dir}")
 
    n_real = sum(1 for m in meta.values() if m["source"] == "real")
    n_sim = sum(1 for m in meta.values() if m["source"] == "sim")
    print(f"  Found {len(files)} image(s): {n_real} real_*, {n_sim} sim_* "
          f"across {len(set(m['action'] for m in meta.values()))} action label(s)")
    return files, meta
 
 
def build_pairs(meta: Dict[str, dict]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, bool]], Dict[str, List[str]]]:
    """
    Build the two sets of comparisons we care about:
      - within_pairs: every combination within the same action label
      - anchor_pairs: every real_* image vs every sim_* image, REGARDLESS of
                       action label, so real_grab vs sim_drop gets checked
    """
    groups: Dict[str, List[str]] = {}
    for name, m in meta.items():
        groups.setdefault(m["action"], []).append(name)
 
    within_pairs = []
    for action, names in groups.items():
        if len(names) < 2:
            print(f"  [WARN] action '{action}' has only 1 image — no within-group comparison")
            continue
        within_pairs.extend(itertools.combinations(sorted(names), 2))
 
    reals = sorted(n for n, m in meta.items() if m["source"] == "real")
    sims = sorted(n for n, m in meta.items() if m["source"] == "sim")
    if not reals:
        print("  [INFO] no real_* images found — skipping real-vs-sim anchor comparison")
    anchor_pairs = [(r, s, meta[r]["action"] == meta[s]["action"]) for r in reals for s in sims]
 
    return within_pairs, anchor_pairs, groups
 
 
#  Similarity primitives
 
def cls_cosine_sim(cls_a: np.ndarray, cls_b: np.ndarray) -> float:
    """Whole-image similarity from the CLS token only."""
    a = cls_a / (np.linalg.norm(cls_a) + 1e-8)
    b = cls_b / (np.linalg.norm(cls_b) + 1e-8)
    return float((a * b).sum())
 
 
def patch_heatmap(patch_a: np.ndarray, patch_b: np.ndarray, grid: int) -> np.ndarray:
    """Spatial cosine-similarity heatmap from raw patch tokens (no CLS)."""
    a = patch_a / (np.linalg.norm(patch_a, axis=1, keepdims=True) + 1e-8)
    b = patch_b / (np.linalg.norm(patch_b, axis=1, keepdims=True) + 1e-8)
    sim = (a * b).sum(axis=1)
    return sim.reshape(grid, grid)
 
 
def mean_patch_sim(heatmap: np.ndarray) -> float:
    """Whole-image similarity from raw patches (alternative to the CLS metric)."""
    return float(heatmap.mean())
 
 
def worst_patches(heatmap: np.ndarray, top_k: int) -> List[Tuple[int, int, float]]:
    """Locations (row, col) of the K lowest-similarity patches, ascending."""
    flat = heatmap.flatten()
    k = min(top_k, flat.size)
    idx = np.argsort(flat)[:k]
    w = heatmap.shape[1]
    return [(int(i // w), int(i % w), float(flat[i])) for i in idx]
 
 
def full_cls_matrix(feats: Dict) -> np.ndarray:
    cls = feats["cls"]
    norm = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-8)
    return norm @ norm.T
 
 
def full_patch_matrix(feats: Dict, cfg: ModelConfig) -> np.ndarray:
    """NxN matrix of mean raw-patch cosine similarity between every pair of
    images (the patch-mode analogue of full_cls_matrix)."""
    patches = feats["patch"]
    n = len(feats["files"])
    matrix = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            heat = patch_heatmap(patches[i], patches[j], cfg.grid_size)
            m = mean_patch_sim(heat)
            matrix[i, j] = matrix[j, i] = m
    return matrix
 
 
def full_worst_matrix(feats: Dict, cfg: ModelConfig, top_k: int) -> np.ndarray:
    """NxN matrix of worst-K-patch mean similarity between every pair of
    images (the worst-patch-mode analogue of full_cls_matrix)."""
    patches = feats["patch"]
    n = len(feats["files"])
    matrix = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            heat = patch_heatmap(patches[i], patches[j], cfg.grid_size)
            wm = float(np.mean([w[2] for w in worst_patches(heat, top_k)]))
            matrix[i, j] = matrix[j, i] = wm
    return matrix
 
 
#  Plotting
 
def plot_bar(pair_labels: List[str], values: List[float], title: str,
             ylabel: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(pair_labels)), 4.5))
    ax.bar(range(len(pair_labels)), values, color="steelblue")
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")
 
 
def plot_similarity_matrix(matrix: np.ndarray, labels: List[str], output_path: Path,
                            title: str = "CLS cosine similarity"):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")
 
 
def plot_patch_heatmap_pair(img_a_path: Path, img_b_path: Path, heatmap: np.ndarray,
                             metric_label: str, metric_value: float, cfg: ModelConfig,
                             output_path: Path):
    img_a = Image.open(img_a_path).convert("RGB").resize((cfg.input_size, cfg.input_size))
    img_b = Image.open(img_b_path).convert("RGB").resize((cfg.input_size, cfg.input_size))
 
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img_a); axes[0].set_title(img_a_path.name, fontsize=9); axes[0].axis("off")
    axes[1].imshow(img_b); axes[1].set_title(img_b_path.name, fontsize=9); axes[1].axis("off")
    im = axes[2].imshow(heatmap, vmin=0, vmax=1, cmap="RdYlGn", origin="upper")
    axes[2].set_title(f"patch cos sim ({metric_label}={metric_value:.3f})", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")
 
 
def plot_worst_patch_pair(img_a_path: Path, img_b_path: Path, heatmap: np.ndarray,
                           worst: List[Tuple[int, int, float]], cfg: ModelConfig,
                           output_path: Path):
    """Side-by-side images + heatmap, with the K worst patches boxed on both
    the heatmap and directly on image B (in pixel space) so you can see
    exactly where real and sim diverge most."""
    img_a = Image.open(img_a_path).convert("RGB").resize((cfg.input_size, cfg.input_size))
    img_b = Image.open(img_b_path).convert("RGB").resize((cfg.input_size, cfg.input_size))
    patch_px = cfg.input_size / cfg.grid_size
 
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(img_a); axes[0].set_title(img_a_path.name, fontsize=9); axes[0].axis("off")
 
    axes[1].imshow(img_b); axes[1].set_title(f"{img_b_path.name} (worst patches boxed)", fontsize=9)
    axes[1].axis("off")
    for (r, c, s) in worst:
        rect = plt.Rectangle((c * patch_px, r * patch_px), patch_px, patch_px,
                              fill=False, edgecolor="blue", linewidth=1.5)
        axes[1].add_patch(rect)
 
    im = axes[2].imshow(heatmap, vmin=0, vmax=1, cmap="RdYlGn", origin="upper")
    axes[2].set_title(f"worst {len(worst)} patches (min={worst[0][2]:.3f})", fontsize=9)
    axes[2].axis("off")
    for (r, c, s) in worst:
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="blue", linewidth=1.5)
        axes[2].add_patch(rect)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {output_path}")
 
 
#  Analysis modes
 
def run_cls_analysis(feats: Dict, file_to_idx: Dict, within_pairs, anchor_pairs,
                      groups: Dict[str, List[str]], output_dir: Path) -> Dict:
    print("\n=== CLS-token analysis ===")
    within_results = {}
    for a, b in within_pairs:
        sim = cls_cosine_sim(feats["cls"][file_to_idx[a]], feats["cls"][file_to_idx[b]])
        within_results[f"{Path(a).stem}__vs__{Path(b).stem}"] = sim
        print(f"  {a} vs {b} -> CLS cos sim = {sim:.4f}")
    if within_results:
        plot_bar(list(within_results.keys()), list(within_results.values()),
                  "CLS similarity (within action label)", "CLS cosine similarity",
                  output_dir / "cls_within_group_bar.png")
 
    anchor_results = {}
    for r, s, match in anchor_pairs:
        sim = cls_cosine_sim(feats["cls"][file_to_idx[r]], feats["cls"][file_to_idx[s]])
        anchor_results[f"{Path(r).stem}__vs__{Path(s).stem}"] = {"cls_cosine_sim": sim, "match": match}
        print(f"  [{'match   ' if match else 'mismatch'}] {r} vs {s} -> CLS cos sim = {sim:.4f}")
    if anchor_results:
        keys = sorted(anchor_results.keys())
        plot_bar(keys, [anchor_results[k]["cls_cosine_sim"] for k in keys],
                  "CLS similarity: real_* vs every sim_*", "CLS cosine similarity",
                  output_dir / "cls_real_vs_sim_bar.png")
 
    print("  Computing full NxN CLS similarity matrix ...")
    matrix = full_cls_matrix(feats)
    labels = [Path(n).stem for n in feats["files"]]
    plot_similarity_matrix(matrix, labels, output_dir / "cls_similarity_matrix.png",
                            title="CLS cosine similarity (all images)")
 
    return {"within_group": within_results, "real_vs_sim": anchor_results,
            "full_matrix": matrix.tolist(), "full_matrix_labels": labels}
 
 
def run_patch_analysis(feats: Dict, file_to_idx: Dict, within_pairs, anchor_pairs,
                        cfg: ModelConfig, output_dir: Path) -> Dict:
    print("\n=== Raw-patch analysis (no CLS token) ===")
    patch_dir = output_dir / "patch_heatmaps"
    patch_dir.mkdir(exist_ok=True)
 
    def compare(a_name, b_name):
        pa, pb = feats["patch"][file_to_idx[a_name]], feats["patch"][file_to_idx[b_name]]
        heat = patch_heatmap(pa, pb, cfg.grid_size)
        return heat, mean_patch_sim(heat)
 
    within_results = {}
    for a, b in within_pairs:
        heat, m = compare(a, b)
        key = f"{Path(a).stem}__vs__{Path(b).stem}"
        within_results[key] = m
        print(f"  {a} vs {b} -> mean patch cos sim = {m:.4f}")
        plot_patch_heatmap_pair(files_lookup[a], files_lookup[b], heat, "mean_patch", m,
                                 cfg, patch_dir / f"{key}_patch.png")
 
    anchor_results = {}
    for r, s, match in anchor_pairs:
        heat, m = compare(r, s)
        key = f"{Path(r).stem}__vs__{Path(s).stem}"
        anchor_results[key] = {"mean_patch_cosine_sim": m, "match": match}
        print(f"  [{'match   ' if match else 'mismatch'}] {r} vs {s} -> mean patch cos sim = {m:.4f}")
        plot_patch_heatmap_pair(files_lookup[r], files_lookup[s], heat, "mean_patch", m,
                                 cfg, patch_dir / f"{key}_patch.png")
 
    if within_results:
        plot_bar(list(within_results.keys()), list(within_results.values()),
                  "Mean patch similarity (within action label)", "mean patch cosine similarity",
                  output_dir / "patch_within_group_bar.png")
    if anchor_results:
        keys = sorted(anchor_results.keys())
        plot_bar(keys, [anchor_results[k]["mean_patch_cosine_sim"] for k in keys],
                  "Mean patch similarity: real_* vs every sim_*", "mean patch cosine similarity",
                  output_dir / "patch_real_vs_sim_bar.png")
 
    print("  Computing full NxN mean-patch similarity matrix ...")
    matrix = full_patch_matrix(feats, cfg)
    labels = [Path(n).stem for n in feats["files"]]
    plot_similarity_matrix(matrix, labels, output_dir / "patch_similarity_matrix.png",
                            title="Mean patch cosine similarity (all images)")
 
    return {"within_group": within_results, "real_vs_sim": anchor_results,
            "full_matrix": matrix.tolist(), "full_matrix_labels": labels}
 
 
def run_worst_patch_analysis(feats: Dict, file_to_idx: Dict, within_pairs, anchor_pairs,
                              cfg: ModelConfig, top_k: int, output_dir: Path) -> Dict:
    print(f"\n=== Worst-patch analysis (bottom {top_k} patches per pair) ===")
    worst_dir = output_dir / "worst_patch_heatmaps"
    worst_dir.mkdir(exist_ok=True)
 
    def compare(a_name, b_name):
        pa, pb = feats["patch"][file_to_idx[a_name]], feats["patch"][file_to_idx[b_name]]
        heat = patch_heatmap(pa, pb, cfg.grid_size)
        worst = worst_patches(heat, top_k)
        worst_mean = float(np.mean([w[2] for w in worst]))
        return heat, worst, worst_mean
 
    results = {}
    # Worst-patch analysis is most useful on the real-vs-sim anchor pairs, but within-group pairs are included too for completeness.
    all_pairs = [(a, b, None) for a, b in within_pairs] + list(anchor_pairs)
    for a, b, match in all_pairs:
        heat, worst, worst_mean = compare(a, b)
        key = f"{Path(a).stem}__vs__{Path(b).stem}"
        results[key] = {
            "worst_patch_mean_sim": worst_mean,
            "worst_patches": [{"row": r, "col": c, "sim": s} for r, c, s in worst],
            "match": match,
        }
        tag = "" if match is None else ("match   " if match else "mismatch")
        print(f"  [{tag}] {a} vs {b} -> worst-{top_k} mean sim = {worst_mean:.4f} "
              f"(single worst patch = {worst[0][2]:.4f})")
        plot_worst_patch_pair(files_lookup[a], files_lookup[b], heat, worst, cfg,
                               worst_dir / f"{key}_worst.png")
 
    if results:
        keys = sorted(results.keys(), key=lambda k: results[k]["worst_patch_mean_sim"])
        plot_bar(keys, [results[k]["worst_patch_mean_sim"] for k in keys],
                  f"Worst-{top_k}-patch mean similarity (lower = diverges more)",
                  "worst-patch mean cosine similarity",
                  output_dir / "worst_patch_ranking_bar.png")
        print("\n  Pairs ranked worst-match-first:")
        for k in keys[:10]:
            print(f"    {k}: {results[k]['worst_patch_mean_sim']:.4f}")
 
    print("  Computing full NxN worst-patch similarity matrix ...")
    matrix = full_worst_matrix(feats, cfg, top_k)
    labels = [Path(n).stem for n in feats["files"]]
    plot_similarity_matrix(matrix, labels, output_dir / "worst_patch_similarity_matrix.png",
                            title=f"Worst-{top_k}-patch mean similarity (all images)")
 
    return {"pairs": results, "full_matrix": matrix.tolist(), "full_matrix_labels": labels}
 
 
#  Legacy single-run / camera-directory mode
#  (no real/sim comparison -- just self-similarity heatmaps per cam)
 
def resolve_camera_files(run_dir: Path, cam: str, frame_indices: Optional[List[int]]) -> List[Path]:
    cam_dir = run_dir / cam
    all_files = sorted(f for pattern in IMAGE_EXTS for f in cam_dir.glob(pattern))
    if frame_indices is None:
        return all_files
    selected = []
    for idx in frame_indices:
        if 0 <= idx < len(all_files):
            selected.append(all_files[idx])
        else:
            print(f"  [WARN] frame index {idx} out of range for {cam} ({len(all_files)} frames)")
    return selected
 
 
def visualize_self_heatmap(image_path: Path, patch_features: np.ndarray, cfg: ModelConfig,
                            output_path: Path, mode: str = "pca"):
    img = Image.open(image_path).convert("RGB").resize((cfg.input_size, cfg.input_size),
                                                          Image.Resampling.LANCZOS)
    grid = cfg.grid_size
 
    if mode == "pca":
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(patch_features).flatten()
        pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-8)
        heatmap, title, cmap = pc1.reshape(grid, grid), "PCA - first component", "viridis"
    elif mode == "norm":
        norms = np.linalg.norm(patch_features, axis=1)
        norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
        heatmap, title, cmap = norms.reshape(grid, grid), "Patch activation strength", "hot"
    elif mode == "cluster":
        from sklearn.cluster import KMeans
        n_clusters = 8
        clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(patch_features)
        heatmap, title, cmap = clusters.reshape(grid, grid), f"Semantic clusters ({n_clusters})", "tab10"
    else:
        raise ValueError(f"Unknown mode: {mode}")
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img); ax1.set_title("Original"); ax1.axis("off")
    im = ax2.imshow(img)
    ax2.imshow(heatmap, cmap=cmap, alpha=0.6, interpolation="bilinear")
    ax2.set_title(title); ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def run_legacy_run_dir_mode(model, cfg: ModelConfig, transform, device: str, args):
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output)
    cameras = args.cameras or CAMERA_NAMES
 
    for cam in cameras:
        cam_dir = run_dir / cam
        if not cam_dir.is_dir():
            print(f"  [SKIP] {cam}: directory not found")
            continue
        print(f"\n[{cam}]")
        files = resolve_camera_files(run_dir, cam, args.frame_indices)
        if not files:
            print(f"  [SKIP] {cam}: no frames found")
            continue
        feats = extract_features(model, cfg, files, transform, device, args.batch_size)
 
        if args.save_features:
            np.save(output_dir / f"{cam}_cls.npy", feats["cls"])
            np.save(output_dir / f"{cam}_patch.npy", feats["patch"])
 
        if args.generate_self_heatmaps:
            cam_out = output_dir / "self_heatmaps" / cam
            cam_out.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(files), args.sample_interval):
                out_path = cam_out / f"{cam}_frame_{i:04d}_{args.heatmap_mode}.png"
                visualize_self_heatmap(files[i], feats["patch"][i], cfg, out_path, args.heatmap_mode)
                print(f"  Generated {args.heatmap_mode} heatmap for frame {i}")
 
 
#  Main
files_lookup: Dict[str, Path] = {}  # filename -> Path, populated in main()
 
 
def main():
    parser = argparse.ArgumentParser(description="Unified DINOv2/v3 sim-vs-real feature comparison")
    parser.add_argument("--input_dir", default=None,
                         help="Folder of real_*/sim_* images to compare (generalized pairs mode)")
    parser.add_argument("--run_dir", default=None,
                         help="Legacy: path to a single run's cam_*/ frame directories "
                              "(no real/sim comparison, just self-similarity heatmaps)")
    parser.add_argument("--model_version", choices=["v2", "v3"], required=True,
                         help="Which DINO version to use")
    parser.add_argument("--analysis", nargs="+", choices=["cls", "patch", "worst", "all"],
                         default=["all"], help="Which analysis mode(s) to run (input_dir mode only)")
    parser.add_argument("--top_k_worst", type=int, default=15,
                         help="Number of worst-matching patches to report/plot per pair")
    parser.add_argument("--output", default="dino_features/", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_features", action="store_true", help="Save raw .npy feature arrays")
 
    # legacy run_dir mode options
    parser.add_argument("--cameras", nargs="+", default=None, help="[run_dir mode] restrict to these cameras")
    parser.add_argument("--frame_indices", type=int, nargs="+", default=None,
                         help="[run_dir mode] only extract these frame positions per camera")
    parser.add_argument("--generate_self_heatmaps", action="store_true",
                         help="[run_dir mode] generate per-frame PCA/norm/cluster heatmaps")
    parser.add_argument("--heatmap_mode", default="pca", choices=["pca", "norm", "cluster"])
    parser.add_argument("--sample_interval", type=int, default=1)
 
    args = parser.parse_args()
 
    if not args.input_dir and not args.run_dir:
        parser.error("must specify either --input_dir (real_/sim_ pairs) or --run_dir (legacy)")
 
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {args.device}")
    print(f"Output : {output_dir}\n")
 
    model, cfg = load_model(args.model_version, args.device)
    transform = make_transform(cfg)
 
    if args.input_dir:
        input_dir = Path(args.input_dir)
        files, meta = discover_images(input_dir)
        global files_lookup
        files_lookup = {f.name: f for f in files}
 
        feats = extract_features(model, cfg, files, transform, args.device, args.batch_size)
        file_to_idx = {name: i for i, name in enumerate(feats["files"])}
 
        if args.save_features:
            np.save(output_dir / "cls_features.npy", feats["cls"])
            np.save(output_dir / "patch_features.npy", feats["patch"])
 
        within_pairs, anchor_pairs, groups = build_pairs(meta)
 
        modes = set(args.analysis)
        if "all" in modes:
            modes = {"cls", "patch", "worst"}
 
        summary = {"model_version": args.model_version, "n_images": len(files)}
        if "cls" in modes:
            summary["cls"] = run_cls_analysis(feats, file_to_idx, within_pairs, anchor_pairs,
                                               groups, output_dir)
        if "patch" in modes:
            summary["patch"] = run_patch_analysis(feats, file_to_idx, within_pairs, anchor_pairs,
                                                    cfg, output_dir)
        if "worst" in modes:
            summary["worst"] = run_worst_patch_analysis(feats, file_to_idx, within_pairs, anchor_pairs,
                                                          cfg, args.top_k_worst, output_dir)
 
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary -> {output_dir / 'summary.json'}")
        return
 
    # legacy mode
    run_legacy_run_dir_mode(model, cfg, transform, args.device, args)
 
 
if __name__ == "__main__":
    main()
    
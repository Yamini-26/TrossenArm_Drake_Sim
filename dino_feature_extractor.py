#!/usr/bin/env python
"""DINO Feature Extractor for sim-to-sim comparison."""
import os
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


#  Config

CAMERA_NAMES = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

# DINOv2 ViT-B/14:  input 518×518, patch 14px → 37×37 = 1369 patches, dim 768
# We resize to 518 (must be divisible by patch size 14: 518 = 37×14)
DINOV2_INPUT_SIZE = 518
DINOV2_PATCH_SIZE = 14
DINOV2_DIM        = 768
DINOV2_MODEL      = "dinov2_vitb14"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# Model loading

def load_dinov2(device: str = "cuda") -> torch.nn.Module:
    """Load DINOv2 ViT-B/14 from torch.hub (downloads once, then cached)."""
    print(f"Loading {DINOV2_MODEL} ...")
    model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL, trust_repo=True, force_reload=False)
    model.eval()
    model.to(device)
    print(f"  Loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


#  Image preprocessing

def make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(DINOV2_INPUT_SIZE,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(DINOV2_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


#  Feature extraction

@torch.no_grad()
def extract_features_from_dir(model: torch.nn.Module, frame_dir: Path, transform: transforms.Compose, device: str, batch_size: int = 8) -> Dict[str, np.ndarray]:
    """
    Extract DINOv2 patch features for all PNGs in frame_dir.
    Returns array of shape (N, 1369, 768)
    """
    png_files = sorted(frame_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {frame_dir}")

    print(f"  Extracting from {len(png_files)} frames in {frame_dir.name}/")

    cls_list = []
    patch_list = []

    for i in range(0, len(png_files), batch_size):
        batch_files = png_files[i : i + batch_size]
        imgs = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            imgs.append(transform(img))
        batch = torch.stack(imgs).to(device)

        # forward_features returns a dict with 'x_norm_clstoken' and
        # 'x_norm_patchtokens' (DINOv2 API)
        out = model.forward_features(batch)
        cls_tokens   = out["x_norm_clstoken"]           # (B, 768)
        patch_tokens = out["x_norm_patchtokens"]        # (B, N_patches, 768)
        
        cls_list.append(cls_tokens.cpu().float().numpy())
        patch_list.append(patch_tokens.cpu().float().numpy())

        if (i // batch_size) % 5 == 0:
            print(f"    batch {i // batch_size + 1} / {len(png_files) // batch_size + 1}")

    return {
        "cls":   np.concatenate(cls_list,   axis=0),   # (N, 768)
        "patch": np.concatenate(patch_list, axis=0),   # (N, 1369, 768)
        "files": [f.name for f in png_files],
    }


def extract_all_cameras(model, run_dir: Path, transform, device) -> Dict[str, Dict]:
    """Extract features for every camera found in run_dir."""
    results = {}
    for cam in CAMERA_NAMES:
        cam_dir = run_dir / cam
        if not cam_dir.is_dir():
            print(f"  [SKIP] {cam}: directory not found")
            continue
        print(f"\n[{cam}]")
        results[cam] = extract_features_from_dir(model, cam_dir, transform, device)
    return results


# Heatmap generation

def patch_cosine_similarity(
    patch_a: np.ndarray, patch_b: np.ndarray, frame_idx: int = 0
) -> np.ndarray:
    """
    Spatial patch-level cosine similarity for a single frame.
    patch_a/b shape: (N, 1369, 768)
    Returns (37, 37) heatmap — high = similar, low = different.
    """
    a = patch_a[frame_idx]   # (1369, 768)
    b = patch_b[frame_idx]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    sim = (a * b).sum(axis=1)           # (1369,)
    H   = W = int(DINOV2_INPUT_SIZE / DINOV2_PATCH_SIZE)
    return sim.reshape(H, W)            # (37, 37)

def plot_patch_heatmaps(
    patch_a: np.ndarray, patch_b: np.ndarray,
    frame_indices: List[int],
    cam_name: str,
    output_path: Path,
):
    """Spatial patch cosine similarity heatmaps for selected frames."""
    n = len(frame_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, fidx in zip(axes, frame_indices):
        hmap = patch_cosine_similarity(patch_a, patch_b, fidx)
        im   = ax.imshow(hmap, vmin=0, vmax=1, cmap="RdYlGn", origin="upper")
        ax.set_title(f"frame {fidx}")
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.03, label="cos sim")
    fig.suptitle(f"{cam_name} — patch similarity (run A vs B)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def visualize_patch_embeddings_heatmap(
    image_path: Path,
    patch_features: np.ndarray,  # (1369, 768) for one frame
    output_path: Path,
    mode: str = "pca"  # "pca", "norm", "cluster"
):
    """Create and save a heatmap visualization of DINOv2 patch embeddings."""
    # Load original image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.Resampling.LANCZOS)  # (518, 518)
    
    # Reshape patches to 37x37 grid
    H = W = DINOV2_INPUT_SIZE // DINOV2_PATCH_SIZE  # 37
    patches = patch_features.reshape(H, W, -1)  # (37, 37, 768)
    
    if mode == "pca":
        # PCA on patch embeddings
        pca = PCA(n_components=1)
        patches_flat = patch_features  # (1369, 768)
        pc1 = pca.fit_transform(patches_flat).flatten()  # (1369,)
        # Normalize to [0,1]
        pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
        heatmap = pc1.reshape(H, W)
        title = "DINOv2 PCA - First Component"
        cmap = "viridis"
        
    elif mode == "norm":
        # L2 norm of each patch (activation strength)
        norms = np.linalg.norm(patch_features, axis=1)  # (1369,)
        heatmap = norms.reshape(H, W)
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        title = "DINOv2 Patch Activation Strength"
        cmap = "hot"
        
    elif mode == "cluster":
        # Semantic segmentation via clustering
        from sklearn.cluster import KMeans
        n_clusters = 8
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(patch_features)  # (1369,)
        heatmap = clusters.reshape(H, W)
        title = f"DINOv2 Semantic Clusters ({n_clusters} regions)"
        cmap = "tab10"
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    # Heatmap overlay
    im = ax2.imshow(img)
    # Overlay heatmap with transparency
    heatmap_plot = ax2.imshow(heatmap, cmap=cmap, alpha=0.6, interpolation='bilinear')
    ax2.set_title(title)
    ax2.axis("off")
    
    plt.colorbar(im, ax=ax2, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return heatmap


def visualize_all_frames_heatmaps(frame_dir: Path, patch_features: np.ndarray, cam_name: str, output_dir: Path,mode: str = "pca", sample_interval: int = 5):
    """Generate heatmaps for every camera, sampling every few frames."""
    frame_files = sorted(frame_dir.glob("*.png"))
    
    for idx in range(0, len(frame_files), sample_interval):
        if idx >= len(patch_features):
            break
            
        output_path = output_dir / f"{cam_name}_frame_{idx:04d}_{mode}.png"
        visualize_patch_embeddings_heatmap(
            frame_files[idx],
            patch_features[idx],
            output_path,
            mode=mode
        )
        print(f"  Generated heatmap for frame {idx}")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 sim-to-sim feature comparison")
    parser.add_argument("--run_a", required=True, help="Path to simulation_frames from run A")
    parser.add_argument("--output", default="dino_output", help="Output directory for features and plots")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_features", action="store_true", help="Save raw .npy feature arrays to disk")
    parser.add_argument("--sample_interval", type=int, default=5, help="Sample every N frames for heatmap generation")
    parser.add_argument("--generate_heatmaps", action="store_true", help="Generate embedding heatmaps for visualization")
    parser.add_argument("--heatmap_mode", default="pca", choices=["pca", "norm", "cluster"], help="Self features heatmap to generate")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {args.device}")
    print(f"Run A  : {args.run_a}")
    print(f"Output : {output_dir}\n")
    # print(f"Heatmap mode : {args.heatmap_mode}")
    # print(f"Sample every : {args.sample_interval} frames\n")

    model     = load_dinov2(args.device)
    transform = make_transform()

    # ── Extract features for run A ──
    print("\\n Extracting Run A features")
    features_a = extract_all_cameras(model, Path(args.run_a), transform, args.device)

    if args.save_features:
        for cam, data in features_a.items():
            np.save(output_dir / f"runA_{cam}_cls.npy",   data["cls"])
            np.save(output_dir / f"runA_{cam}_patch.npy", data["patch"])
        print("  Run A features saved.")

    # Generate heatmaps for single run
    if args.generate_heatmaps:
        print("\n── Generating embedding heatmaps ──")
        heatmap_dir = output_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        
        for cam, data in features_a.items():
            print(f"\n[{cam}] Generating {args.heatmap_mode} heatmaps...")
            cam_dir = Path(args.run_a) / cam
            
            # Create per-camera subdirectory
            cam_heatmap_dir = heatmap_dir / cam
            cam_heatmap_dir.mkdir(exist_ok=True)
            
            # Generate for all frames (or sample)
            visualize_all_frames_heatmaps(
                cam_dir,
                data["patch"],
                cam,
                cam_heatmap_dir,
                mode=args.heatmap_mode,
                sample_interval=args.sample_interval  # Every Nth frame
            )


if __name__ == "__main__":
    main()
    
"""DINOv3 patch features."""

import torch
from transformers import AutoImageProcessor, AutoModel

# Swap models here and rerun -- everything below derives from whatever loads.
# The facebook/* DINOv3 repos are gated behind a licence click-through, so
# this points at an open mirror of the same weights.
MODEL_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# DINOv3 prepends a CLS token and some register tokens to the patch tokens
NUM_PREFIX_TOKENS = 1 + model.config.num_register_tokens

# DINOv3 uses RoPE rather than learned position embeddings, so it accepts any
# input size that divides into whole patches. The processor would otherwise
# squash everything to a distorted 224x224; this keeps the cameras' own 4:3
# shape (matches CAM_WIDTH/CAM_HEIGHT in lerobot_simulation.py).
INPUT_HEIGHT, INPUT_WIDTH = 480, 640
PATCH = model.config.patch_size
GRID = (INPUT_HEIGHT // PATCH, INPUT_WIDTH // PATCH)


def patch_features(image):
    """Run DINOv3 on an image, returning its (num_patches, channels) tokens."""
    inputs = processor(
        images=image,
        size={"height": INPUT_HEIGHT, "width": INPUT_WIDTH},
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[0, NUM_PREFIX_TOKENS:, :]


def patch_at(x, y):
    """The flat index of the patch covering a pixel of a displayed frame."""
    row = min(int(y / INPUT_HEIGHT * GRID[0]), GRID[0] - 1)
    column = min(int(x / INPUT_WIDTH * GRID[1]), GRID[1] - 1)
    return row * GRID[1] + column


def on_grid(values):
    """Lay a per-patch scalar out on the patch grid, for imshow."""
    return values.float().reshape(*GRID).cpu().numpy()

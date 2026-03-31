"""
Load DenoisingAutoencoder from project-root dae_best.pt (notebook format).
"""

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch

from src.dae_model import DenoisingAutoencoder


def load_dae_from_checkpoint(
    ckpt_path: Optional[Path] = None,
    map_location: Optional[torch.device] = None,
) -> Tuple[DenoisingAutoencoder, dict[str, Any]]:
    """
    Load weights from dict with 'model_state' / 'state_dict' and optional 'cfg'.
    Falls back to base_ch=48, latent_dim=128 if cfg missing (common trained setup).
    """
    if ckpt_path is None:
        ckpt_path = Path(__file__).resolve().parent.parent / "dae_best.pt"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blob = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(blob, dict) and "model_state" in blob:
        state = blob["model_state"]
        cfg_saved = blob.get("cfg") or {}
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
        cfg_saved = blob.get("cfg") or {}
    else:
        state = blob
        cfg_saved = {}

    base_ch = int(cfg_saved.get("base_ch", 48))
    latent_dim = int(cfg_saved.get("latent_dim", 128))
    model = DenoisingAutoencoder(base_ch=base_ch, latent_dim=latent_dim).to(map_location)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, blob if isinstance(blob, dict) else {}


@torch.no_grad()
def denoise_gray_28(
    model: DenoisingAutoencoder,
    gray_uint8_28: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """gray_uint8_28: (28, 28) uint8 → same shape uint8."""
    if gray_uint8_28.shape != (28, 28):
        raise ValueError(f"Expected (28, 28), got {gray_uint8_28.shape}")
    if device is None:
        device = next(model.parameters()).device
    x = torch.from_numpy(gray_uint8_28.astype(np.float32) / 255.0).view(1, 1, 28, 28).to(device)
    y = model(x)
    out = (y.squeeze(0).squeeze(0).float().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    return out

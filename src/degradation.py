"""
Degradations aligned with main.ipynb (float tensor (1, H, W) in [0, 1]).

Also provides numpy uint8 bridges for Streamlit (28×28 grayscale).
"""

from __future__ import annotations

import io
import random
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# Same ranges as notebook
DEGRADE_CFG = dict(
    gaussian_std=(0.08, 0.15),
    sp_amount=(0.03, 0.08),
    blur_kernel=(3, 4),
    jpeg_quality=(20, 50),
)

NOTEBOOK_DEGRADATION_MODES: tuple[str, ...] = ("gaussian", "salt_pepper", "blur", "jpeg")


def _sample_degrade_param(param_name: str):
    lo, hi = DEGRADE_CFG[param_name]
    if param_name == "blur_kernel":
        kernel_choices = [k for k in range(lo, hi + 1) if k % 2 == 1]
        return random.choice(kernel_choices)
    if param_name == "jpeg_quality":
        return random.randint(lo, hi)
    return random.uniform(lo, hi)


def add_gaussian_noise(img: torch.Tensor, std: float | None = None) -> torch.Tensor:
    if std is None:
        std = _sample_degrade_param("gaussian_std")
    noise = torch.randn_like(img) * std
    return (img + noise).clamp(0.0, 1.0)


def add_salt_pepper(img: torch.Tensor, amount: float | None = None) -> torch.Tensor:
    if amount is None:
        amount = _sample_degrade_param("sp_amount")
    out = img.clone()
    mask = torch.rand_like(img)
    out[mask < amount / 2] = 0.0
    out[(mask >= amount / 2) & (mask < amount)] = 1.0
    return out


def add_blur(img: torch.Tensor, kernel_size: int | None = None) -> torch.Tensor:
    if kernel_size is None:
        kernel_size = _sample_degrade_param("blur_kernel")
    blurrer = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1.0, 2.0))
    return blurrer(img)


def add_jpeg_artifacts(img: torch.Tensor, quality: int | None = None) -> torch.Tensor:
    if quality is None:
        quality = _sample_degrade_param("jpeg_quality")
    pil = transforms.ToPILImage(mode="L")(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    pil_dec = Image.open(buf).convert("L")
    return transforms.ToTensor()(pil_dec)


DEGRADATIONS = {
    "gaussian": add_gaussian_noise,
    "salt_pepper": add_salt_pepper,
    "blur": add_blur,
    "jpeg": add_jpeg_artifacts,
}

DEGRADE_PARAMS = {
    "gaussian": {"std": "gaussian_std"},
    "salt_pepper": {"amount": "sp_amount"},
    "blur": {"kernel_size": "blur_kernel"},
    "jpeg": {"quality": "jpeg_quality"},
}


def degrade(img: torch.Tensor, mode: str = "random") -> torch.Tensor:
    """Apply one degradation; (1, H, W) float [0,1]. mode='random' picks one of four."""
    if mode == "random":
        mode = random.choice(list(DEGRADATIONS.keys()))
    if mode not in DEGRADATIONS:
        raise ValueError(f"Unknown degradation mode: {mode}")
    fn = DEGRADATIONS[mode]
    params = {k: _sample_degrade_param(v) for k, v in DEGRADE_PARAMS[mode].items()}
    return fn(img, **params)


def apply_degradation_pipeline_tensor(
    img_1hw: torch.Tensor,
    modes: Sequence[str],
) -> torch.Tensor:
    """
    Apply degradations in order. Empty modes → unchanged.
    img_1hw: (1, H, W) float32 [0, 1].
    """
    if img_1hw.dim() != 3 or img_1hw.shape[0] != 1:
        raise ValueError(f"Expected shape (1, H, W), got {tuple(img_1hw.shape)}")
    out = img_1hw.clone()
    for mode in modes:
        if mode not in DEGRADATIONS:
            raise ValueError(f"Unknown mode: {mode}. Use one of {list(DEGRADATIONS.keys())}")
        out = degrade(out, mode)
    return out


def gray_uint8_to_tensor_1hw(arr: np.ndarray) -> torch.Tensor:
    """(H, W) uint8 → (1, H, W) float [0,1]."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale, got shape {arr.shape}")
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def tensor_1hw_to_gray_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, H, W) float → (H, W) uint8."""
    return (t.squeeze(0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)


def apply_degradation_pipeline_uint8(
    gray_uint8: np.ndarray,
    modes: Iterable[str],
) -> np.ndarray:
    """Grayscale uint8 (H, W) — same shape after pipeline."""
    modes_list: List[str] = list(modes)
    t = gray_uint8_to_tensor_1hw(gray_uint8)
    t = apply_degradation_pipeline_tensor(t, modes_list)
    return tensor_1hw_to_gray_uint8(t)


def apply_degradation(img: np.ndarray) -> np.ndarray:
    """
    Legacy helper: random 2–4 notebook degradations on uint8 BGR or grayscale.
    For new code use apply_degradation_pipeline_uint8 with explicit modes.
    """
    if img.ndim == 2:
        gray = img
    else:
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t = gray_uint8_to_tensor_1hw(gray)
    k = random.randint(2, 4)
    chosen = random.sample(list(DEGRADATIONS.keys()), k=k)
    t = apply_degradation_pipeline_tensor(t, chosen)
    out = tensor_1hw_to_gray_uint8(t)
    if img.ndim == 2:
        return out
    import cv2

    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

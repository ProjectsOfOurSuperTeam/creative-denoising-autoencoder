"""
Synthetic degradation pipeline for denoising model training.

Degradations: Gaussian noise, Poisson noise, impulse noise (salt & pepper),
Gaussian blur, JPEG artifacts, downscale → upscale.
"""

import random
from typing import Optional

import cv2
import numpy as np


def _add_poisson_noise(img: np.ndarray, lam: Optional[float] = None) -> np.ndarray:
    """
    Add Poisson noise (photon shot noise).
    lam: mean value for Poisson; if None, scaled from intensity.
    """
    out = img.astype(np.float64)
    if lam is None:
        lam = random.uniform(10, 80)
    # Poisson: variance = mean; scale for pixels [0,255]
    noisy = np.random.poisson(np.maximum(out * lam / 255.0, 1e-6))
    out = noisy * 255.0 / lam
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_impulse_noise(img: np.ndarray, amount: float, salt_vs_pepper: float = 0.5) -> np.ndarray:
    """
    Impulse noise (salt & pepper): random pixels become 0 or 255.
    amount: fraction of pixels to replace (0..1)
    salt_vs_pepper: fraction of salt (255) among replaced (0.5 = 50/50)
    """
    out = img.copy()
    h, w = out.shape[:2]
    n_pixels = h * w
    n_salt = int(n_pixels * amount * salt_vs_pepper)
    n_pepper = int(n_pixels * amount * (1 - salt_vs_pepper))

    # Salt (white pixels) — per pixel (y,x), all channels
    ys = np.random.randint(0, h, n_salt)
    xs = np.random.randint(0, w, n_salt)
    out[ys, xs] = 255

    # Pepper (black pixels)
    yp = np.random.randint(0, h, n_pepper)
    xp = np.random.randint(0, w, n_pepper)
    out[yp, xp] = 0

    return out


def apply_degradation(img: np.ndarray) -> np.ndarray:
    """
    Apply a random set of degradations to RGB or grayscale image (uint8).

    Degradations (each applied independently with given probability):
      • Gaussian noise (80%)
      • Poisson noise (40%)
      • Impulse noise / salt & pepper (30%)
      • Gaussian blur (50%)
      • JPEG artifacts (50%)
      • Downscale → Upscale (30%)

    Parameters scale with image size so the effect is visible on both small and large images.
    """
    out = img.copy()
    h, w = out.shape[:2]
    n_pixels = h * w
    # Reference: 256×256; scale so larger images get stronger degradation
    scale = (n_pixels / (256 * 256)) ** 0.25  # gentle scaling

    # Gaussian noise — scale sigma so noise is visible on large images
    if random.random() < 0.8:
        sigma = random.uniform(5, 50) * scale
        sigma = min(sigma, 80)
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Poisson noise
    if random.random() < 0.4:
        out = _add_poisson_noise(out)

    # Impulse noise (salt & pepper)
    if random.random() < 0.3:
        amount = random.uniform(0.01, 0.05)
        out = _add_impulse_noise(out, amount=amount)

    # Gaussian blur — scale kernel with image size (on large images, 7px blur is negligible)
    if random.random() < 0.5:
        base = max(3, min(31, 3 + 2 * (min(h, w) // 256)))
        ksize = base if base % 2 == 1 else base + 1
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    # JPEG compression
    if random.random() < 0.5:
        quality = random.randint(10, 60)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", out, encode_param)
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if out is None:
            out = img.copy()

    # Downscale → Upscale — scale factor with image size
    if random.random() < 0.3:
        min_side = min(h, w)
        factor = min(8, max(2, min_side // 256))
        small = cv2.resize(out, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    return out

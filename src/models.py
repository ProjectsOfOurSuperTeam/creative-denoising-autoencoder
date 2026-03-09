"""
Denoising models: CDAE, DnCNN, U-Net.
Interface for loading checkpoints and inference.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CDAE(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 2, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base, base)
        self.out = nn.Conv2d(base, in_ch, 1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bottleneck(self.pool2(x2))
        x = self.dec2(self.up2(xb))
        x = self.dec1(self.up1(x))
        return inp + self.out(x)


class DnCNN(nn.Module):
    def __init__(self, in_ch: int = 1, depth: int = 8, width: int = 64):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(width, width, 3, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(width, in_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred_noise = self.net(x)
        return x - pred_noise


class UNetSmall(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bridge = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, in_ch, 1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        xb = self.bridge(self.pool2(x2))

        x = self.up2(xb)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        return inp + self.out(x)


def build_model(model_name: str, in_channels: int = 1, **kwargs) -> nn.Module:
    key = model_name.lower()
    if key == "cdae":
        return CDAE(in_ch=in_channels, base=kwargs.get("base", 32))
    if key == "dncnn":
        return DnCNN(
            in_ch=in_channels,
            depth=kwargs.get("depth", 8),
            width=kwargs.get("width", 64),
        )
    if key == "unet":
        return UNetSmall(in_ch=in_channels, base=kwargs.get("base", 32))
    raise ValueError(f"Unknown model: {model_name}")


def load_best_model(
    model_name: str,
    in_channels: int = 1,
    checkpoint_root: Optional[Path] = None,
    map_location: Optional[torch.device] = None,
    **model_kwargs,
) -> Tuple[nn.Module, Path]:
    """
    Load the best model checkpoint.
    Uses weights_only=False for compatibility with checkpoints containing numpy objects.
    """
    if checkpoint_root is None:
        checkpoint_root = Path(__file__).resolve().parent.parent / "checkpoints"
    path = Path(checkpoint_root) / model_name.lower() / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    bp = payload.get("best_params", {})
    key = model_name.lower()
    if key == "cdae":
        model_kwargs = {**model_kwargs, "base": bp.get("base", 32)}
    elif key == "dncnn":
        model_kwargs = {**model_kwargs, "depth": bp.get("depth", 8), "width": bp.get("width", 64)}
    elif key == "unet":
        model_kwargs = {**model_kwargs, "base": bp.get("base", 32)}
    model = build_model(model_name, in_channels=in_channels, **model_kwargs)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, path


@torch.no_grad()
def denoise_image(
    model: nn.Module,
    image: "torch.Tensor",
    patch_size: int = 256,
    device: Optional[torch.device] = None,
) -> "torch.Tensor":
    """
    Inference on a single image.
    image: (C, H, W) or (H, W, C) float [0,1] or uint8 [0,255]
    Returns (C, H, W) float [0,1].
    """
    if device is None:
        device = next(model.parameters()).device

    if image.dim() == 3 and image.shape[-1] in (1, 3):
        image = image.permute(2, 0, 1)
    if image.dtype == torch.uint8 or image.max() > 1.0:
        image = image.float() / 255.0

    c, h, w = image.shape
    image = image.unsqueeze(0).to(device)

    if h <= patch_size and w <= patch_size:
        pred = model(image)
        return pred.squeeze(0).clamp(0, 1).cpu()
    # Sliding window for large images
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, ph, pw = image.shape

    output = torch.zeros_like(image)
    count = torch.zeros_like(image)
    for i in range(0, ph, patch_size):
        for j in range(0, pw, patch_size):
            patch = image[:, :, i : i + patch_size, j : j + patch_size]
            out_patch = model(patch)
            output[:, :, i : i + patch_size, j : j + patch_size] += out_patch
            count[:, :, i : i + patch_size, j : j + patch_size] += 1
    output = output / count
    output = output[:, :, :h, :w].squeeze(0).clamp(0, 1).cpu()
    return output

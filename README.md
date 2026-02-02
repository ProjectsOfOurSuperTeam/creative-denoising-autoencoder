# Creative CV — Denoising Autoencoder

Restoration of historical photographs using denoising autoencoders (CDAE, U-Net, DnCNN).  
The university project for "Deep Learning for Computer Vision" course.

## Structure

```
creative-denoising-autoencoder/
├── data/
│   ├── raw/           # Original datasets (DIV2K, Flickr2K)
│   ├── processed/     # Processed data
│   └── synthetic/     # Generated pairs for verification
├── src/
│   ├── dataset.py     # PyTorch Dataset and degradations
│   ├── models.py      # CDAE, U-Net, DnCNN
│   ├── losses.py      # L1, Perceptual, SSIM
│   ├── train.py       # Training loop
│   ├── evaluate.py    # PSNR, SSIM, LPIPS
│   └── utils.py       # Helpers
├── weights/           # Saved checkpoints (.pth)
├── results/           # Before/After images, plots
├── configs/
│   └── config.yaml    # Hyperparameters and paths
├── requirements.txt
├── main.py            # Entry point
└── README.md
```

## Setup

1. Create a virtual environment and install PyTorch (see [pytorch.org](https://pytorch.org)) for your CUDA version.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download DIV2K (or add your image paths) and set `data.train_images` / `data.val_images` in `configs/config.yaml`.

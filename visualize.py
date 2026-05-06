"""Lightweight visualization: run model on a few samples and save images.

Saves per-sample comparison (`guide`, `prediction`, `target`, `error`) and
per-band grayscale images to the `outputs/visuals` folder.

Designed to run quickly on a small indexed subset (default caps set small).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import MODEL, TARGET_BANDS_20M, TRAINING
from data_loader import WaldProtocolDataset
from model import EnhancedDSen2


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    mn = float(np.min(image))
    mx = float(np.max(image))
    if mx <= mn:
        return np.zeros_like(image)
    return (image - mn) / (mx - mn)


def _rgb_from_tensor(tensor: torch.Tensor, channels: Tuple[int, int, int]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    image = array[list(channels), :, :]
    image = np.transpose(image, (1, 2, 0))
    return _normalize_for_display(image)


def save_comparison(sample, pred, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    guide_rgb = _rgb_from_tensor(sample["guide_20m"], (2, 1, 0))
    target_rgb = _rgb_from_tensor(sample["target"], (0, 1, 2))
    pred_rgb = _rgb_from_tensor(pred, (0, 1, 2))
    error_map = np.mean(np.abs(sample["target"].detach().cpu().numpy() - pred.detach().cpu().numpy()), axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(guide_rgb)
    axes[0].set_title("Guide (20m)")
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Prediction")
    axes[2].imshow(target_rgb)
    axes[2].set_title("Target")
    axes[3].imshow(_normalize_for_display(error_map), cmap="magma")
    axes[3].set_title("Mean Abs Error")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_band_grays(sample, pred, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = sample["target"].detach().cpu().numpy()
    pred = pred.squeeze(0).detach().cpu().numpy()
    blurry = sample["blurry_20m"].detach().cpu().numpy()
    for i, band in enumerate(TARGET_BANDS_20M):
        t = _normalize_for_display(target[i])
        p = _normalize_for_display(pred[i])
        b = _normalize_for_display(blurry[i])
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(b, cmap="gray"); axes[0].set_title(f"Blurry {band}"); axes[0].axis('off')
        axes[1].imshow(p, cmap="gray"); axes[1].set_title(f"Pred {band}"); axes[1].axis('off')
        axes[2].imshow(t, cmap="gray"); axes[2].set_title(f"Target {band}"); axes[2].axis('off')
        fig.tight_layout()
        fig.savefig(out_dir / f"band_{band}.png", dpi=160)
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--samples", type=int, default=4, help="Number of samples to visualize")
    p.add_argument("--max-items", type=int, default=2)
    p.add_argument("--max-patches-per-item", type=int, default=8)
    p.add_argument("--max-total-patches", type=int, default=16)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/visuals"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WaldProtocolDataset(
        max_patches_per_item=args.max_patches_per_item,
        max_items_to_use=args.max_items,
        max_total_patches=args.max_total_patches,
        verbose=True,
    )

    model = EnhancedDSen2(
        input_channels=MODEL.input_channels,
        output_channels=MODEL.output_channels,
        base_channels=MODEL.base_channels,
        num_residual_blocks=MODEL.num_residual_blocks,
        se_reduction=MODEL.se_reduction,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()

    n = min(args.samples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        inp = sample["input"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp).cpu()

        comp_path = args.output_dir / f"sample_{i}_comparison.png"
        save_comparison(sample, pred, comp_path)
        save_band_grays(sample, pred, args.output_dir / f"sample_{i}_bands")
        print(f"Wrote visuals for sample {i} -> {comp_path}")


if __name__ == '__main__':
    main()

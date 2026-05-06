"""Inference and visualization utilities for the enhanced DSen2 model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import MODEL, TRAINING
from data_loader import WaldProtocolDataset, build_dataloader
from model import EnhancedDSen2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the enhanced DSen2 model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--num-workers", type=int, default=TRAINING.num_workers)
    parser.add_argument("--output-dir", type=Path, default=TRAINING.output_dir)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def _load_checkpoint(model: EnhancedDSen2, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)


@torch.no_grad()
def compute_rmse(model: EnhancedDSen2, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    for batch in dataloader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        predictions = model(inputs)
        squared_error += torch.sum((predictions - targets) ** 2).item()
        pixel_count += targets.numel()
    return math.sqrt(squared_error / max(1, pixel_count))


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros_like(image)
    return (image - minimum) / (maximum - minimum)


def _rgb_from_tensor(tensor: torch.Tensor, channels: Tuple[int, int, int]) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    image = array[list(channels), :, :]
    image = np.transpose(image, (1, 2, 0))
    return _normalize_for_display(image)


def plot_comparison(sample: Dict[str, torch.Tensor], prediction: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    guide_20m = sample["guide_20m"]
    target = sample["target"]
    pred = prediction.squeeze(0)

    guide_rgb = _rgb_from_tensor(guide_20m, (2, 1, 0))
    target_rgb = _rgb_from_tensor(target, (0, 1, 2))
    pred_rgb = _rgb_from_tensor(pred, (0, 1, 2))
    error_map = np.mean(np.abs(target.detach().cpu().numpy() - pred.detach().cpu().numpy()), axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(guide_rgb)
    axes[0].set_title("Guide (20 m)")
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Prediction")
    axes[2].imshow(target_rgb)
    axes[2].set_title("Ground Truth")
    axes[3].imshow(error_map, cmap="magma")
    axes[3].set_title("Mean Abs Error")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WaldProtocolDataset()
    dataloader = build_dataloader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = EnhancedDSen2(
        input_channels=MODEL.input_channels,
        output_channels=MODEL.output_channels,
        base_channels=MODEL.base_channels,
        num_residual_blocks=MODEL.num_residual_blocks,
        se_reduction=MODEL.se_reduction,
    ).to(device)
    _load_checkpoint(model, args.checkpoint, device)

    rmse = compute_rmse(model, dataloader, device)
    print(f"Final RMSE: {rmse:.6f}")

    sample = dataset[args.sample_index]
    prediction = model(sample["input"].unsqueeze(0).to(device)).cpu()
    plot_comparison(sample, prediction, args.output_dir / "comparison.png")


if __name__ == "__main__":
    main()
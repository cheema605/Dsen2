"""Inference and visualization utilities for the enhanced DSen2 model."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import MODEL, TARGET_BANDS_20M, TRAINING
from data_loader import WaldProtocolDataset, build_dataloader
from model import EnhancedDSen2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the enhanced DSen2 model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--num-workers", type=int, default=TRAINING.num_workers)
    parser.add_argument("--output-dir", type=Path, default=TRAINING.output_dir)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--band-report", action="store_true", help="Print and plot per-band improvement metrics")
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
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
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


def _band_metrics(sample: Dict[str, torch.Tensor], prediction: torch.Tensor) -> Dict[str, Dict[str, float]]:
    target = sample["target"].detach().cpu().numpy()
    blurry = sample["blurry_20m"].detach().cpu().numpy()
    pred = prediction.squeeze(0).detach().cpu().numpy()

    metrics: Dict[str, Dict[str, float]] = {}
    for band_index, band_name in enumerate(TARGET_BANDS_20M):
        target_band = target[band_index]
        blurry_band = blurry[band_index]
        pred_band = pred[band_index]

        blurry_rmse = float(np.sqrt(np.mean((blurry_band - target_band) ** 2)))
        pred_rmse = float(np.sqrt(np.mean((pred_band - target_band) ** 2)))
        improvement = blurry_rmse - pred_rmse
        change_from_blurry = float(np.mean(np.abs(pred_band - blurry_band)))

        metrics[band_name] = {
            "blurry_rmse": blurry_rmse,
            "pred_rmse": pred_rmse,
            "improvement": improvement,
            "change_from_blurry": change_from_blurry,
        }
    return metrics


def plot_band_report(sample: Dict[str, torch.Tensor], prediction: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = _band_metrics(sample, prediction)
    band_names = list(TARGET_BANDS_20M)
    blurry_rmse = [metrics[name]["blurry_rmse"] for name in band_names]
    pred_rmse = [metrics[name]["pred_rmse"] for name in band_names]
    improvement = [metrics[name]["improvement"] for name in band_names]
    change_from_blurry = [metrics[name]["change_from_blurry"] for name in band_names]

    x = np.arange(len(band_names))
    width = 0.28

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].bar(x - width, blurry_rmse, width=width, label="Blurry input RMSE")
    axes[0].bar(x, pred_rmse, width=width, label="Prediction RMSE")
    axes[0].bar(x + width, change_from_blurry, width=width, label="|Pred - Blurry|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(band_names)
    axes[0].set_ylabel("Value")
    axes[0].set_title("Per-band error and adjustment")
    axes[0].legend(loc="upper right")

    colors = ["tab:green" if value >= 0 else "tab:red" for value in improvement]
    axes[1].bar(x, improvement, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(band_names)
    axes[1].set_ylabel("RMSE reduction")
    axes[1].set_title("Prediction improvement over blurry input by band")

    for axis in axes:
        axis.tick_params(axis="x", rotation=0)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_band_report(sample: Dict[str, torch.Tensor], prediction: torch.Tensor) -> None:
    metrics = _band_metrics(sample, prediction)
    print("Band-wise report (20m output bands):")
    print("  band   blurry_rmse   pred_rmse   improvement   |pred-blurry|")
    for band_name in TARGET_BANDS_20M:
        item = metrics[band_name]
        print(
            f"  {band_name:>4}   {item['blurry_rmse']:.6f}    {item['pred_rmse']:.6f}    "
            f"{item['improvement']:+.6f}      {item['change_from_blurry']:.6f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using device: cuda")

    dataset = WaldProtocolDataset()
    pin_memory = True if device.type == "cuda" else False
    dataloader = build_dataloader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=pin_memory
    )

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
    input_tensor = sample["input"].unsqueeze(0).to(device, non_blocking=True)
    prediction = model(input_tensor).cpu()
    plot_comparison(sample, prediction, args.output_dir / "comparison.png")
    if args.band_report:
        print_band_report(sample, prediction)
        plot_band_report(sample, prediction, args.output_dir / "band_report.png")


if __name__ == "__main__":
    main()
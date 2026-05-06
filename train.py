"""Training entry point for the enhanced DSen2 Sentinel-2 super-resolution model."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from config import MODEL, TRAINING
from data_loader import WaldProtocolDataset, build_dataloader, split_dataset
from model import EnhancedDSen2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["input"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    return inputs, targets


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    squared_error = 0.0
    pixel_count = 0
    for batch in dataloader:
        inputs, targets = _move_batch_to_device(batch, device)
        predictions = model(inputs)
        squared_error += torch.sum((predictions - targets) ** 2).item()
        pixel_count += targets.numel()
    return math.sqrt(squared_error / max(1, pixel_count))


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float,
) -> float:
    model.train()
    running_squared_error = 0.0
    pixel_count = 0
    for batch in dataloader:
        inputs, targets = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        running_squared_error += torch.sum((predictions.detach() - targets) ** 2).item()
        pixel_count += targets.numel()

    return math.sqrt(running_squared_error / max(1, pixel_count))


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_rmse: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_rmse": best_val_rmse,
        },
        checkpoint_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the enhanced DSen2 model")
    parser.add_argument("--epochs", type=int, default=TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--num-workers", type=int, default=TRAINING.num_workers)
    parser.add_argument("--learning-rate", type=float, default=TRAINING.learning_rate)
    parser.add_argument("--validation-split", type=float, default=TRAINING.validation_split)
    parser.add_argument("--checkpoint-dir", type=Path, default=TRAINING.checkpoint_dir)
    parser.add_argument("--seed", type=int, default=TRAINING.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WaldProtocolDataset()
    train_dataset, val_dataset = split_dataset(dataset, validation_split=args.validation_split, seed=args.seed)
    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = EnhancedDSen2(
        input_channels=MODEL.input_channels,
        output_channels=MODEL.output_channels,
        base_channels=MODEL.base_channels,
        num_residual_blocks=MODEL.num_residual_blocks,
        se_reduction=MODEL.se_reduction,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_rmse = float("inf")
    last_checkpoint = args.checkpoint_dir / "enhanced_dsen2_last.pt"
    best_checkpoint = args.checkpoint_dir / "enhanced_dsen2_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_rmse = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_norm=TRAINING.gradient_clip_norm,
        )
        val_rmse = evaluate(model, val_loader, device)

        save_checkpoint(last_checkpoint, model, optimizer, epoch, best_val_rmse)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            save_checkpoint(best_checkpoint, model, optimizer, epoch, best_val_rmse)

        print(f"Epoch {epoch:03d} | train RMSE: {train_rmse:.6f} | val RMSE: {val_rmse:.6f} | best: {best_val_rmse:.6f}")


if __name__ == "__main__":
    main()

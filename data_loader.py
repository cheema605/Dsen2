"""Sentinel-2 STAC extraction, degradation, and patch dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import GUIDE_BANDS_10M, PATCH, STAC, TARGET_BANDS_20M

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    rasterio = None
    Window = None
    _RASTERIO_IMPORT_ERROR = exc
else:
    _RASTERIO_IMPORT_ERROR = None

try:
    from pystac_client import Client
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    Client = None
    _PYSTAC_IMPORT_ERROR = exc
else:
    _PYSTAC_IMPORT_ERROR = None


BAND_ASSET_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "B02": ("B02", "blue"),
    "B03": ("B03", "green"),
    "B04": ("B04", "red"),
    "B08": ("B08", "nir", "nir08"),
    "B05": ("B05", "rededge1"),
    "B06": ("B06", "rededge2"),
    "B07": ("B07", "rededge3"),
    "B8A": ("B8A", "B08A", "nir08", "nir_narrow"),
    "B11": ("B11", "swir16"),
    "B12": ("B12", "swir22"),
}


@dataclass(frozen=True)
class PatchIndex:
    """Location of a patch in a specific STAC item."""

    item_index: int
    row_10m: int
    col_10m: int


def _require_rasterio() -> None:
    if rasterio is None:
        raise ImportError("rasterio is required for data loading") from _RASTERIO_IMPORT_ERROR


def _require_pystac() -> None:
    if Client is None:
        raise ImportError("pystac-client is required for STAC search") from _PYSTAC_IMPORT_ERROR


def search_stac_items(
    bbox: Tuple[float, float, float, float] = STAC.bbox,
    datetime_range: str = STAC.datetime_range,
    max_cloud_cover: int = STAC.max_cloud_cover,
    max_items: int = STAC.max_items,
    url: str = STAC.url,
    collection: str = STAC.collection,
) -> List[object]:
    """Query Earth Search for Sentinel-2 L2A items."""

    _require_pystac()
    client = Client.open(url)
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=max_items,
    )
    return list(search.items())


def _resolve_asset_href(item: object, band_name: str) -> str:
    assets = getattr(item, "assets")
    candidates = BAND_ASSET_CANDIDATES[band_name]
    for candidate in candidates:
        if candidate in assets:
            return assets[candidate].href
    available = ", ".join(sorted(assets.keys()))
    raise KeyError(f"Unable to resolve band {band_name}; available assets: {available}")


def _build_start_positions(image_length: int, patch_size: int, stride: int) -> List[int]:
    if image_length <= patch_size:
        return [0]
    starts = list(range(0, image_length - patch_size + 1, stride))
    final_start = image_length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    unique_starts = sorted(set(max(0, start - (start % 2)) for start in starts))
    return unique_starts


def _read_window_stack(
    asset_hrefs: Sequence[str],
    window: Window,
    normalize: bool = True,
) -> torch.Tensor:
    _require_rasterio()
    bands: List[np.ndarray] = []
    for href in asset_hrefs:
        with rasterio.open(href) as src:
            array = src.read(1, window=window, boundless=False).astype(np.float32)
        if normalize:
            array = array / 10000.0
        bands.append(array)
    stacked = np.stack(bands, axis=0)
    return torch.from_numpy(stacked)


def _downsample(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


def _upsample(tensor: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(tensor.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)


class WaldProtocolDataset(Dataset):
    """Patch-level dataset implementing Wald's protocol on Sentinel-2 L2A COGs."""

    def __init__(
        self,
        items: Optional[Sequence[object]] = None,
        bbox: Tuple[float, float, float, float] = STAC.bbox,
        datetime_range: str = STAC.datetime_range,
        max_cloud_cover: int = STAC.max_cloud_cover,
        max_items: int = STAC.max_items,
        patch_size_10m: int = PATCH.patch_size_10m,
        patch_size_20m: int = PATCH.patch_size_20m,
        overlap_10m: int = PATCH.overlap_10m,
        stac_url: str = STAC.url,
        collection: str = STAC.collection,
    ) -> None:
        self.patch_size_10m = patch_size_10m
        self.patch_size_20m = patch_size_20m
        self.overlap_10m = overlap_10m
        self.stride_10m = patch_size_10m - overlap_10m

        if items is None:
            items = search_stac_items(
                bbox=bbox,
                datetime_range=datetime_range,
                max_cloud_cover=max_cloud_cover,
                max_items=max_items,
                url=stac_url,
                collection=collection,
            )
        self.items = list(items)
        self.patch_index: List[PatchIndex] = self._build_patch_index()

    def _build_patch_index(self) -> List[PatchIndex]:
        _require_rasterio()
        patch_index: List[PatchIndex] = []
        for item_index, item in enumerate(self.items):
            guide_href = _resolve_asset_href(item, "B02")
            with rasterio.open(guide_href) as src:
                width = src.width
                height = src.height
            for row_10m in _build_start_positions(height, self.patch_size_10m, self.stride_10m):
                for col_10m in _build_start_positions(width, self.patch_size_10m, self.stride_10m):
                    patch_index.append(PatchIndex(item_index=item_index, row_10m=row_10m, col_10m=col_10m))
        return patch_index

    def __len__(self) -> int:
        return len(self.patch_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        _require_rasterio()
        patch = self.patch_index[index]
        item = self.items[patch.item_index]

        guide_hrefs = [_resolve_asset_href(item, band) for band in GUIDE_BANDS_10M]
        target_hrefs = [_resolve_asset_href(item, band) for band in TARGET_BANDS_20M]

        guide_window_10m = Window(patch.col_10m, patch.row_10m, self.patch_size_10m, self.patch_size_10m)
        guide_10m = _read_window_stack(guide_hrefs, guide_window_10m)

        row_20m = patch.row_10m // 2
        col_20m = patch.col_10m // 2
        target_window_20m = Window(col_20m, row_20m, self.patch_size_20m, self.patch_size_20m)

        guide_20m = _downsample(guide_10m, (self.patch_size_20m, self.patch_size_20m))
        target_20m = _read_window_stack(target_hrefs, target_window_20m)
        blurry_40m = _downsample(target_20m, (self.patch_size_20m // 2, self.patch_size_20m // 2))
        blurry_20m = _upsample(blurry_40m, (self.patch_size_20m, self.patch_size_20m))

        model_input = torch.cat([guide_20m, blurry_20m], dim=0)

        return {
            "input": model_input.float(),
            "target": target_20m.float(),
            "guide_10m": guide_10m.float(),
            "guide_20m": guide_20m.float(),
            "blurry_20m": blurry_20m.float(),
            "metadata": torch.tensor([patch.item_index, patch.row_10m, patch.col_10m], dtype=torch.int64),
        }


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def split_dataset(dataset: Dataset, validation_split: float, seed: int = 42) -> Tuple[Dataset, Dataset]:
    validation_size = max(1, int(len(dataset) * validation_split))
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [train_size, validation_size], generator=generator)

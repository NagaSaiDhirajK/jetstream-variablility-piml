"""Chunked ERA5 data access for phase-1 PIML training.

Uses xarray + dask-backed arrays so large multi-year datasets do not need to be
loaded fully into host memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

EPS = 1e-6


@dataclass
class Era5Metadata:
    lat: np.ndarray
    lon: np.ndarray
    times: np.ndarray
    levels_hpa: tuple[int, int, int]
    time_coord: str
    level_coord: str
    input_channel_names: list[str]
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None


@dataclass
class InputNormArtifacts:
    input_mean: np.ndarray
    input_std: np.ndarray
    levels_hpa: tuple[int, int, int]
    input_channel_names: list[str]
    train_sample_count: int


def _resolve_coord_names(ds: xr.Dataset) -> tuple[str, str]:
    time_coord = "time" if "time" in ds.coords else "valid_time"
    level_coord = "level" if "level" in ds.coords else "pressure_level"
    if time_coord not in ds.coords:
        raise ValueError("Could not find time coordinate. Expected 'time' or 'valid_time'.")
    if level_coord not in ds.coords:
        raise ValueError("Could not find level coordinate. Expected 'level' or 'pressure_level'.")
    return time_coord, level_coord


def _select_levels(available_levels: list[int], requested: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if requested is not None:
        if not all(v in available_levels for v in requested):
            raise ValueError(f"Requested levels {requested} not present. Available: {available_levels}")
        return requested

    preferred = (200, 250, 300)
    if all(v in available_levels for v in preferred):
        return preferred

    fallback = (200, 250, 350)
    if all(v in available_levels for v in fallback):
        warnings.warn(
            "300 hPa is unavailable; using (200, 250, 350) levels for phase-1 training.",
            stacklevel=2,
        )
        return fallback

    if len(available_levels) < 3:
        raise ValueError(f"Not enough pressure levels in dataset: {available_levels}")
    inferred = tuple(sorted(available_levels)[:3])
    warnings.warn(
        f"Using inferred levels {inferred}; verify this matches your intended cruise layer.",
        stacklevel=2,
    )
    return inferred


def _input_channel_names(levels_hpa: tuple[int, int, int]) -> list[str]:
    l_top, l_mid, l_bottom = levels_hpa
    return [
        f"t{l_top}",
        f"t{l_mid}",
        f"t{l_bottom}",
        f"z{l_mid}",
        f"u{l_bottom}",
        f"v{l_bottom}",
    ]


def _compute_input_stats(
    ds: xr.Dataset,
    time_coord: str,
    level_coord: str,
    levels_hpa: tuple[int, int, int],
    start: int,
    end: int,
    sample_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-split mean/std for input channels only."""
    l_top, l_mid, l_bottom = levels_hpa
    if sample_indices is not None:
        subset = ds.isel({time_coord: sample_indices})
    else:
        subset = ds.isel({time_coord: slice(start, end)})
    reduce_dims = [time_coord, "latitude", "longitude"]

    channels = [
        subset["t"].sel({level_coord: l_top}),
        subset["t"].sel({level_coord: l_mid}),
        subset["t"].sel({level_coord: l_bottom}),
        subset["z"].sel({level_coord: l_mid}),
        subset["u"].sel({level_coord: l_bottom}),
        subset["v"].sel({level_coord: l_bottom}),
    ]

    means: list[float] = []
    stds: list[float] = []
    for ch in channels:
        mean = float(ch.mean(dim=reduce_dims).compute())
        std = float(ch.std(dim=reduce_dims).compute())
        means.append(mean)
        stds.append(max(std, EPS))

    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def _validate_input_norm(
    input_norm: tuple[np.ndarray, np.ndarray] | None,
    expected_channels: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if input_norm is None:
        return None
    mean, std = input_norm
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(-1)
    if mean_arr.size != expected_channels or std_arr.size != expected_channels:
        raise ValueError(
            f"input_norm channel count mismatch: expected {expected_channels}, got mean={mean_arr.size}, std={std_arr.size}."
        )
    std_arr = np.maximum(std_arr, EPS)
    return mean_arr, std_arr


def load_input_norm_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cached input normalization arrays from an NPZ file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input norm file not found: {p}")
    with np.load(p) as data:
        if "input_mean" not in data or "input_std" not in data:
            raise ValueError(f"Invalid input norm file {p}: expected keys 'input_mean' and 'input_std'.")
        mean = np.asarray(data["input_mean"], dtype=np.float32)
        std = np.asarray(data["input_std"], dtype=np.float32)
    return mean, std


def save_input_norm_npz(
    path: str | Path,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    input_channel_names: list[str] | None = None,
    levels_hpa: tuple[int, int, int] | None = None,
) -> None:
    """Save normalization arrays (+ optional metadata) to NPZ."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "input_mean": np.asarray(input_mean, dtype=np.float32),
        "input_std": np.asarray(input_std, dtype=np.float32),
    }
    if input_channel_names is not None:
        payload["input_channel_names"] = np.asarray(input_channel_names, dtype=str)
    if levels_hpa is not None:
        payload["levels_hpa"] = np.asarray(levels_hpa, dtype=np.int32)
    np.savez_compressed(p, **payload)


def compute_input_norm_artifacts(
    file_pattern: str | Path,
    chunks: dict[str, int] | None = None,
    levels_hpa: tuple[int, int, int] | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    time_stride: int = 1,
    max_samples: int | None = None,
) -> InputNormArtifacts:
    """Compute train-split input normalization once for reuse across runs."""
    probe = Era5WindDataset(
        file_pattern=file_pattern,
        levels_hpa=levels_hpa,
        split="train",
        train_frac=train_frac,
        val_frac=val_frac,
        chunks=chunks,
        input_norm=None,
        time_stride=time_stride,
        max_samples=max_samples,
    )
    mean, std = _compute_input_stats(
        ds=probe.ds,
        time_coord=probe.time_coord,
        level_coord=probe.level_coord,
        levels_hpa=probe.levels_hpa,
        start=probe.start,
        end=probe.end,
        sample_indices=probe.time_indices,
    )
    return InputNormArtifacts(
        input_mean=mean,
        input_std=std,
        levels_hpa=probe.levels_hpa,
        input_channel_names=probe.input_channel_names,
        train_sample_count=len(probe),
    )


class Era5WindDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset over time-indexed ERA5 fields for upper-level surrogate training.

    Expected variables in source: t, z, u, v with dims [time, level, latitude, longitude].
    """

    def __init__(
        self,
        file_pattern: str | Path,
        levels_hpa: tuple[int, int, int] | None = None,
        split: str = "train",
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        chunks: dict[str, int] | None = None,
        input_norm: tuple[np.ndarray, np.ndarray] | None = None,
        time_stride: int = 1,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()

        if chunks is None:
            chunks = {"time": 32}
        if time_stride < 1:
            raise ValueError(f"time_stride must be >= 1, got {time_stride}.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive when provided, got {max_samples}.")

        ds = xr.open_mfdataset(str(file_pattern), combine="by_coords", chunks=chunks)
        time_coord, level_coord = _resolve_coord_names(ds)

        ds = ds.sortby(time_coord)
        available_levels = [int(v) for v in ds[level_coord].values.tolist()]
        levels_hpa = _select_levels(available_levels, levels_hpa)

        ds = ds.sel({level_coord: list(levels_hpa)})

        self.ds = ds
        self.levels_hpa = levels_hpa
        self.time_coord = time_coord
        self.level_coord = level_coord
        self.lat = ds["latitude"].values.astype(np.float32)
        self.lon = ds["longitude"].values.astype(np.float32)
        self.times = ds[time_coord].values
        self.input_channel_names = _input_channel_names(levels_hpa)
        self.input_norm = _validate_input_norm(input_norm, expected_channels=len(self.input_channel_names))
        self.time_stride = time_stride
        self.max_samples = max_samples

        n = ds.sizes[time_coord]
        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        if split == "train":
            self.start, self.end = 0, train_end
        elif split == "val":
            self.start, self.end = train_end, val_end
        elif split == "test":
            self.start, self.end = val_end, n
        else:
            raise ValueError(f"Unknown split: {split}")

        if self.end - self.start <= 0:
            raise ValueError(f"Split {split} is empty. start={self.start}, end={self.end}, n={n}")

        indices = np.arange(self.start, self.end, self.time_stride, dtype=np.int64)
        if self.max_samples is not None:
            indices = indices[: self.max_samples]
        if indices.size == 0:
            raise ValueError(
                f"Split {split} became empty after time_stride={self.time_stride}, max_samples={self.max_samples}."
            )
        self.time_indices = indices

    def __len__(self) -> int:
        return int(self.time_indices.size)

    def _at_level(self, var_name: str, t_idx: int, level_hpa: int) -> np.ndarray:
        arr = (
            self.ds[var_name]
            .isel({self.time_coord: t_idx})
            .sel({self.level_coord: level_hpa})
            .transpose("latitude", "longitude")
            .data
        )
        return np.asarray(arr.compute(), dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t_idx = int(self.time_indices[idx])
        l_top, l_mid, l_bottom = self.levels_hpa

        t_top = self._at_level("t", t_idx, l_top)
        t_mid = self._at_level("t", t_idx, l_mid)
        t_bottom = self._at_level("t", t_idx, l_bottom)
        z_mid = self._at_level("z", t_idx, l_mid)

        u_top = self._at_level("u", t_idx, l_top)
        v_top = self._at_level("v", t_idx, l_top)
        u_bottom = self._at_level("u", t_idx, l_bottom)
        v_bottom = self._at_level("v", t_idx, l_bottom)

        # Input channels: [T_top, T_mid, T_bottom, Z_mid, U_bottom, V_bottom]
        x = np.stack([t_top, t_mid, t_bottom, z_mid, u_bottom, v_bottom], axis=0)
        if self.input_norm is not None:
            x_mean, x_std = self.input_norm
            x = (x - x_mean[:, None, None]) / (x_std[:, None, None] + EPS)

        # Target channels at cruise level: [U_top, V_top]
        target_uv = np.stack([u_top, v_top], axis=0)

        sample = {
            "x": torch.from_numpy(x),
            "target_uv": torch.from_numpy(target_uv),
            "geopotential": torch.from_numpy(z_mid),
            "temperature_mid": torch.from_numpy(t_mid),
            "u_bottom": torch.from_numpy(u_bottom),
            "v_bottom": torch.from_numpy(v_bottom),
        }
        return sample


def build_era5_dataloaders(
    file_pattern: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    chunks: dict[str, int] | None = None,
    levels_hpa: tuple[int, int, int] | None = None,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    normalize_inputs: bool = True,
    time_stride: int = 1,
    max_samples: int | None = None,
    input_norm: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, DataLoader], Era5Metadata]:
    train_probe = Era5WindDataset(
        file_pattern=file_pattern,
        levels_hpa=levels_hpa,
        split="train",
        train_frac=train_frac,
        val_frac=val_frac,
        chunks=chunks,
        input_norm=None,
        time_stride=time_stride,
        max_samples=max_samples,
    )

    resolved_input_norm = _validate_input_norm(input_norm, expected_channels=len(train_probe.input_channel_names))
    if normalize_inputs and resolved_input_norm is None:
        resolved_input_norm = _compute_input_stats(
            ds=train_probe.ds,
            time_coord=train_probe.time_coord,
            level_coord=train_probe.level_coord,
            levels_hpa=train_probe.levels_hpa,
            start=train_probe.start,
            end=train_probe.end,
            sample_indices=train_probe.time_indices,
        )

    train_ds = Era5WindDataset(
        file_pattern=file_pattern,
        levels_hpa=train_probe.levels_hpa,
        split="train",
        train_frac=train_frac,
        val_frac=val_frac,
        chunks=chunks,
        input_norm=resolved_input_norm,
        time_stride=time_stride,
        max_samples=max_samples,
    )
    val_ds = Era5WindDataset(
        file_pattern=file_pattern,
        levels_hpa=train_probe.levels_hpa,
        split="val",
        train_frac=train_frac,
        val_frac=val_frac,
        chunks=chunks,
        input_norm=resolved_input_norm,
        time_stride=time_stride,
        max_samples=max_samples,
    )
    test_ds = Era5WindDataset(
        file_pattern=file_pattern,
        levels_hpa=train_probe.levels_hpa,
        split="test",
        train_frac=train_frac,
        val_frac=val_frac,
        chunks=chunks,
        input_norm=resolved_input_norm,
        time_stride=time_stride,
        max_samples=max_samples,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (num_workers > 0),
    }

    loaders = {
        "train": DataLoader(
            train_ds,
            shuffle=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            val_ds,
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            test_ds,
            shuffle=False,
            **loader_kwargs,
        ),
    }

    input_mean = resolved_input_norm[0] if resolved_input_norm is not None else None
    input_std = resolved_input_norm[1] if resolved_input_norm is not None else None
    meta = Era5Metadata(
        lat=train_ds.lat,
        lon=train_ds.lon,
        times=train_ds.times[train_ds.time_indices],
        levels_hpa=train_ds.levels_hpa,
        time_coord=train_ds.time_coord,
        level_coord=train_ds.level_coord,
        input_channel_names=train_ds.input_channel_names,
        input_mean=input_mean,
        input_std=input_std,
    )
    return loaders, meta

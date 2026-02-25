"""Entry-point training script for the phase-1 PIML baseline."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from models.pinn_model import PhysicsInformedWindModel
from training.train import TrainConfig, fit
from utils.era5_loader import build_era5_dataloaders


def _resolve_data_glob() -> str:
    """Find ERA5 NetCDF files in standard project locations."""
    candidates = [
        Path("data/raw/*.nc"),
        Path("data/*.nc"),
    ]
    for pattern in candidates:
        if list(Path(".").glob(str(pattern))):
            return str(pattern)
    raise FileNotFoundError(
        "No ERA5 NetCDF files found. Checked: data/raw/*.nc and data/*.nc"
    )


def _load_config(config_path: str | Path = "configs/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = _load_config()
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    physics_cfg = cfg.get("physics", {})

    data_glob = _resolve_data_glob()
    levels = tuple(data_cfg["levels_hpa"]) if "levels_hpa" in data_cfg else None
    split_cfg = data_cfg.get("split", {})
    try:
        loaders, meta = build_era5_dataloaders(
            file_pattern=data_glob,
            batch_size=int(train_cfg.get("batch_size", 8)),
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            chunks=data_cfg.get("chunking", {"time": 32}),
            levels_hpa=levels,
            train_frac=float(split_cfg.get("train", 0.7)),
            val_frac=float(split_cfg.get("val", 0.15)),
            normalize_inputs=bool(train_cfg.get("normalize_inputs", True)),
        )
    except ValueError as exc:
        if "Requested levels" not in str(exc):
            raise
        print(f"[warning] {exc}. Falling back to available levels in dataset.")
        loaders, meta = build_era5_dataloaders(
            file_pattern=data_glob,
            batch_size=int(train_cfg.get("batch_size", 8)),
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            chunks=data_cfg.get("chunking", {"time": 32}),
            levels_hpa=None,
            train_frac=float(split_cfg.get("train", 0.7)),
            val_frac=float(split_cfg.get("val", 0.15)),
            normalize_inputs=bool(train_cfg.get("normalize_inputs", True)),
        )

    model = PhysicsInformedWindModel(in_channels=6, hidden_channels=64)
    fit_cfg = TrainConfig(
        epochs=int(train_cfg.get("epochs", 20)),
        lr=float(train_cfg.get("lr", 1e-3)),
        use_mixed_precision=bool(train_cfg.get("use_mixed_precision", True)),
        lambda_geo=float(physics_cfg.get("lambda_geo", 0.05)),
        lambda_tw=float(physics_cfg.get("lambda_tw", 0.05)),
        device=str(train_cfg.get("device", "auto")),
        max_train_batches=(
            int(train_cfg["max_train_batches"]) if train_cfg.get("max_train_batches") is not None else None
        ),
        max_val_batches=(int(train_cfg["max_val_batches"]) if train_cfg.get("max_val_batches") is not None else None),
    )

    lat = torch.from_numpy(meta.lat)
    lon = torch.from_numpy(meta.lon)

    history = fit(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        lat_deg=lat,
        lon_deg=lon,
        cfg=fit_cfg,
    )
    print({k: v[-1] for k, v in history.items() if len(v) > 0})


if __name__ == "__main__":
    main()

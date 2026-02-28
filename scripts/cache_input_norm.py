"""Compute and cache ERA5 input normalization stats for faster training startup."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.era5_loader import compute_input_norm_artifacts, save_input_norm_npz


def _resolve_data_glob(cfg: dict) -> str:
    data_cfg = cfg.get("data", {})
    raw_glob = data_cfg.get("raw_glob")
    candidates: list[Path] = []
    if raw_glob:
        candidates.append(Path(str(raw_glob)))
    candidates.extend([Path("data/raw/*.nc"), Path("data/*.nc")])

    for pattern in candidates:
        if list(Path(".").glob(str(pattern))):
            return str(pattern)
    raise FileNotFoundError("No ERA5 NetCDF files found. Checked config raw_glob + default paths.")


def _load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache ERA5 input normalization stats to NPZ.")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--output", type=str, default="data/processed/input_norm_stats.npz")
    parser.add_argument("--time-stride", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})
    levels = tuple(data_cfg["levels_hpa"]) if "levels_hpa" in data_cfg else None

    time_stride = int(data_cfg.get("time_stride", 1))
    max_samples = data_cfg.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)

    if args.time_stride is not None:
        time_stride = args.time_stride
    if args.max_samples is not None:
        max_samples = args.max_samples

    data_glob = _resolve_data_glob(cfg)

    try:
        artifacts = compute_input_norm_artifacts(
            file_pattern=data_glob,
            chunks=data_cfg.get("chunking", {"time": 32}),
            levels_hpa=levels,
            train_frac=float(split_cfg.get("train", 0.7)),
            val_frac=float(split_cfg.get("val", 0.15)),
            time_stride=time_stride,
            max_samples=max_samples,
        )
    except ValueError as exc:
        if "Requested levels" not in str(exc):
            raise
        print(f"[warning] {exc}. Falling back to available levels in dataset.")
        artifacts = compute_input_norm_artifacts(
            file_pattern=data_glob,
            chunks=data_cfg.get("chunking", {"time": 32}),
            levels_hpa=None,
            train_frac=float(split_cfg.get("train", 0.7)),
            val_frac=float(split_cfg.get("val", 0.15)),
            time_stride=time_stride,
            max_samples=max_samples,
        )

    save_input_norm_npz(
        path=args.output,
        input_mean=artifacts.input_mean,
        input_std=artifacts.input_std,
        input_channel_names=artifacts.input_channel_names,
        levels_hpa=artifacts.levels_hpa,
    )

    print(
        {
            "saved": args.output,
            "data_glob": data_glob,
            "levels_hpa": artifacts.levels_hpa,
            "channels": artifacts.input_channel_names,
            "train_sample_count_used": artifacts.train_sample_count,
            "time_stride": time_stride,
            "max_samples": max_samples,
        }
    )


if __name__ == "__main__":
    main()

"""Entry-point training script for the phase-1 PIML baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from models.pinn_model import PhysicsInformedWindModel
from training.train import TrainConfig, fit
from utils.era5_loader import build_era5_dataloaders, load_input_norm_npz


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the phase-1 physics-informed wind model.")
    parser.add_argument("--config", type=str, default="configs/project.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Fast smoke run: epochs=2, batch_size=2, max_train_batches=3, max_val_batches=1, "
            "num_workers=0, device=cpu, normalize_inputs=false, time_stride=4, max_samples=256."
        ),
    )
    parser.add_argument(
        "--colab-gpu",
        action="store_true",
        help="Apply Colab GPU-friendly training settings (device=cuda, workers=2, pin_memory=true, AMP=true).",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-stride", type=int, default=None, help="Use every Nth timestamp (>=1).")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per split after striding.")
    parser.add_argument("--input-norm", type=str, default=None, help="Path to cached input norm NPZ.")
    parser.add_argument(
        "--realtime-logs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable realtime per-batch training logs.",
    )
    parser.add_argument(
        "--log-every-n-batches",
        type=int,
        default=None,
        help="Print training/validation logs every N batches (>=1).",
    )
    return parser.parse_args()


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    train_cfg = cfg.setdefault("training", {})
    data_cfg = cfg.setdefault("data", {})

    if args.quick:
        train_cfg["epochs"] = 2
        train_cfg["batch_size"] = 2
        train_cfg["max_train_batches"] = 3
        train_cfg["max_val_batches"] = 1
        train_cfg["num_workers"] = 0
        train_cfg["device"] = "cpu"
        train_cfg["use_mixed_precision"] = False
        train_cfg["normalize_inputs"] = False
        data_cfg["time_stride"] = 4
        data_cfg["max_samples"] = 256

    if args.colab_gpu:
        train_cfg["device"] = "cuda"
        train_cfg["num_workers"] = 2
        train_cfg["pin_memory"] = True
        train_cfg["use_mixed_precision"] = True
        train_cfg["realtime_logs"] = True

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.max_train_batches is not None:
        train_cfg["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        train_cfg["max_val_batches"] = args.max_val_batches
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.time_stride is not None:
        data_cfg["time_stride"] = args.time_stride
    if args.max_samples is not None:
        data_cfg["max_samples"] = args.max_samples
    if args.input_norm is not None:
        data_cfg["input_norm_path"] = args.input_norm
        train_cfg["normalize_inputs"] = True
    if args.realtime_logs is not None:
        train_cfg["realtime_logs"] = bool(args.realtime_logs)
    if args.log_every_n_batches is not None:
        train_cfg["log_every_n_batches"] = args.log_every_n_batches

    return cfg


def _probe_cuda() -> None:
    _ = torch.zeros(1, device="cuda")


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _probe_cuda()
        return True
    except Exception:
        return False


def _runtime_defaults(train_cfg: dict) -> tuple[int, bool, bool, str]:
    requested_device = str(train_cfg.get("device", "auto")).lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Unsupported training.device '{requested_device}'. Use one of auto, cpu, cuda."
        )

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        try:
            _probe_cuda()
        except Exception as exc:
            raise RuntimeError(f"CUDA was requested, but is unusable in this environment: {exc}") from exc
        use_gpu = True
    elif requested_device == "auto":
        use_gpu = _cuda_usable()
        if torch.cuda.is_available() and not use_gpu:
            warnings.warn("CUDA detected but unusable; defaulting to CPU settings.")
    else:
        use_gpu = False

    default_num_workers = 2 if use_gpu else 0
    default_pin_memory = use_gpu
    default_amp = use_gpu
    return default_num_workers, default_pin_memory, default_amp, requested_device


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    cfg = _apply_overrides(cfg, args)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    physics_cfg = cfg.get("physics", {})

    data_glob = _resolve_data_glob()
    levels = tuple(data_cfg["levels_hpa"]) if "levels_hpa" in data_cfg else None
    split_cfg = data_cfg.get("split", {})
    time_stride = int(data_cfg.get("time_stride", 1))
    max_samples = int(data_cfg["max_samples"]) if data_cfg.get("max_samples") is not None else None

    default_workers, default_pin_memory, default_amp, requested_device = _runtime_defaults(train_cfg)
    num_workers = int(train_cfg.get("num_workers", default_workers))
    pin_memory = bool(train_cfg.get("pin_memory", default_pin_memory))
    use_mixed_precision = bool(train_cfg.get("use_mixed_precision", default_amp))

    input_norm = None
    input_norm_path = data_cfg.get("input_norm_path")
    if input_norm_path:
        norm_path = Path(str(input_norm_path))
        if (not norm_path.exists()) and (not norm_path.is_absolute()):
            alt = PROJECT_ROOT / norm_path
            if alt.exists():
                norm_path = alt
        input_norm = load_input_norm_npz(norm_path)
        print(f"[info] Using cached input normalization from {norm_path}", flush=True)

    print(
        (
            f"[stage] Building ERA5 dataloaders... device={requested_device} "
            f"num_workers={num_workers} pin_memory={pin_memory} amp={use_mixed_precision}"
        ),
        flush=True,
    )
    try:
        loaders, meta = build_era5_dataloaders(
            file_pattern=data_glob,
            batch_size=int(train_cfg.get("batch_size", 8)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            chunks=data_cfg.get("chunking", {"time": 32}),
            levels_hpa=levels,
            train_frac=float(split_cfg.get("train", 0.7)),
            val_frac=float(split_cfg.get("val", 0.15)),
            normalize_inputs=bool(train_cfg.get("normalize_inputs", True)),
            time_stride=time_stride,
            max_samples=max_samples,
            input_norm=input_norm,
        )
    except ValueError as exc:
        if "Requested levels" not in str(exc):
            raise
        print(f"[warning] {exc}. Falling back to available levels in dataset.", flush=True)
        loaders, meta = build_era5_dataloaders(
            file_pattern=data_glob,
            batch_size=int(train_cfg.get("batch_size", 8)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            chunks=data_cfg.get("chunking", {"time": 32}),
            levels_hpa=None,
            train_frac=float(split_cfg.get("train", 0.7)),
            val_frac=float(split_cfg.get("val", 0.15)),
            normalize_inputs=bool(train_cfg.get("normalize_inputs", True)),
            time_stride=time_stride,
            max_samples=max_samples,
            input_norm=input_norm,
        )

    print(
        f"[stage] Dataloaders ready: train_batches={len(loaders['train'])} val_batches={len(loaders['val'])}",
        flush=True,
    )
    print("[stage] Initializing model...", flush=True)
    model = PhysicsInformedWindModel(
        in_channels=6,
        hidden_channels=int(model_cfg.get("hidden_channels", 64)),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_bottom_residual=bool(model_cfg.get("use_bottom_residual", True)),
    )
    p_top_hpa = float(meta.levels_hpa[0])
    p_bottom_hpa = float(meta.levels_hpa[2])
    fit_cfg = TrainConfig(
        epochs=int(train_cfg.get("epochs", 20)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        grad_clip_norm=(
            float(train_cfg["grad_clip_norm"]) if train_cfg.get("grad_clip_norm") is not None else None
        ),
        data_loss=str(train_cfg.get("data_loss", "huber")),
        huber_delta=float(train_cfg.get("huber_delta", 5.0)),
        use_mixed_precision=use_mixed_precision,
        lambda_geo=float(physics_cfg.get("lambda_geo", 0.05)),
        lambda_tw=float(physics_cfg.get("lambda_tw", 0.05)),
        lambda_div=float(physics_cfg.get("lambda_div", 0.0)),
        lambda_vort=float(physics_cfg.get("lambda_vort", 0.0)),
        physics_warmup_epochs=int(train_cfg.get("physics_warmup_epochs", 3)),
        lr_scheduler=str(train_cfg.get("lr_scheduler", "plateau")),
        plateau_factor=float(train_cfg.get("plateau_factor", 0.5)),
        plateau_patience=int(train_cfg.get("plateau_patience", 3)),
        min_lr=float(train_cfg.get("min_lr", 1e-6)),
        early_stopping_patience=(
            int(train_cfg["early_stopping_patience"])
            if train_cfg.get("early_stopping_patience") is not None
            else None
        ),
        early_stopping_min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
        p_top_hpa=p_top_hpa,
        p_bottom_hpa=p_bottom_hpa,
        device=requested_device,
        max_train_batches=(
            int(train_cfg["max_train_batches"]) if train_cfg.get("max_train_batches") is not None else None
        ),
        max_val_batches=(int(train_cfg["max_val_batches"]) if train_cfg.get("max_val_batches") is not None else None),
        realtime_logs=bool(train_cfg.get("realtime_logs", True)),
        log_every_n_batches=int(train_cfg.get("log_every_n_batches", 1)),
    )

    lat = torch.from_numpy(meta.lat)
    lon = torch.from_numpy(meta.lon)

    print("[stage] Starting training loop...", flush=True)
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

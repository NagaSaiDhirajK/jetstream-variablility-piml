"""Train pure-ML and PIML models, then compare atmosphere + aviation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from evaluation.evaluate_atmosphere import atmospheric_metrics
from evaluation.evaluate_aviation import evaluate_wind_products_on_corridor
from models.pinn_model import PhysicsInformedWindModel
from training.train import TrainConfig, fit
from utils.era5_loader import build_era5_dataloaders


def _resolve_data_glob() -> str:
    candidates = [
        Path("data/raw/*.nc"),
        Path("data/*.nc"),
    ]
    for pattern in candidates:
        if list(Path(".").glob(str(pattern))):
            return str(pattern)
    raise FileNotFoundError("No ERA5 NetCDF files found. Checked data/raw/*.nc and data/*.nc.")


def _load_config(config_path: str | Path = "configs/project.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_loaders(data_glob: str, cfg: dict):
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    split_cfg = data_cfg.get("split", {})
    levels = tuple(data_cfg["levels_hpa"]) if "levels_hpa" in data_cfg else None

    try:
        return build_era5_dataloaders(
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
        if "Requested levels" in str(exc):
            print(f"[warning] {exc}. Falling back to available levels in dataset.")
            return build_era5_dataloaders(
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
        raise


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device

    pred_uv = []
    true_uv = []
    geopotential = []
    temperature_mid = []
    u_bottom = []
    v_bottom = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        pred = model(x).cpu()

        pred_uv.append(pred)
        true_uv.append(batch["target_uv"].cpu())
        geopotential.append(batch["geopotential"].cpu())
        temperature_mid.append(batch["temperature_mid"].cpu())
        u_bottom.append(batch["u_bottom"].cpu())
        v_bottom.append(batch["v_bottom"].cpu())

    return {
        "pred_uv": torch.cat(pred_uv, dim=0),
        "true_uv": torch.cat(true_uv, dim=0),
        "geopotential": torch.cat(geopotential, dim=0),
        "temperature_mid": torch.cat(temperature_mid, dim=0),
        "u_bottom": torch.cat(u_bottom, dim=0),
        "v_bottom": torch.cat(v_bottom, dim=0),
    }


def _fit_cfg(cfg: dict, lambda_geo: float, lambda_tw: float, lambda_div: float) -> TrainConfig:
    train_cfg = cfg.get("training", {})
    return TrainConfig(
        epochs=int(train_cfg.get("epochs", 20)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        grad_clip_norm=(
            float(train_cfg["grad_clip_norm"]) if train_cfg.get("grad_clip_norm") is not None else None
        ),
        use_mixed_precision=bool(train_cfg.get("use_mixed_precision", True)),
        lambda_geo=lambda_geo,
        lambda_tw=lambda_tw,
        lambda_div=lambda_div,
        device=str(train_cfg.get("device", "auto")),
        max_train_batches=(
            int(train_cfg["max_train_batches"]) if train_cfg.get("max_train_batches") is not None else None
        ),
        max_val_batches=(int(train_cfg["max_val_batches"]) if train_cfg.get("max_val_batches") is not None else None),
    )


def main() -> None:
    cfg = _load_config()
    data_glob = _resolve_data_glob()
    loaders, meta = _build_loaders(data_glob, cfg)

    physics_cfg = cfg.get("physics", {})
    eval_cfg = cfg.get("evaluation", {})
    corridor_name = str(eval_cfg.get("corridor", "transcon_us"))
    max_eval_batches = eval_cfg.get("max_test_batches", 8)
    if max_eval_batches is not None:
        max_eval_batches = int(max_eval_batches)

    lat = torch.from_numpy(meta.lat)
    lon = torch.from_numpy(meta.lon)

    ml_model = PhysicsInformedWindModel(in_channels=6, hidden_channels=64)
    ml_history = fit(
        model=ml_model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        lat_deg=lat,
        lon_deg=lon,
        cfg=_fit_cfg(cfg, lambda_geo=0.0, lambda_tw=0.0, lambda_div=0.0),
    )

    piml_model = PhysicsInformedWindModel(in_channels=6, hidden_channels=64)
    piml_history = fit(
        model=piml_model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        lat_deg=lat,
        lon_deg=lon,
        cfg=_fit_cfg(
            cfg,
            lambda_geo=float(physics_cfg.get("lambda_geo", 0.05)),
            lambda_tw=float(physics_cfg.get("lambda_tw", 0.05)),
            lambda_div=float(physics_cfg.get("lambda_div", 0.0)),
        ),
    )

    ml_eval = _collect_predictions(ml_model, loaders["test"], max_batches=max_eval_batches)
    piml_eval = _collect_predictions(piml_model, loaders["test"], max_batches=max_eval_batches)

    true_uv = ml_eval["true_uv"]
    ml_pred = ml_eval["pred_uv"]
    piml_pred = piml_eval["pred_uv"]

    atm_ml = atmospheric_metrics(
        pred_uv=ml_pred,
        true_uv=true_uv,
        geopotential=ml_eval["geopotential"],
        temperature_mid=ml_eval["temperature_mid"],
        u_bottom=ml_eval["u_bottom"],
        v_bottom=ml_eval["v_bottom"],
        lat_deg=lat,
        lon_deg=lon,
    )
    atm_piml = atmospheric_metrics(
        pred_uv=piml_pred,
        true_uv=true_uv,
        geopotential=ml_eval["geopotential"],
        temperature_mid=ml_eval["temperature_mid"],
        u_bottom=ml_eval["u_bottom"],
        v_bottom=ml_eval["v_bottom"],
        lat_deg=lat,
        lon_deg=lon,
    )

    aviation = evaluate_wind_products_on_corridor(
        true_uv=true_uv.numpy(),
        ml_uv=ml_pred.numpy(),
        piml_uv=piml_pred.numpy(),
        lat_deg=meta.lat,
        lon_deg=meta.lon,
        corridor_name=corridor_name,
        aircraft_true_airspeed_ms=float(cfg.get("aviation", {}).get("aircraft_true_airspeed_ms", 230.0)),
    )

    summary = {
        "levels_hpa": list(meta.levels_hpa),
        "input_channels": meta.input_channel_names,
        "ml_last_val_total": float(ml_history["val_total"][-1]),
        "piml_last_val_total": float(piml_history["val_total"][-1]),
        "ml_atmosphere": atm_ml,
        "piml_atmosphere": atm_piml,
        "aviation_compare": aviation,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

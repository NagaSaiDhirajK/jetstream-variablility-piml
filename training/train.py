"""Phase-1 training loop for physics-informed cruise-level wind surrogate."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from training.losses import LossWeights, compute_total_loss


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    use_mixed_precision: bool = True
    lambda_geo: float = 0.05
    lambda_tw: float = 0.05
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None


def _device(cfg: TrainConfig) -> torch.device:
    requested = cfg.device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device setting: {cfg.device}. Use one of: auto, cpu, cuda.")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available.")
        return torch.device("cuda")

    if torch.cuda.is_available():
        try:
            # Probe a tiny CUDA kernel; some GPU/PyTorch combinations report available
            # but fail at first real kernel launch.
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as exc:
            warnings.warn(f"CUDA is visible but unusable; falling back to CPU. Reason: {exc}")
            return torch.device("cpu")

    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    cfg: TrainConfig,
    dev: torch.device,
) -> dict[str, float]:
    model.train()

    amp_enabled = cfg.use_mixed_precision and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    weights = LossWeights(lambda_geo=cfg.lambda_geo, lambda_tw=cfg.lambda_tw)

    sums = {"total": 0.0, "data": 0.0, "geo": 0.0, "tw": 0.0}
    n_batches = 0

    lat_deg = lat_deg.to(dev)
    lon_deg = lon_deg.to(dev)

    for batch in loader:
        if cfg.max_train_batches is not None and n_batches >= cfg.max_train_batches:
            break
        x = batch["x"].to(dev, non_blocking=True)
        y = batch["target_uv"].to(dev, non_blocking=True)
        geopotential = batch["geopotential"].to(dev, non_blocking=True)
        temperature_mid = batch["temperature_mid"].to(dev, non_blocking=True)
        u_bottom = batch["u_bottom"].to(dev, non_blocking=True)
        v_bottom = batch["v_bottom"].to(dev, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=dev.type, enabled=amp_enabled):
            pred = model(x)
            losses = compute_total_loss(
                pred_uv=pred,
                true_uv=y,
                geopotential=geopotential,
                temperature_mid=temperature_mid,
                u_ref_bottom=u_bottom,
                v_ref_bottom=v_bottom,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                weights=weights,
            )

        scaler.scale(losses.total).backward()
        scaler.step(optimizer)
        scaler.update()

        sums["total"] += float(losses.total.detach().cpu())
        sums["data"] += float(losses.data.detach().cpu())
        sums["geo"] += float(losses.geostrophic.detach().cpu())
        sums["tw"] += float(losses.thermal_wind.detach().cpu())
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    cfg: TrainConfig,
    dev: torch.device,
) -> dict[str, float]:
    model.eval()
    weights = LossWeights(lambda_geo=cfg.lambda_geo, lambda_tw=cfg.lambda_tw)

    sums = {"total": 0.0, "data": 0.0, "geo": 0.0, "tw": 0.0, "rmse_u": 0.0, "rmse_v": 0.0}
    n_batches = 0

    lat_deg = lat_deg.to(dev)
    lon_deg = lon_deg.to(dev)

    for batch in loader:
        if cfg.max_val_batches is not None and n_batches >= cfg.max_val_batches:
            break
        x = batch["x"].to(dev, non_blocking=True)
        y = batch["target_uv"].to(dev, non_blocking=True)
        geopotential = batch["geopotential"].to(dev, non_blocking=True)
        temperature_mid = batch["temperature_mid"].to(dev, non_blocking=True)
        u_bottom = batch["u_bottom"].to(dev, non_blocking=True)
        v_bottom = batch["v_bottom"].to(dev, non_blocking=True)

        pred = model(x)
        losses = compute_total_loss(
            pred_uv=pred,
            true_uv=y,
            geopotential=geopotential,
            temperature_mid=temperature_mid,
            u_ref_bottom=u_bottom,
            v_ref_bottom=v_bottom,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            weights=weights,
        )

        rmse_u = torch.sqrt(torch.mean((pred[:, 0] - y[:, 0]) ** 2))
        rmse_v = torch.sqrt(torch.mean((pred[:, 1] - y[:, 1]) ** 2))

        sums["total"] += float(losses.total.detach().cpu())
        sums["data"] += float(losses.data.detach().cpu())
        sums["geo"] += float(losses.geostrophic.detach().cpu())
        sums["tw"] += float(losses.thermal_wind.detach().cpu())
        sums["rmse_u"] += float(rmse_u.detach().cpu())
        sums["rmse_v"] += float(rmse_v.detach().cpu())
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    cfg: TrainConfig,
) -> dict[str, list[float]]:
    dev = _device(cfg)
    model.to(dev)
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    history = {
        "train_total": [],
        "train_data": [],
        "train_geo": [],
        "train_tw": [],
        "val_total": [],
        "val_data": [],
        "val_geo": [],
        "val_tw": [],
        "val_rmse_u": [],
        "val_rmse_v": [],
    }

    for _ in range(cfg.epochs):
        train_stats = train_epoch(model, train_loader, optimizer, lat_deg, lon_deg, cfg, dev)
        val_stats = validate_epoch(model, val_loader, lat_deg, lon_deg, cfg, dev)

        history["train_total"].append(train_stats["total"])
        history["train_data"].append(train_stats["data"])
        history["train_geo"].append(train_stats["geo"])
        history["train_tw"].append(train_stats["tw"])
        history["val_total"].append(val_stats["total"])
        history["val_data"].append(val_stats["data"])
        history["val_geo"].append(val_stats["geo"])
        history["val_tw"].append(val_stats["tw"])
        history["val_rmse_u"].append(val_stats["rmse_u"])
        history["val_rmse_v"].append(val_stats["rmse_v"])

    return history

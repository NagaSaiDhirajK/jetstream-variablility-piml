"""Phase-1 training loop for physics-informed cruise-level wind surrogate."""

from __future__ import annotations

from dataclasses import dataclass
import time
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
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    data_loss: str = "huber"
    huber_delta: float = 5.0
    use_mixed_precision: bool = True
    lambda_geo: float = 0.05
    lambda_tw: float = 0.05
    lambda_div: float = 0.0
    lambda_vort: float = 0.0
    physics_warmup_epochs: int = 3
    lr_scheduler: str = "plateau"
    plateau_factor: float = 0.5
    plateau_patience: int = 3
    min_lr: float = 1e-6
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    p_top_hpa: float = 200.0
    p_bottom_hpa: float = 350.0
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    realtime_logs: bool = True
    log_every_n_batches: int = 1


def _probe_cuda() -> None:
    _ = torch.zeros(1, device="cuda")


def _device(cfg: TrainConfig) -> torch.device:
    requested = cfg.device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device setting: {cfg.device}. Use one of: auto, cpu, cuda.")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is not available.")
        try:
            _probe_cuda()
        except Exception as exc:
            raise RuntimeError(f"CUDA was explicitly requested but is unusable: {exc}") from exc
        return torch.device("cuda")

    if torch.cuda.is_available():
        try:
            _probe_cuda()
            return torch.device("cuda")
        except Exception as exc:
            warnings.warn(f"CUDA is visible but unusable; falling back to CPU. Reason: {exc}")
            return torch.device("cpu")

    return torch.device("cpu")


def _physics_weight_factor(epoch_idx: int, cfg: TrainConfig) -> float:
    if cfg.physics_warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch_idx + 1) / float(cfg.physics_warmup_epochs))


def _loss_weights(cfg: TrainConfig, factor: float) -> LossWeights:
    return LossWeights(
        data_loss=cfg.data_loss,
        huber_delta=cfg.huber_delta,
        lambda_geo=cfg.lambda_geo * factor,
        lambda_tw=cfg.lambda_tw * factor,
        lambda_div=cfg.lambda_div * factor,
        lambda_vort=cfg.lambda_vort * factor,
    )


def _forward_model(
    model: nn.Module,
    x: torch.Tensor,
    u_bottom: torch.Tensor,
    v_bottom: torch.Tensor,
) -> torch.Tensor:
    try:
        return model(x, u_bottom=u_bottom, v_bottom=v_bottom)
    except TypeError:
        return model(x)


def _effective_total_batches(loader: DataLoader, cap: int | None) -> int:
    total = len(loader)
    if cap is not None:
        total = min(total, cap)
    return max(1, total)


def _should_log_batch(batch_idx: int, total_batches: int, cfg: TrainConfig) -> bool:
    if not cfg.realtime_logs:
        return False
    every = max(1, int(cfg.log_every_n_batches))
    return batch_idx == 1 or (batch_idx % every == 0) or (batch_idx == total_batches)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    cfg: TrainConfig,
    weights: LossWeights,
    dev: torch.device,
    epoch_idx: int,
) -> dict[str, float]:
    model.train()

    amp_enabled = cfg.use_mixed_precision and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    sums = {"total": 0.0, "data": 0.0, "geo": 0.0, "tw": 0.0, "div": 0.0, "vort": 0.0}
    n_batches = 0

    lat_deg = lat_deg.to(dev)
    lon_deg = lon_deg.to(dev)
    total_batches = _effective_total_batches(loader, cfg.max_train_batches)

    for batch_idx, batch in enumerate(loader, start=1):
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
            pred = _forward_model(model=model, x=x, u_bottom=u_bottom, v_bottom=v_bottom)
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
                p_top_hpa=cfg.p_top_hpa,
                p_bottom_hpa=cfg.p_bottom_hpa,
            )

        scaler.scale(losses.total).backward()
        if cfg.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        sums["total"] += float(losses.total.detach().cpu())
        sums["data"] += float(losses.data.detach().cpu())
        sums["geo"] += float(losses.geostrophic.detach().cpu())
        sums["tw"] += float(losses.thermal_wind.detach().cpu())
        sums["div"] += float(losses.divergence.detach().cpu())
        sums["vort"] += float(losses.vorticity.detach().cpu())
        n_batches += 1

        if _should_log_batch(batch_idx=batch_idx, total_batches=total_batches, cfg=cfg):
            print(
                (
                    f"[train] epoch={epoch_idx + 1} batch={batch_idx}/{total_batches} "
                    f"total={float(losses.total):.4f} data={float(losses.data):.4f} "
                    f"geo={float(losses.geostrophic):.4f} tw={float(losses.thermal_wind):.4f} "
                    f"div={float(losses.divergence):.4f} vort={float(losses.vorticity):.4f}"
                ),
                flush=True,
            )

    return {k: v / max(1, n_batches) for k, v in sums.items()}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    cfg: TrainConfig,
    weights: LossWeights,
    dev: torch.device,
    epoch_idx: int,
) -> dict[str, float]:
    model.eval()

    sums = {"total": 0.0, "data": 0.0, "geo": 0.0, "tw": 0.0, "div": 0.0, "vort": 0.0, "rmse_u": 0.0, "rmse_v": 0.0}
    n_batches = 0

    lat_deg = lat_deg.to(dev)
    lon_deg = lon_deg.to(dev)
    total_batches = _effective_total_batches(loader, cfg.max_val_batches)

    for batch_idx, batch in enumerate(loader, start=1):
        if cfg.max_val_batches is not None and n_batches >= cfg.max_val_batches:
            break
        x = batch["x"].to(dev, non_blocking=True)
        y = batch["target_uv"].to(dev, non_blocking=True)
        geopotential = batch["geopotential"].to(dev, non_blocking=True)
        temperature_mid = batch["temperature_mid"].to(dev, non_blocking=True)
        u_bottom = batch["u_bottom"].to(dev, non_blocking=True)
        v_bottom = batch["v_bottom"].to(dev, non_blocking=True)

        pred = _forward_model(model=model, x=x, u_bottom=u_bottom, v_bottom=v_bottom)
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
            p_top_hpa=cfg.p_top_hpa,
            p_bottom_hpa=cfg.p_bottom_hpa,
        )

        rmse_u = torch.sqrt(torch.mean((pred[:, 0] - y[:, 0]) ** 2))
        rmse_v = torch.sqrt(torch.mean((pred[:, 1] - y[:, 1]) ** 2))

        sums["total"] += float(losses.total.detach().cpu())
        sums["data"] += float(losses.data.detach().cpu())
        sums["geo"] += float(losses.geostrophic.detach().cpu())
        sums["tw"] += float(losses.thermal_wind.detach().cpu())
        sums["div"] += float(losses.divergence.detach().cpu())
        sums["vort"] += float(losses.vorticity.detach().cpu())
        sums["rmse_u"] += float(rmse_u.detach().cpu())
        sums["rmse_v"] += float(rmse_v.detach().cpu())
        n_batches += 1

        if _should_log_batch(batch_idx=batch_idx, total_batches=total_batches, cfg=cfg):
            print(
                (
                    f"[val]   epoch={epoch_idx + 1} batch={batch_idx}/{total_batches} "
                    f"total={float(losses.total):.4f} data={float(losses.data):.4f} "
                    f"rmse_u={float(rmse_u):.4f} rmse_v={float(rmse_v):.4f}"
                ),
                flush=True,
            )

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
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.lr_scheduler not in {"none", "plateau"}:
        raise ValueError(f"Unsupported lr_scheduler '{cfg.lr_scheduler}'. Use 'none' or 'plateau'.")
    scheduler = None
    if cfg.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            min_lr=cfg.min_lr,
        )

    history = {
        "lr": [],
        "physics_weight_factor": [],
        "train_total": [],
        "train_data": [],
        "train_geo": [],
        "train_tw": [],
        "train_div": [],
        "train_vort": [],
        "val_total": [],
        "val_data": [],
        "val_geo": [],
        "val_tw": [],
        "val_div": [],
        "val_vort": [],
        "val_rmse_u": [],
        "val_rmse_v": [],
    }

    best_val_total = float("inf")
    epochs_without_improve = 0

    for epoch_idx in range(cfg.epochs):
        epoch_start = time.time()
        weight_factor = _physics_weight_factor(epoch_idx=epoch_idx, cfg=cfg)
        weights = _loss_weights(cfg=cfg, factor=weight_factor)

        if cfg.realtime_logs:
            print(
                f"[epoch-start] epoch={epoch_idx + 1}/{cfg.epochs} "
                f"physics_weight_factor={weight_factor:.4f}",
                flush=True,
            )

        train_stats = train_epoch(model, train_loader, optimizer, lat_deg, lon_deg, cfg, weights, dev, epoch_idx)
        val_stats = validate_epoch(model, val_loader, lat_deg, lon_deg, cfg, weights, dev, epoch_idx)
        if scheduler is not None:
            scheduler.step(val_stats["total"])
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["lr"].append(current_lr)
        history["physics_weight_factor"].append(weight_factor)
        history["train_total"].append(train_stats["total"])
        history["train_data"].append(train_stats["data"])
        history["train_geo"].append(train_stats["geo"])
        history["train_tw"].append(train_stats["tw"])
        history["train_div"].append(train_stats["div"])
        history["train_vort"].append(train_stats["vort"])
        history["val_total"].append(val_stats["total"])
        history["val_data"].append(val_stats["data"])
        history["val_geo"].append(val_stats["geo"])
        history["val_tw"].append(val_stats["tw"])
        history["val_div"].append(val_stats["div"])
        history["val_vort"].append(val_stats["vort"])
        history["val_rmse_u"].append(val_stats["rmse_u"])
        history["val_rmse_v"].append(val_stats["rmse_v"])

        if cfg.realtime_logs:
            elapsed = time.time() - epoch_start
            print(
                (
                    f"[epoch-end] epoch={epoch_idx + 1}/{cfg.epochs} elapsed_sec={elapsed:.2f} lr={current_lr:.6f} "
                    f"train_total={train_stats['total']:.4f} val_total={val_stats['total']:.4f} "
                    f"val_rmse_u={val_stats['rmse_u']:.4f} val_rmse_v={val_stats['rmse_v']:.4f}"
                ),
                flush=True,
            )

        if cfg.early_stopping_patience is not None:
            improved = val_stats["total"] < (best_val_total - cfg.early_stopping_min_delta)
            if improved:
                best_val_total = val_stats["total"]
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= cfg.early_stopping_patience:
                    if cfg.realtime_logs:
                        print(
                            f"[early-stop] Triggered at epoch={epoch_idx + 1} after {epochs_without_improve} non-improving epochs.",
                            flush=True,
                        )
                    break

    return history

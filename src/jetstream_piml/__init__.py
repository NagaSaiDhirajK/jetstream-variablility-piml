"""Jetstream PIML package exports."""

from models.base_model import BaselineWindCNN
from models.pinn_model import PhysicsInformedWindModel
from evaluation.evaluate_atmosphere import atmospheric_metrics
from evaluation.evaluate_aviation import evaluate_route_fuel, evaluate_wind_products_on_corridor
from training.train import TrainConfig, fit
from utils.era5_loader import Era5Metadata, Era5WindDataset, build_era5_dataloaders

__all__ = [
    "BaselineWindCNN",
    "PhysicsInformedWindModel",
    "atmospheric_metrics",
    "Era5Metadata",
    "Era5WindDataset",
    "TrainConfig",
    "build_era5_dataloaders",
    "evaluate_route_fuel",
    "evaluate_wind_products_on_corridor",
    "fit",
]

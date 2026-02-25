"""Compatibility layer: import model classes from modular models package."""

from models.base_model import BaselineWindCNN
from models.pinn_model import PhysicsInformedWindModel

__all__ = ["BaselineWindCNN", "PhysicsInformedWindModel"]

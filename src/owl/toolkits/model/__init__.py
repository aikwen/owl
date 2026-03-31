from .base import OwlModel
from ..common import registry

MODELS = registry.Registry[OwlModel]("model")

__all__ = ["MODELS"]
from .base import OwlCriterion
from ..common import registry

CRITERIA = registry.Registry[OwlCriterion]("criterion")

__all__ = ["CRITERIA"]
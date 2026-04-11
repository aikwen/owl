from ..common.registry import Registry

EVALUATORS = Registry("evaluator")

from .default import  DefaultEvaluator

__all__ = ["EVALUATORS"]
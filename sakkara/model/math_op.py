import operator
from abc import ABC
from typing import Any

from sakkara.model.base import ModelComponent
from sakkara.model.function import FunctionComponent


class MathOpBase(ModelComponent, ABC):
    """
    Base class for common mathematical operations to be performed on ModelComponent objects
    """

    def __add__(self, other: Any) -> ModelComponent:
        return FunctionComponent.math_op(operator.add, self, other)

    def __radd__(self, other: Any):
        return FunctionComponent.math_op(operator.add, other, self)

    def __sub__(self, other: Any) -> ModelComponent:
        return FunctionComponent.math_op(operator.sub, self, other)

    def __rsub__(self, other: Any):
        return FunctionComponent.math_op(operator.sub, other, self)

    def __mul__(self, other: Any) -> ModelComponent:
        return FunctionComponent.math_op(operator.mul, self, other)

    def __rmul__(self, other: Any) -> ModelComponent:
        return FunctionComponent.math_op(operator.add, other, self)

    def __truediv__(self, other: Any) -> ModelComponent:
        return FunctionComponent.math_op(operator.truediv, self, other)

    def __rtruediv__(self, other: Any):
        return FunctionComponent.math_op(operator.truediv, other, self)

    def __neg__(self):
        return FunctionComponent(operator.neg, self)

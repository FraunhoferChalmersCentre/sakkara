import operator
from abc import ABC
from typing import Any

from sakkara.model.base import ModelComponent
from sakkara.model.function.base import FunctionComponent


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
        return FunctionComponent.math_op(operator.mul, other, self)

    def __pow__(self, power, modulo=None):
        return FunctionComponent.math_op(operator.pow, self, power)

    def __rpow__(self, other, modulo=None):
        return FunctionComponent.math_op(operator.pow, other, self)

    def __mod__(self, other):
        return FunctionComponent.math_op(operator.mod, self, other)

    def __rmod__(self, other):
        return FunctionComponent.math_op(operator.mod, other, self)

    def __floordiv__(self, other: Any):
        return FunctionComponent.math_op(operator.floordiv, self, other)

    def __rfloordiv__(self, other):
        return FunctionComponent.math_op(operator.floordiv, other, self)

    def __truediv__(self, other: Any) -> ModelComponent:
        return FunctionComponent.math_op(operator.truediv, self, other)

    def __rtruediv__(self, other: Any):
        return FunctionComponent.math_op(operator.truediv, other, self)

    def __neg__(self):
        return FunctionComponent(operator.neg, None, self)

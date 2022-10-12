import operator
from abc import ABC
from typing import Callable, Any, Set

import aesara.tensor as at

from sakkara.model.base import ModelComponent
from sakkara.relation.node import NodePair
from sakkara.relation.groupset import GroupSet


class CompositeComponent(ModelComponent, ABC):
    """
    Class for intermediate states from mathematical operations between components
    """

    def __init__(self, a: ModelComponent, b: ModelComponent, op: Callable[[Any, Any], Any], name: str = None):
        super().__init__(name)
        self.a = a
        self.b = b
        self.op = op

    def clear(self):
        self.a.clear()
        self.b.clear()

    def prebuild(self, groupset: GroupSet) -> None:
        for c in [self.a, self.b]:
            if c.variable is None:
                c.build(groupset)

    def build_node(self, groupset: GroupSet) -> None:
        self.node = NodePair(self.a.node, self.b.node).reduce_pair()

    def get_mapped_variable(self, component: ModelComponent) -> at.TensorVariable:
        if len(component.node) == 1:
            return component.variable
        mapping = self.node.map_to(component.node)
        if mapping is None:
            return component.variable
        else:
            return component.variable[mapping]

    def build_variable(self) -> None:
        a_var = self.get_mapped_variable(self.a)
        b_var = self.get_mapped_variable(self.b)
        self.variable = self.op(a_var, b_var)

    def retrieve_columns(self) -> Set[str]:
        return self.a.retrieve_columns().union(self.b.retrieve_columns())

    def __add__(self, other) -> ModelComponent:
        return CompositeComponent(self, other, operator.add)

    def __sub__(self, other) -> ModelComponent:
        return CompositeComponent(self, other, operator.sub)

    def __mul__(self, other) -> ModelComponent:
        return CompositeComponent(self, other, operator.mul)

    def __rmul__(self, other) -> ModelComponent:
        return CompositeComponent(other, self, operator.mul)

    def __truediv__(self, other) -> ModelComponent:
        return CompositeComponent(self, other, operator.truediv)


class OperationBaseComponent(ModelComponent, ABC):
    """
    Base class for components
    """
    def __init__(self, name: str, column: str):
        super().__init__(name)
        self.column = column

    def __add__(self, other) -> ModelComponent:
        return CompositeComponent(self, other, operator.add)

    def __sub__(self, other) -> ModelComponent:
        return CompositeComponent(self, other, operator.sub)

    def __mul__(self, other) -> ModelComponent:
        return CompositeComponent(self, other, operator.mul)

    def __rmul__(self, other) -> ModelComponent:
        return CompositeComponent(other, self, operator.mul)

    def __truediv__(self, other) -> ModelComponent:
        return CompositeComponent(self, other, operator.truediv)

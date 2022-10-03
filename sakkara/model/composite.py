import operator
from abc import ABC
from typing import Callable, Any, Set

import aesara.tensor as at

from sakkara.model.base import ModelComponent
from sakkara.relation.compositegroup import CompositeGroupPair
from sakkara.relation.groupset import GroupSet


class CompositeComponent(ModelComponent, ABC):
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

    def build_group(self, groupset: GroupSet) -> None:
        self.group = CompositeGroupPair(self.a.group, self.b.group)

    def get_mapped_variable(self, component: ModelComponent) -> at.TensorVariable:
        if self.group != component.group:
            mapping = list(map(lambda m: m.index, self.group.map_from(component.group)))
            return component.variable[mapping]
        else:
            return component.variable

    def build_variable(self) -> None:
        a_var = self.get_mapped_variable(self.a)
        b_var = self.get_mapped_variable(self.b)
        self.variable = self.op(a_var, b_var)

    def retrieve_group_names(self) -> Set[str]:
        return self.a.retrieve_group_names().union(self.b.retrieve_group_names())

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

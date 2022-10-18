import operator
from abc import ABC
from typing import Callable, Any, Set, Optional

import aesara.tensor as at

from sakkara.model.base import ModelComponent
from sakkara.relation.node import NodePair
from sakkara.relation.groupset import GroupSet

OP_SYMBOLS = {
    operator.add: '+',
    operator.sub: '-',
    operator.mul: '*',
    operator.truediv: '/'
}


class CompositeComponent(ModelComponent, ABC):
    """
    Class for intermediate states of mathematical operations between components
    """

    def __init__(self, a: ModelComponent, b: ModelComponent, op: Callable[[Any, Any], Any], name: str = None):
        super().__init__()
        self.name = name
        self.a = a
        self.b = b
        self.op = op

    def get_name(self) -> Optional[str]:
        if self.name is not None:
            return self.name
        a_name = self.a.get_name()
        if a_name is None:
            return None
        b_name = self.b.get_name()
        if b_name is None:
            return None

        if self.op not in OP_SYMBOLS:
            raise NotImplementedError('Operation not implemented')

        return a_name + ' ' + OP_SYMBOLS[self.op] + ' ' + b_name

    def set_name(self, name: str) -> None:
        if self.a.get_name() is None:
            self.a.set_name(name + '(left)')
        if self.b.get_name() is None:
            self.a.set_name(name + '(right)')

    def clear(self):
        self.a.clear()
        self.b.clear()

    def prebuild(self, groupset: GroupSet) -> None:
        for c in [self.a, self.b]:
            if c.get_name() is None:
                raise ValueError('All components involved in mathematical operations must be named')
            if c.variable is None:
                c.build(groupset)

    def build_node(self, groupset: GroupSet) -> None:
        self.node = NodePair(self.a.node, self.b.node).reduced_repr()

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

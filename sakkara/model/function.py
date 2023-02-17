import abc
import operator
from abc import ABC
from typing import Callable, Any, Set, Optional

import pytensor.tensor as pt

from sakkara.model.base import ModelComponent
from sakkara.relation.nodepair import NodePair
from sakkara.relation.groupset import GroupSet


class FunctionComponent(ModelComponent, ABC):
    """
    Class for intermediate states of mathematical operations between components
    """

    def __init__(self, fct: Callable[[Any, ...], Any], name: str = None, *args: ModelComponent):
        super().__init__()
        self.name = name
        self.args = args
        self.fct = fct

    def get_name(self) -> Optional[str]:
        if self.name is not None:
            return self.name

        name = ''
        for comp in self.args:
            comp_name = comp.get_name()
            if comp_name is None:
                return None
            name += comp_name

        return name

    def set_name(self, name: str) -> None:
        for i, comp in enumerate(self.args):
            if comp.get_name() is None:
                comp.set_name(name + '_arg' + str(i))

    def clear(self):
        self.variable = None
        for comp in self.args:
            comp.clear()

    def prebuild(self, groupset: GroupSet) -> None:
        unbuilt = [c for c in self.args]
        counter = 0
        while 0 < len(unbuilt):
            component = unbuilt.pop(0)
            if component.get_name() is None:
                # Other components may be needed to be built first
                if counter < len(self.args) * (1 + len(self.args)) / 2:
                    unbuilt.append(component)
                else:
                    # Did not help to build other component first, name must defined for this explicitly
                    raise ValueError('All arguments to must be named')
            elif component.variable is None:
                component.build(groupset)
            counter += 1

    def build_node(self, groupset: GroupSet) -> None:
        self.node = self.args[0].node
        for comp in self.args[1:]:
            self.node = NodePair(self.node, comp.node).reduced_repr()

    def build_variable(self) -> None:
        def get_mapped_variable(component: ModelComponent) -> pt.TensorVariable:
            if component.node.get_members().shape == (1,):
                return component.variable
            mapping = self.node.map_to(component.node)
            return component.variable[mapping]

        mapped_vars = tuple(map(get_mapped_variable, self.args))
        self.variable = self.fct(*mapped_vars)

    def retrieve_groups(self) -> Set[str]:
        group = self.args[0].retrieve_groups()
        for component in self.args[1:]:
            group = group.union(component.retrieve_groups())
        return group

    @staticmethod
    def math_op(fct: Callable, left: Any, right: Any) -> ModelComponent:
        if isinstance(left, ModelComponent) and isinstance(right, ModelComponent):
            return FunctionComponent(fct, None, left, right)
        if isinstance(left, ModelComponent):
            return FunctionComponent(lambda x: fct(x, right), None, left)
        return FunctionComponent(lambda x: fct(left, x), None, right)

    def __add__(self, other: Any) -> ModelComponent:
        return self.math_op(operator.add, self, other)

    def __sub__(self, other: Any) -> ModelComponent:
        return self.math_op(operator.sub, self, other)

    def __radd__(self, other):
        return self.math_op(operator.add, other, self)

    def __rsub__(self, other):
        return self.math_op(operator.sub, other, self)

    def __mul__(self, other: Any) -> ModelComponent:
        return self.math_op(operator.mul, self, other)

    def __rmul__(self, other: Any) -> ModelComponent:
        return self.math_op(operator.add, other, self)

    def __truediv__(self, other: Any) -> ModelComponent:
        return self.math_op(operator.truediv, self, other)

    def __rtruediv__(self, other: Any):
        return self.math_op(operator.truediv, other, self)

    def __neg__(self):
        return FunctionComponent(operator.neg, None, self)

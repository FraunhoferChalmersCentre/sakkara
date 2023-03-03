import operator
from abc import ABC
from itertools import chain
from typing import Callable, Any, Set, Optional

import pytensor.tensor as pt

from sakkara.model.base import ModelComponent
from sakkara.relation.nodepair import NodePair
from sakkara.relation.groupset import GroupSet


class FunctionComponent(ModelComponent, ABC):
    """
    Class for intermediate states of mathematical operations between components

    :param fct: Function to evaluate

    :param \**kwargs: See below

    :Arguments:
        * *arg* (``ModelComponent``) --
          Arguments passed to fct. If object does not inherit ``ModelComponent``, you may wrap it with :class:`sakkara.model.FixedValueComponent`

    :Keyword Arguments:
        * *kwarg* (``ModelComponent``) --
          Keyword argument passed to fct. If object does not inherit ``ModelComponent``, you may wrap it with :class:`sakkara.model.FixedValueComponent`
    """

    def __init__(self, fct: Callable[[Any, ...], Any], *args: ModelComponent, **kwargs: ModelComponent):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.fct = fct

    def get_name(self) -> Optional[str]:
        for comp in chain(self.args, self.kwargs.values()):
            comp_name = comp.get_name()
            if comp_name is None:
                return None

        return str(self.fct)

    def set_name(self, name: str) -> None:
        for i, comp in enumerate(self.args):
            if comp.get_name() is None:
                comp.set_name(name + '_' + str(self.fct) + '_arg' + str(i))

        for k, comp in self.kwargs.items():
            if comp.get_name() is None:
                comp.set_name(name + '_' + str(self.fct) + '_' + k)

    def clear(self):
        self.variable = None
        for comp in chain(self.args, self.kwargs.values()):
            comp.clear()

    def prebuild(self, groupset: GroupSet) -> None:
        unbuilt = [c for c in chain(self.args, self.kwargs.values())]

        all_args = len(self.args) + len(self.kwargs)
        counter = 0
        while 0 < len(unbuilt):
            component = unbuilt.pop(0)
            if component.get_name() is None:
                # Other components may be needed to be built first
                if counter < all_args * (1 + all_args) / 2:
                    unbuilt.append(component)
                else:
                    # Did not help to build other component first, name must be defined for this explicitly
                    raise ValueError('All arguments to must be named')
            elif component.variable is None:
                component.build(groupset)
            counter += 1

    def build_node(self, groupset: GroupSet) -> None:
        self.node = None
        for comp in chain(self.args, self.kwargs.values()):
            if self.node is None:
                self.node = comp.node
            else:
                self.node = NodePair(self.node, comp.node).reduced_repr()

    def build_variable(self) -> None:
        def get_mapped_variable(component: ModelComponent) -> pt.TensorVariable:
            if component.node.get_members().shape == (1,):
                return component.variable
            mapping = self.node.map_to(component.node)
            return component.variable[mapping]

        mapped_args = tuple(map(get_mapped_variable, self.args))
        mapped_kwargs = dict({k: get_mapped_variable(v) for k, v in self.kwargs.items()})
        self.variable = self.fct(*mapped_args, **mapped_kwargs)

    def retrieve_groups(self) -> Set[str]:
        group = set()
        for component in chain(self.args, self.kwargs.values()):
            group = group.union(component.retrieve_groups())
        return group

    @staticmethod
    def math_op(fct: Callable, left: Any, right: Any) -> ModelComponent:
        if isinstance(left, ModelComponent) and isinstance(right, ModelComponent):
            return FunctionComponent(fct, left, right)
        if isinstance(left, ModelComponent):
            return FunctionComponent(lambda x: fct(x, right), left)
        return FunctionComponent(lambda x: fct(left, x), right)

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
        return FunctionComponent(operator.neg, self)

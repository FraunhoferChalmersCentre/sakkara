import operator
from abc import ABC
from itertools import chain
from typing import Callable, Any, Set, Optional, Tuple

from sakkara.model.base import ModelComponent
from sakkara.relation.groupset import GroupSet
from sakkara.relation.representation import MinimalTensorRepresentation


class FunctionComponent(ModelComponent, ABC):
    """
    Class for intermediate states of mathematical operations between components. This class is intended for internal
    usage, to use a generic function in your model see :meth:`sakkara.model.f_`

    :param fct: Function to evaluate

    :param \**kwargs: See below

    :Arguments:
        * *arg* (``ModelComponent``) --
          Arguments passed to fct. If object does not inherit ``ModelComponent``, you may wrap it with :class:`sakkara.model.UnrepeatableComponent`

    :Keyword Arguments:
        * *kwarg* (``ModelComponent``) --
          Keyword argument passed to fct. If object does not inherit ``ModelComponent``, you may wrap it with :class:`sakkara.model.UnrepeatableComponent`
    """

    def __init__(self, fct: Callable[[Any, ...], Any], output_group: Optional[Tuple[str,]], *args: ModelComponent,
                 **kwargs: ModelComponent):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.fct = fct
        self.output_group = output_group

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
        self.representation = None
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

    def build_representation(self, groupset: GroupSet) -> None:
        self.input_representation = MinimalTensorRepresentation()
        for comp in chain(self.args, self.kwargs.values()):
            self.input_representation = MinimalTensorRepresentation(*self.input_representation.get_groups(),
                                                                    *comp.representation.get_groups())
        if self.output_group is None:
            self.representation = self.input_representation
        else:
            self.representation = MinimalTensorRepresentation(*tuple(map(lambda g: groupset[g], self.output_group)))

    def build_variable(self) -> None:
        mapped_args = tuple([c.representation.map(c.variable, self.input_representation) for c in self.args])
        mapped_kwargs = dict(
            {k: c.representation.map(c.variable, self.input_representation) for k, c in self.kwargs.items()})
        self.variable = self.fct(*mapped_args, **mapped_kwargs)

    def retrieve_groups(self) -> Set[str]:
        group = set()
        for component in chain(self.args, self.kwargs.values()):
            group = group.union(component.retrieve_groups())
        return group

    def to_minibatch(self, batch_size: int, group: str) -> 'ModelComponent':
        return FunctionComponent(self.fct,
                                 self.output_group,
                                 *[m.to_minibatch(batch_size, group) for m in self.args],
                                 **{k: v.to_minibatch(batch_size, group) for k, v in self.kwargs.items()})

    @staticmethod
    def math_op(fct: Callable, left: Any, right: Any) -> ModelComponent:
        if isinstance(left, ModelComponent) and isinstance(right, ModelComponent):
            return FunctionComponent(fct, None, left, right)
        if isinstance(left, ModelComponent):
            return FunctionComponent(lambda x: fct(x, right), None, left)
        return FunctionComponent(lambda x: fct(left, x), None, right)

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

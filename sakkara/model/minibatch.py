from abc import ABC
from functools import cache
from typing import Set, Optional

from sakkara.model.base import ModelComponent
from sakkara.model.math_op import MathOpBase
from sakkara.relation.groupset import GroupSet
from sakkara.relation.representation import MinimalTensorRepresentation


class MinibatchComponent(MathOpBase, ABC):
    """
    Wrapper component for mini-batching of any (repeatable component)

    :param component: Component to be wrapped.
    :param batch_size: Batch size of the mini-batch.
    :param group: Group to apply mini-batching on.
    """
    def __init__(self, component: ModelComponent, batch_size: int, group: str):
        super().__init__()
        self.component = component
        self.batch_size = batch_size
        self.group = group
        self.transformed_variable = None

    def clear(self) -> None:
        self.component.clear()

    @cache
    def retrieve_groups(self) -> Set[str]:
        return self.component.retrieve_groups()

    def prebuild(self, groupset: GroupSet) -> None:
        if self.component.variable is None:
            self.component.build(groupset)

    def build_representation(self, groupset: GroupSet) -> None:
        self.representation = MinimalTensorRepresentation(groupset[self.group])

        self.transformed_variable = self.component.representation.map(self.component.variable, self.representation)

    def to_minibatch(self, batch_size: int, group: str) -> 'ModelComponent':
        return self.component.to_minibatch(batch_size, group)

    def get_name(self) -> Optional[str]:
        return self.component.get_name()

    def set_name(self, name: str) -> None:
        return self.component.set_name(name)

    def build_variable(self) -> None:
        minibatch = self.representation.get_groups()[0].get_minibatch(self.batch_size)
        self.variable = self.transformed_variable[minibatch]

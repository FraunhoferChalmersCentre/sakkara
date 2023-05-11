from abc import ABC
from typing import Set

import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.model.wrapper import WrapperComponent
from sakkara.relation.groupset import GroupSet


class AbstractDeterministicComponent(WrapperComponent, ABC):

    def __init__(self, name: str, component: ModelComponent):
        super().__init__(component)
        self.name = name

    def build_representation(self, groupset: GroupSet) -> None:
        self.representation = self.component.representation

    def retrieve_groups(self) -> Set[str]:
        return self.component.retrieve_groups()


class MinibatchDeterministic(AbstractDeterministicComponent, ABC):
    def __init__(self, name: str, component: ModelComponent):
        super().__init__(name, component)

    def to_minibatch(self, batch_size: int, group: str) -> ModelComponent:
        return self

    def build_variable(self) -> None:
        self.variable = pm.Deterministic(self.name, self.component.variable)


class DeterministicComponent(AbstractDeterministicComponent, ABC):
    """
    Wrapper for :class:`pymc.Deterministic`

    :param name: Name that will be applied to the :class:`pymc.Deterministic` object.
    :param component: :class:`ModelComponent` whose corresponding PyMC variable wil be wrapped into :class:`pymc.Deterministic`
    """

    def __init__(self, name: str, component: ModelComponent):
        super().__init__(name, component)

    def to_minibatch(self, batch_size: int, group: str) -> ModelComponent:
        return MinibatchDeterministic(self.name, self.component.to_minibatch(batch_size, group))

    def build_variable(self) -> None:
        self.variable = pm.Deterministic(self.name, self.component.variable,
                                         dims=tuple(map(str, self.representation.groups)))

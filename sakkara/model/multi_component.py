from abc import ABC
from typing import List, Set

import aesara.tensor as at

from sakkara.model.base import ModelComponent
from sakkara.model.composite import OperationBaseComponent
from sakkara.relation.compositegroup import CompositeGroupPair
from sakkara.relation.groupset import GroupSet


class MultiComponent(OperationBaseComponent, ABC):
    def __init__(self, components: List[ModelComponent], name: str = None):
        super().__init__(name)
        self.components = components

    def clear(self):
        for c in self.components:
            c.clear()


class ConcatComponent(MultiComponent, ABC):
    def __init__(self, components: List[ModelComponent], keys: List[str], name: str = None):
        super().__init__(components, name)
        self.keys = keys
        self.key_mapping = {k: i for i, k in enumerate(keys)}
        self.components = components

    def prebuild(self, groupset: GroupSet) -> None:
        for c in self.components:
            if c.variable is None:
                c.build(groupset)

    def build_group(self, groupset: GroupSet) -> None:
        group = self.components[0].group
        for component in self.components[1:]:
            group = CompositeGroupPair(group, component.group)
        self.group = group

    def build_variable(self) -> None:
        self.variable = at.concatenate([c.variable for c in self.components])

    def __getitem__(self, item) -> ModelComponent:
        return self.components[self.key_mapping[item]]

    def retrieve_group_names(self) -> Set[str]:
        return set().union(*map(lambda c: c.retrieve_group_names(), self.components))


class StackedComponent(MultiComponent, ABC):
    def __init__(self, components: List[ModelComponent], name: str = None, group_name: str = None):
        super().__init__(components, name)
        self.group_name = group_name
        self.components = components

    def prebuild(self, groupset: GroupSet) -> None:
        for component, group_member in zip(groupset[self.group_name].get_members(), self.components):
            component.name = group_member.name
        for component in self.components:
            if component.variable is None:
                component.build(groupset)

    def build_group(self, groupset: GroupSet) -> None:
        self.group = groupset[self.group_name]

    def retrieve_group_names(self) -> Set[str]:
        return {self.group_name}

    def build_variable(self) -> None:
        self.variable = at.stack([c.variable for c in self.components])

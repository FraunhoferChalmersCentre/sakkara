from abc import ABC
from typing import Any, Set, Callable

from sakkara.model.base import ModelComponent
from sakkara.model.composite import OperationBaseComponent
from sakkara.relation.groupset import GroupSet


class SingleComponent(OperationBaseComponent, ABC):
    def build_group(self, groupset: GroupSet) -> None:
        self.group = groupset[self.group_name]


class ConstantComponent(SingleComponent, ABC):
    def __init__(self, value: Any, name: str = None, group_name: str = 'global'):
        super().__init__(name, group_name)
        self.values = value

    def clear(self):
        self.variable = None
        self.group = None

    def prebuild(self, groupset: GroupSet) -> None:
        pass

    def build_variable(self) -> None:
        self.variable = self.values

    def retrieve_group_names(self) -> Set[str]:
        return {self.group_name}


class HierarchicalComponent(SingleComponent, ABC):
    def __init__(self, distribution: Callable, group_name: str = 'global', name=None, **kwargs):
        super().__init__(name, group_name)
        self.distribution = distribution
        self.params = {k: v if isinstance(v, ModelComponent) else ConstantComponent(v) for k, v in kwargs.items()}
        self.variable = None
        self.group = None

    def clear(self):
        self.variable = None
        self.group = None
        for c in self.params.values():
            c.clear()

    def prebuild(self, groupset: GroupSet) -> None:
        for param_name, component in self.params.items():
            if component.variable is None:
                component.name = f'{param_name}_{self.name}' if component.name is None else component.name
            component.build(groupset)

    def build_variable(self) -> None:

        built_params = {}
        self_repr = self.group.get_representation_groups()

        for param_name, component in self.params.items():
            other_repr = component.group.get_representation_groups()

            if self_repr == other_repr or len(component.group) == 1:
                built_params[param_name] = component.variable
            else:
                mapping = list(map(lambda m: m.index, self.group.map_from(component.group)))
                built_params[param_name] = component.variable[mapping]

        dims = str(self.group) if str(self.group) != 'global' else None
        self.variable = self.distribution(self.name, **built_params, dims=dims)

    def retrieve_group_names(self) -> Set[str]:
        groups = set()
        for k, v in self.params.items():
            parent_groups = v.retrieve_group_names()
            groups = groups.union(parent_groups)
        if self.group_name is not None:
            groups.add(self.group_name)

        return groups




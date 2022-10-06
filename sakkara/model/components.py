import abc
from abc import ABC
from typing import Any, Set, Callable, Dict

import aesara.tensor as at

from sakkara.model.base import ModelComponent
from sakkara.model.composite import OperationBaseComponent
from sakkara.relation.group import GroupPair
from sakkara.relation.groupset import GroupSet


class Deterministic(OperationBaseComponent, ABC):
    def __init__(self, value: Any, name: str = None, group_name: str = 'global'):
        super().__init__(name, group_name)
        self.values = value

    def clear(self):
        self.variable = None
        self.group = None

    def prebuild(self, groupset: GroupSet) -> None:
        pass

    def build_group(self, groupset: GroupSet) -> None:
        self.group = groupset[self.group_name]

    def build_variable(self) -> None:
        self.variable = self.values

    def retrieve_group_names(self) -> Set[str]:
        return {self.group_name}


class HierarchicalComponent(OperationBaseComponent, ABC):
    def __init__(self, components: Dict[Any, Any], group_name: str, name: str):
        super().__init__(name, group_name)
        self.components = {k: v if isinstance(v, ModelComponent) else Deterministic(v) for k, v in
                           components.items()}
        self.variable = None
        self.group = None

    def __getitem__(self, item: Any):
        return self.components[item]

    def clear(self):
        self.variable = None
        self.group = None
        for c in self.components.values():
            c.clear()

    def prebuild(self, groupset: GroupSet) -> None:
        for param_name, component in self.components.items():
            if component.variable is None:
                component.name = f'{param_name}_{self.name}' if component.name is None else component.name
                component.build(groupset)

    def build_group(self, groupset: GroupSet) -> None:
        self.group = groupset[self.group_name]
        for component in self.components.values():
            self.group = GroupPair(self.group, component.group)

    def build_components(self) -> Dict[Any, at.TensorVariable]:
        built_params = {}
        self_repr = self.group.representation()

        for param_name, component in self.components.items():
            other_repr = component.group.representation()

            if self_repr == other_repr or len(component.group) == 1:
                built_params[param_name] = component.variable
            else:
                mapping = list(map(lambda m: m.index, self.group.map_from(component.group)))
                built_params[param_name] = component.variable[mapping]

        return built_params

    @abc.abstractmethod
    def build_root_variable(self, built_components: Dict[Any, at.TensorVariable]) -> None:
        raise NotImplementedError

    def build_variable(self) -> None:
        built_components = self.build_components()
        self.build_root_variable(built_components)

    def retrieve_group_names(self) -> Set[str]:
        groups = set()
        for k, v in self.components.items():
            parent_groups = v.retrieve_group_names()
            groups = groups.union(parent_groups)
        if self.group_name is not None:
            groups.add(self.group_name)

        return groups


class Distribution(HierarchicalComponent, ABC):
    def __init__(self, generator: Callable, group_name: str = 'global', name: str = None, **kwargs):
        super().__init__(components=kwargs, group_name=group_name, name=name)
        self.distribution = generator

    def build_root_variable(self, built_components: Dict[Any, at.TensorVariable]) -> None:
        rep_groups = self.group.representation()
        dims = None if len(rep_groups) == 1 and len(self.group) == 1 else str(self.group)
        self.variable = self.distribution(self.name, **built_components, dims=dims)


class Stacked(HierarchicalComponent, ABC):
    def __init__(self, components: Dict[Any, Any], group_name: str, name: str = None):
        super().__init__(components, group_name, name)

    def build_root_variable(self, built_components: Dict[Any, at.TensorVariable]) -> None:
        self.variable = at.stack([built_components[m] for m in self.group.get_members()]).flatten()


class Concat(HierarchicalComponent, ABC):
    def __init__(self, components: Dict[str, Any], group_name: str = 'global', name: str = None):
        super().__init__(components, group_name, name)

    def build_root_variable(self, built_components: Dict[str, at.TensorVariable]) -> None:
        self.variable = at.concatenate([built_components[m] for m in self.group.get_members()])

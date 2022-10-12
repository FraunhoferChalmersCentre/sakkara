import abc
from abc import ABC
from typing import Any, Set, Callable, Dict, Union, Optional

import aesara
import aesara.tensor as at
import pymc as pm
import numpy.typing as npt

from sakkara.model.base import ModelComponent
from sakkara.model.composite import OperationBaseComponent
from sakkara.relation.node import NodePair
from sakkara.relation.groupset import GroupSet


class Hyperparameter(ModelComponent, ABC):
    def __init__(self, value: Any, name: str = None):
        super().__init__(name)
        self.values = value

    def clear(self):
        self.variable = None
        self.node = None

    def prebuild(self, groupset: GroupSet) -> None:
        pass

    def build_node(self, groupset: GroupSet) -> None:
        self.node = groupset['global']

    def build_variable(self) -> None:
        self.variable = self.values

    def retrieve_columns(self) -> Set[str]:
        return {'global'}


class HierarchicalComponent(OperationBaseComponent, ABC):
    def __init__(self, name: str, column: str, components: Dict[Any, ModelComponent]):
        super().__init__(name, column)
        self.components = components

    def __getitem__(self, item: Any):
        return self.components[item]

    def clear(self):
        self.variable = None
        self.node = None
        for c in self.components.values():
            c.clear()

    def prebuild(self, groupset: GroupSet) -> None:
        for param_name, component in self.components.items():
            if component.variable is None:
                component.name = f'{param_name}_{self.name}' if component.name is None else component.name
                component.build(groupset)

    def retrieve_columns(self) -> Set[str]:
        columns = set()
        for k, v in self.components.items():
            parent_groups = v.retrieve_columns()
            columns = columns.union(parent_groups)
        if self.column is not None:
            columns.add(self.column)

        return columns


class PymcGenerateable(HierarchicalComponent):
    def __init__(self, generator: Callable, name: str, column: str, components: Dict[Any, ModelComponent]):
        super().__init__(name, column, components)
        self.generator = generator

    def build_node(self, groupset: GroupSet) -> None:
        self.node = groupset[self.column]

    def build_variable(self) -> None:
        built_components = {}

        for key, component in self.components.items():

            if self.node.representation() == component.node.representation() or len(component.node) == 1:
                built_components[key] = component.variable
            else:
                built_components[key] = component.variable[self.node.map_to(component.node)]

        if self.column is not None and 1 < len(self.node):
            dims = self.column
        else:
            dims = None

        self.variable = self.generator(self.name, **built_components, dims=dims)


class Deterministic(PymcGenerateable, ABC):
    def __init__(self, value: Union[float, npt.NDArray[float]], name: str = None, column: str = 'global'):
        super().__init__(pm.Deterministic, name, column, {'var': Hyperparameter(aesara.shared(value))})


class Distribution(PymcGenerateable, ABC):
    def __init__(self, generator: Callable, name: str = None, column: str = 'global', **kwargs):
        components = {k: v if isinstance(v, ModelComponent) else Hyperparameter(v) for k, v in kwargs.items()}
        super().__init__(generator, name, column, components)


class Concat(HierarchicalComponent, ABC):
    def __init__(self, components: Dict[Any, Any], column: str, name: str = None):
        components = {k: v if isinstance(v, ModelComponent) else Deterministic(v) for k, v in components.items()}
        super().__init__(name, column, components)
        self.column_node = None
        self.components_node = None

    def clear(self):
        super(Concat, self).clear()
        self.column_node = None
        self.components_node = None

    def build_node(self, groupset: GroupSet) -> None:
        self.column_node = groupset[self.column]

        self.components_node = next(iter(self.components.values())).node
        for key, component in self.components.items():
            self.components_node = NodePair(self.components_node, component.node).reduce_pair()

        self.node = NodePair(self.column_node, self.components_node).reduce_pair()

    def build_variable(self) -> None:
        built_components = {}

        for key, component in self.components.items():
            member_index = next(i for i, m in enumerate(map(str, self.column_node.get_members())) if m == key)
            node_to_column = self.node.map_to(self.column_node)
            node_to_component = self.node.map_to(self.components_node)
            components = node_to_component[node_to_column == member_index]

            if len(component.node) == 1:
                built_components[key] = at.stack([component.variable.flatten()] * len(components))
            else:
                mapping = self.components_node.map_to(component.node)[components]
                if mapping is None:
                    built_components[key] = component.variable
                else:
                    built_components[key] = component.variable[mapping]

        self.variable = at.concatenate([built_components[str(m)].flatten() for m in self.column_node.get_members()])

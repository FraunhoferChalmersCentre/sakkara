import abc
from abc import ABC
from functools import cache
from typing import Generic, Optional, Union, Tuple, Collection, Any, Dict, Set, TypeVar

import numpy as np
from aesara import tensor as at

from sakkara.model.base import ModelComponent
from sakkara.model.math_op import MathOpBase
from sakkara.relation.groupset import GroupSet
from sakkara.relation.nodepair import NodePair

S = TypeVar('S', bound=Any)
T = TypeVar('T', bound=ModelComponent)


class Composable(MathOpBase, ABC, Generic[S, T]):
    """
    Base class for a component that can be built with underlying components
    """

    def __init__(self, name: Optional[str], columns: Optional[Union[str, Tuple[str, ...]]],
                 members: Optional[Collection[Any]], components: Dict[S, T]):
        super().__init__()
        self.name = name
        self.members = members
        self.member_nodes = None
        self.columns = (columns,) if isinstance(columns, str) else columns
        self.components = components
        self.column_node = None
        self.components_node = None

    @abc.abstractmethod
    def __getitem__(self, item: Any) -> T:
        raise NotImplementedError

    def get_name(self) -> Optional[str]:
        return self.name

    def set_name(self, name: str) -> None:
        self.name = name

    def build_components(self, groupset: GroupSet) -> None:
        for param_name, component in self.components.items():
            if component.get_name() is None:
                component.set_name(f'{param_name}_{self.get_name()}')
            if component.variable is None:
                component.build(groupset)

    @abc.abstractmethod
    def get_built_components(self) -> Dict[S, at.Variable]:
        raise NotImplementedError

    def build_group_nodes(self, groupset: GroupSet) -> None:
        self.column_node = groupset[self.columns[0]]
        for column in self.columns[1:]:
            self.column_node = NodePair(groupset[column], self.column_node).reduced_repr()

        self.components_node = next(iter(self.components.values())).node if 0 < len(self.components) else groupset[
            'global']
        for component in self.components.values():
            self.components_node = NodePair(self.components_node, component.node).reduced_repr()

        self.node = NodePair(self.column_node, self.components_node).reduced_repr()

    def build_member_nodes(self):
        if self.members is None:
            self.member_nodes = self.column_node.get_members().ravel()
        else:
            self.members = {m if isinstance(m, Tuple) else (m,) for m in self.members}
            key_match = np.vectorize(lambda node: node.get_key() in self.members)
            self.member_nodes = self.column_node.get_members().ravel()[
                key_match(self.column_node.get_members().ravel())]

    def build_node(self, groupset: GroupSet):
        self.build_group_nodes(groupset)
        self.build_member_nodes()

    def clear(self):
        self.variable = None
        self.node = None
        for c in self.components.values():
            c.clear()

    @cache
    def retrieve_columns(self) -> Set[str]:
        columns = set()
        for k, v in self.components.items():
            parent_groups = v.retrieve_columns()
            columns = columns.union(parent_groups)
        if self.columns is not None:
            columns = columns.union(self.columns)

        return columns

    def dims(self):
        return tuple(map(str, self.node.representation()))

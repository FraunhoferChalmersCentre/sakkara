import abc
from abc import ABC
from functools import cache
from typing import Generic, Optional, Union, Tuple, Collection, Any, Dict, Set, TypeVar

import numpy as np
from pytensor import tensor as at

from sakkara.model.base import ModelComponent
from sakkara.model.math_op import MathOpBase
from sakkara.relation.groupset import GroupSet
from sakkara.relation.nodepair import NodePair

S = TypeVar('S', bound=Any)
T = TypeVar('T', bound=ModelComponent)


class Composable(MathOpBase, ABC, Generic[S, T]):
    """
    Base class for a component that can be built with underlying subcomponents
    """

    def __init__(self, name: Optional[str], group: Optional[Union[str, Tuple[str, ...]]],
                 members: Optional[Collection[Any]], subcomponents: Dict[S, T]):
        """

        Parameters
        ----------
        name: Name of the corresponding variable to register in PyMC.
        group: Group of which the component is defined for.
        members: Subset of members of column the component is defined for.
        subcomponents: Dict of underlying ModelComponent objects.
        """
        super().__init__()
        self.name = name
        self.members = members
        self.member_nodes = None
        self.group = (group,) if isinstance(group, str) else group
        self.subcomponents = subcomponents
        self.group_node = None
        self.components_node = None

    @abc.abstractmethod
    def __getitem__(self, item: Any) -> T:
        raise NotImplementedError

    def get_name(self) -> Optional[str]:
        return self.name

    def set_name(self, name: str) -> None:
        self.name = name

    def build_components(self, groupset: GroupSet) -> None:
        for param_name, component in self.subcomponents.items():
            if component.get_name() is None:
                component.set_name(f'{param_name}_{self.get_name()}')
            if component.variable is None:
                component.build(groupset)

    @abc.abstractmethod
    def get_built_components(self) -> Dict[S, at.Variable]:
        raise NotImplementedError

    def build_group_nodes(self, groupset: GroupSet) -> None:
        self.group_node = groupset[self.group[0]]
        for column in self.group[1:]:
            self.group_node = NodePair(self.group_node, groupset[column]).reduced_repr()

        self.components_node = next(iter(self.subcomponents.values())).node if 0 < len(self.subcomponents) else groupset[
            'global']
        for component in self.subcomponents.values():
            self.components_node = NodePair(self.components_node, component.node).reduced_repr()

        self.node = NodePair(self.group_node, self.components_node).reduced_repr()

    def build_member_nodes(self):
        if self.members is None:
            self.member_nodes = self.group_node.get_members().ravel()
        else:
            self.members = {m if isinstance(m, Tuple) else (m,) for m in self.members}
            key_match = np.vectorize(lambda node: node.get_key() in self.members)
            self.member_nodes = self.group_node.get_members().ravel()[
                key_match(self.group_node.get_members().ravel())]

    def build_node(self, groupset: GroupSet):
        self.build_group_nodes(groupset)
        self.build_member_nodes()

    def clear(self):
        self.variable = None
        self.node = None
        for c in self.subcomponents.values():
            c.clear()

    @cache
    def retrieve_groups(self) -> Set[str]:
        group = set()
        for k, v in self.subcomponents.items():
            parent_groups = v.retrieve_groups()
            group = group.union(parent_groups)
        if self.group is not None:
            group = group.union(self.group)

        return group

    def dims(self):
        return tuple(map(str, self.node.representation()))

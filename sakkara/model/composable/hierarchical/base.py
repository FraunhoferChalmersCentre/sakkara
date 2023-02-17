from abc import ABC
from typing import Any, Dict, Union, Tuple, Optional, Collection

import pytensor.tensor as pt

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import Composable, T
from sakkara.relation.groupset import GroupSet


class HierarchicalComponent(Composable[str, T], ABC):
    """
    Base class for hierarchical components, i.e., components that may be defined on one or several column, composed
    by other ModelComponent objects
    """
    def __init__(self, name: Optional[str], group: Optional[Union[str, Tuple[str, ...]]],
                 members: Optional[Collection[Any]], subcomponents: Dict[str, T]):
        """

        Parameters
        ----------
        name: Name of the corresponding variable to register in PyMC.
        group: Group of which the component is defined for.
        members: Subset of members of column the component is defined for.
        subcomponents: Dict of underlying ModelComponent objects.
        """
        super().__init__(name, group, members, subcomponents)

    def __getitem__(self, item: Any) -> ModelComponent:
        return self.subcomponents[item]

    def prebuild(self, groupset: GroupSet) -> None:
        self.build_components(groupset)

    def get_built_components(self) -> Dict[str, pt.Variable]:
        built_components = {}
        for key, comp in self.subcomponents.items():
            built_components[key] = comp.variable[self.node.map_to(comp.node)]

            if self.member_nodes.size < self.group_node.get_members().ravel().size:
                built_components[key] = built_components[key][self.node.member_subset(self.member_nodes)]

        return built_components



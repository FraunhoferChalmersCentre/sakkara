from abc import ABC
from typing import Any, Dict, Union, Tuple, Optional, Collection

import pytensor.tensor as at

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import Composable, T
from sakkara.relation.groupset import GroupSet


class HierarchicalComponent(Composable[str, T], ABC):
    def __init__(self, name: Optional[str], columns: Optional[Union[str, Tuple[str, ...]]],
                 members: Optional[Collection[Any]], components: Dict[str, T]):
        super().__init__(name, columns, members, components)

    def __getitem__(self, item: Any) -> ModelComponent:
        return self.components[item]

    def prebuild(self, groupset: GroupSet) -> None:
        self.build_components(groupset)

    def get_built_components(self) -> Dict[str, at.Variable]:
        built_components = {}
        for key, comp in self.components.items():
            built_components[key] = comp.variable[self.node.map_to(comp.node)]

            if self.member_nodes.size < self.column_node.get_members().ravel().size:
                built_components[key] = built_components[key][self.node.member_subset(self.member_nodes)]

        return built_components



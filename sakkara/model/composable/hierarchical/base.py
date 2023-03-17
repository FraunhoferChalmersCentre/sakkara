from abc import ABC
from typing import Any, Dict, Union, Tuple, Optional

import pytensor.tensor as pt

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import Composable, T
from sakkara.relation.groupset import GroupSet
from sakkara.relation.representation import MinimalTensorRepresentation


class HierarchicalComponent(Composable[str, T], ABC):
    """
    Base class for hierarchical components, i.e., components that may be defined on one or several column, composed
    by other :class:`ModelComponent` objects


    :param name: Name of the corresponding variable to register in PyMC.
    :param group: Group of which the component is defined for.
    :param members: Subset of members of column the component is defined for.
    :param subcomponents: Dict of underlying :class:`ModelComponent` objects.
    """

    def __init__(self, name: Optional[str], group: Optional[Union[str, Tuple[str, ...]]], subcomponents: Dict[str, T]):
        super().__init__(name, group, subcomponents)

    def __getitem__(self, item: Any) -> ModelComponent:
        return self.subcomponents[item]

    def prebuild(self, groupset: GroupSet) -> None:
        self.build_components(groupset)

    def build_representation(self, groupset: GroupSet):
        self.representation = MinimalTensorRepresentation(groupset['global'])

        if len(self.group) == 0:
            for comp in self.subcomponents.values():
                for g in comp.representation.get_groups():
                    self.representation.add_group(g)
        else:
            for g in self.group:
                self.representation.add_group(groupset[g])
            if tuple(map(str, self.representation.groups)) != self.group:
                raise ValueError('Groups are not a minimal')

    def get_built_components(self) -> Dict[str, pt.Variable]:
        built_components = {}
        for key, comp in self.subcomponents.items():
            intermediate_repr = MinimalTensorRepresentation(*self.representation.get_groups(),
                                                            *comp.representation.get_groups())

            intermediate_var = comp.representation.map(comp.variable, intermediate_repr)
            built_components[key] = intermediate_repr.map(intermediate_var, self.representation)

        return built_components

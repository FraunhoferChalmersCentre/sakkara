from abc import ABC
from typing import Optional, Union, Tuple

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import T
from sakkara.model.composable.hierarchical.base import HierarchicalComponent


class Reshaper(HierarchicalComponent[T], ABC):
    """
    Reshape a component

    :param component: The component to be reshaped.
    :param group: Group to reshape component into.
    """

    def __init__(self, component: ModelComponent, group: Optional[Union[str, Tuple[str, ...]]]):
        super().__init__(None, group, subcomponents={'var': component})

    def build_variable(self) -> None:
        self.variable = self.get_built_components()['var']

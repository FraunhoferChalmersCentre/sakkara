from abc import ABC
from typing import Union, Tuple, Optional, Collection, Any

from sakkara.model.composable.base import T
from sakkara.model.composable.hierarchical.base import HierarchicalComponent


class ComponentWrapper(HierarchicalComponent[T], ABC):
    def __init__(self, component: T, columns: Union[str, Tuple[str, ...]],
                 members: Optional[Collection[Any]], name: str = None):
        super().__init__(name, columns, members, {'component': component})

    def build_variable(self) -> None:
        self.variable = self.get_built_components()['component']

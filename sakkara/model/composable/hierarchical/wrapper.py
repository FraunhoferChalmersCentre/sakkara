from abc import ABC
from typing import Union, Tuple, Optional, Collection, Any

from sakkara.model.composable.base import T
from sakkara.model.composable.hierarchical.base import HierarchicalComponent


class ComponentWrapper(HierarchicalComponent[T], ABC):
    """
    Base class for wrapping a single component.


    :param component: ModelComponent object to wrap.
    :param group: Group which the component should be wrapped into.
    :param members: Members of the group that the component is defined for.
    :param name: Name of the component to appear in PyMC.
    """
    def __init__(self, component: T, group: Union[str, Tuple[str, ...]],
                 members: Optional[Collection[Any]], name: str = None):
        super().__init__(name, group, members, {'component': component})

    def build_variable(self) -> None:
        self.variable = self.get_built_components()['component']

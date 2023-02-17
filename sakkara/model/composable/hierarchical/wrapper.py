from abc import ABC
from typing import Union, Tuple, Optional, Collection, Any

from sakkara.model.composable.base import T
from sakkara.model.composable.hierarchical.base import HierarchicalComponent


class ComponentWrapper(HierarchicalComponent[T], ABC):
    """
    Base class for wrapping a single component.
    """
    def __init__(self, component: T, group: Union[str, Tuple[str, ...]],
                 members: Optional[Collection[Any]], name: str = None):
        """

        Parameters
        ----------
        component: ModelComponent object to wrap.
        group: Group which the component should be wrapped into.
        members: Members of the group that the component is defined for.
        name: Name of the component to appear in PyMC.
        """
        super().__init__(name, group, members, {'component': component})

    def build_variable(self) -> None:
        self.variable = self.get_built_components()['component']

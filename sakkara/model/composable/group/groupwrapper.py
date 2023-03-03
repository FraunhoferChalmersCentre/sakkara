from abc import ABC
from typing import Union, Tuple

from sakkara.model.base import ModelComponent
from sakkara.model.composable.hierarchical.wrapper import ComponentWrapper


class GroupWrapper(ComponentWrapper[ModelComponent], ABC):
    """
    Helper class for wrapping component in a (potentially) different group than originally defined.

    :param component: ModelComponent object to wrap.
    :param group: Group which the component should be wrapped into.
    :param name: Name of the component to appear in PyMC.
    """

    def __init__(self, component: ModelComponent, group: Union[str, Tuple[str, ...]] = None, name: str = None):
        super().__init__(component, group, None, name)

    def __getitem__(self, item) -> ModelComponent:
        return self.subcomponents[item]

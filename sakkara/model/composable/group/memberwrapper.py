from abc import ABC
from typing import Union, Tuple, Collection, Any

from sakkara.model.composable.hierarchical.wrapper import ComponentWrapper


class MemberWrapper(ComponentWrapper[ComponentWrapper], ABC):
    """
    Helper class for wrapping component that is defined only for a subset of members in a group.


    :param component: ModelComponent object to wrap.
    :param group: Group which the component should be wrapped into.
    :param members: Members of the group that the component is defined for.
    :param name: Name of the component to appear in PyMC.
    """

    def __init__(self, component: ComponentWrapper, group: Union[str, Tuple[str, ...]] = None,
                 members=Collection[Any], name: str = None):
        super().__init__(component, group, members, name)

    def __getitem__(self, item) -> ComponentWrapper:
        return self.subcomponents[item]

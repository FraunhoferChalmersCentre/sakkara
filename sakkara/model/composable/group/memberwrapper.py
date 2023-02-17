from abc import ABC
from typing import Union, Tuple, Collection, Any

from sakkara.model.composable.hierarchical.wrapper import ComponentWrapper


class MemberWrapper(ComponentWrapper[ComponentWrapper], ABC):
    """
    Helper class for wrapping component that is defined only for a subset of members in a group.
    """

    def __init__(self, component: ComponentWrapper, group: Union[str, Tuple[str, ...]] = None,
                 members=Collection[Any], name: str = None):
        """

        Parameters
        ----------
        component: ModelComponent object to wrap.
        group: Group which the component should be wrapped into.
        members: Members of the group that the component is defined for.
        name: Name of the component to appear in PyMC.
        """
        super().__init__(component, group, members, name)

    def __getitem__(self, item) -> ComponentWrapper:
        return self.subcomponents[item]

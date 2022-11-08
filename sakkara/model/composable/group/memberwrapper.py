from abc import ABC
from typing import Union, Tuple, Collection, Any

from sakkara.model.composable.hierarchical.wrapper import ComponentWrapper


class MemberWrapper(ComponentWrapper[ComponentWrapper], ABC):
    def __init__(self, component: ComponentWrapper, columns: Union[str, Tuple[str, ...]] = None,
                 members=Collection[Any], name: str = None):
        super().__init__(component, columns, members, name)

    def __getitem__(self, item) -> ComponentWrapper:
        return self.components[item]

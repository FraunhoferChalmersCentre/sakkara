from abc import ABC
from functools import cache
from typing import Optional, Set

from sakkara.model.base import ModelComponent
from sakkara.model.math_op import MathOpBase
from sakkara.relation.groupset import GroupSet


class WrapperComponent(MathOpBase, ABC):
    """
    Abstract class for component that wraps exactly one underlying component
    """
    def __init__(self, component: ModelComponent):
        super().__init__()
        self.component = component

    def get_name(self) -> Optional[str]:
        return self.component.get_name()

    def set_name(self, name: str) -> None:
        return self.component.set_name(name)

    def clear(self) -> None:
        self.component.clear()
        self.representation = None
        self.variable = None

    @cache
    def retrieve_groups(self) -> Set[str]:
        return self.component.retrieve_groups()

    def prebuild(self, groupset: GroupSet) -> None:
        if self.component.variable is None:
            self.component.build(groupset)
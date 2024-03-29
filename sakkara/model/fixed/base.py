from abc import ABC
from copy import deepcopy
from typing import Any, Union, Tuple, Optional, Set

from sakkara.model.base import ModelComponent
from sakkara.model.math_op import MathOpBase
from sakkara.relation.groupset import GroupSet

from sakkara.relation.representation import UnrepeatableRepresentation


class FixedValueComponent(MathOpBase, ABC):
    """
    Class for non-random fixed values that appear in model that are assigned to a group.

    :param value: Value to wrap into the component.
    :param name: Name of the corresponding variable to register in PyMC.
    :param group: Group of which the component is defined for.

    """

    def __init__(self, value: Any, group: Union[str, Tuple[str, ...]], name: Optional[str] = None):
        super().__init__()
        self.group = (group,) if isinstance(group, str) else group
        self.name = name
        self.values = value

    def set_name(self, name: str) -> None:
        self.name = name

    def clear(self):
        self.variable = None
        self.representation = None

    def prebuild(self, groupset: GroupSet) -> None:
        pass

    def retrieve_groups(self) -> Set[str]:
        return set(self.group)


class UnrepeatableComponent(FixedValueComponent, ABC):
    """
    Class for components that are fixed and cannot be repeated
    """

    def __init__(self, value: Any):
        super().__init__(value, 'global')

    def build_representation(self, groupset: GroupSet) -> None:
        self.representation = UnrepeatableRepresentation()

    def get_name(self) -> Optional[str]:
        return self.name if self.name is not None else 'fixed'

    def build_variable(self) -> None:
        self.variable = self.values

    def to_minibatch(self, batch_size: int, group: str) -> 'ModelComponent':
        return self

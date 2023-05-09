import abc
from abc import ABC
from functools import cache
from typing import Generic, Optional, Union, Tuple, Any, Dict, Set, TypeVar

from sakkara.model.base import ModelComponent
from sakkara.model.minibatch import MinibatchComponent
from sakkara.model.math_op import MathOpBase
from sakkara.relation.groupset import GroupSet

S = TypeVar('S', bound=Any)
T = TypeVar('T', bound=ModelComponent)


class Composable(MathOpBase, ABC, Generic[S, T]):
    """
    Base class for a component that can be built with underlying subcomponents



    :param name: Name of the corresponding variable to register in PyMC.

    :param group: Group of which the component is defined for.

    :param members: Subset of members of column the component is defined for.

    :param subcomponents: Dict of underlying ModelComponent objects.

    """

    def __init__(self, name: Optional[str], group: Optional[Union[str, Tuple[str, ...]]], subcomponents: Dict[S, T]):
        super().__init__()
        self.name = name
        if isinstance(group, str):
            self.group = (group,)
        elif group is None:
            self.group = tuple()
        else:
            self.group = group
        self.subcomponents = subcomponents
        self.base_representation = None
        self.components_representation = None

    @abc.abstractmethod
    def __getitem__(self, item: Any) -> T:
        raise NotImplementedError

    def to_minibatch(self, batch_size: int, group: str) -> 'ModelComponent':
        return MinibatchComponent(self, batch_size, group)

    def get_name(self) -> Optional[str]:
        return self.name

    def set_name(self, name: str) -> None:
        self.name = name

    def build_components(self, groupset: GroupSet) -> None:
        for param_name, component in self.subcomponents.items():
            if component.get_name() is None:
                component.set_name(f'{param_name}_{self.get_name()}')
            if component.variable is None:
                component.build(groupset)

    def clear(self):
        self.variable = None
        self.representation = None
        for c in self.subcomponents.values():
            c.clear()

    @cache
    def retrieve_groups(self) -> Set[str]:
        group = set()
        for k, v in self.subcomponents.items():
            parent_groups = v.retrieve_groups()
            group = group.union(parent_groups)
        if self.group is not None:
            group = group.union(self.group)

        return group

    def dims(self):
        return tuple(map(str, self.representation.groups))

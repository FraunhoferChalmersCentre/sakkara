from abc import ABC
from typing import Set, Any, Tuple

from sakkara.relation.node import Node
from sakkara.relation.representation import Representation
import numpy.typing as npt


class AtomicNode(Node, ABC):
    """
        Class for a single instance node.
    """

    def __init__(self, name: Any, members: npt.NDArray['AtomicNode'], parents: Set['AtomicNode']):
        """

        Parameters
        ----------
        name: Name of group
        parents: All (atomic) immediate and non-immediate parents to this atomic group.
        """
        self.name = name
        self.members = members
        self.parents = set(parents)
        self.children = set()
        self.twins = set()

    def get_key(self) -> Tuple[Any, ...]:
        return self.name,

    def __str__(self):
        return self.name

    def get_members(self) -> npt.NDArray[Node]:
        return self.members

    def add_child(self, child: 'AtomicNode'):
        self.children.add(child)

    def get_children(self) -> Set[Node]:
        return self.children

    def is_parent_to(self, other: Node) -> bool:
        return self in other.get_parents()

    def get_parents(self) -> Set[Node]:
        return self.parents

    def representation(self) -> Representation:
        return Representation({self})

    def reduced_repr(self) -> 'Node':
        return self

    def add_twin(self, twin: 'AtomicNode'):
        self.twins.add(twin)

    def get_twins(self) -> Set[Node]:
        return self.twins

    def is_twin_to(self, other: Node) -> bool:
        return self in other.get_twins()

    def __lt__(self, other) -> bool:
        return hash(self) < hash(other)

    def __len__(self) -> int:
        return 1


import abc
from abc import ABC
from functools import cache
from typing import Set, Optional, List

import numpy as np
import numpy.typing as npt

class Node:
    """
    Abstract class for relational nodes
    """

    @abc.abstractmethod
    def is_parent_to(self, other: 'Node') -> bool:
        """
        Test if another node is a child of the node

        Parameters
        ----------
        other : Node
            Other node to test

        Returns
        -------
        True if the node is parent to other, False otherwise

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_members(self) -> List['Node']:
        """
        Get all nodes linked as members of the node
        """

    def __len__(self):
        return len(self.get_members())

    @abc.abstractmethod
    def get_parents(self) -> Set['Node']:
        """
            Get all nodes linked as parents of the node
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_children(self) -> Set['Node']:
        raise NotImplementedError

    @abc.abstractmethod
    def representation(self) -> Set['Node']:
        """
            Returns the set of nodes that represents the information level of the group
        """
        raise NotImplementedError

    @abc.abstractmethod
    def map_to(self, other: 'Node') -> Optional[npt.NDArray[int]]:
        """
        Get a mapping from other group to this group. If the other group is unrelated to this group, the mapping will
        be from the other group to a pair group.

        Parameters
        ----------
        other: group to get mapping from

        Returns
        -------
        List of the member nodes of other, in an order to be mapped to the correct member of this group (or a pair group)
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class NodePair(Node, ABC):
    """
        Class for pair nodes, to allow for multi-node representations
    """

    def __init__(self, a: Node, b: Node):
        """

        Parameters
        ----------
        a: First group of pair
        b: Second group of pair
        """
        self.a = a
        self.b = b

    def parent_child_member_pairs(self) -> List['NodePair']:
        """
        Member initialization if a and b has a parent-child relation
        Parameters
        ----------
        a_parent: True if a is parent to b, False if b is parent to a

        Returns
        -------
        List of members, size same as member of child group
        """
        members = list()
        parent, child = (self.a, self.b) if self.a.is_parent_to(self.b) else (self.b, self.a)
        for i, child_member in enumerate(child.get_members()):
            # Find the member of parent group that is parent to this child member
            parent_member = next(filter(lambda p: p.is_parent_to(child_member), parent.get_members()))
            members.append(NodePair(parent_member, child_member))
        return members

    def same_level_member_pairs(self) -> List['NodePair']:
        """
        Member initialization when a and b are not related. Creates a pair member for each pair of (member a, member b)
        where both member_a and member_b must be linked to the same parent member of a common parent of groups a and b.
        If group a and b have no common parent, the pair members will be the carthesian product of members of a and the
        members of b.
        Returns
        -------
        List of members, size dependent on relations to common parent
        """
        members = list()
        for a_member in self.a.get_members():
            for b_member in self.b.get_members():
                if 0 < len(a_member.get_children().intersection(b_member.get_children())):
                    members.append(NodePair(a_member, b_member))
        return members

    @cache
    def get_members(self) -> List['NodePair']:
        """
        Init the member list, varying strategies depending on relation of a and b.
        Returns
        -------
        List of OrderedGroupPair, where a element of each OrderedGroupPair object corresponds to a member of the a group
        and b corresponds to a member of the b group.
        """
        if len(self.a.get_members()) == 0 and len(self.b.get_members()) == 0:
            return list()

        # Check if a and b has the same representation
        if self.a.representation() == self.b.representation():
            # a and b have the tha same representation, create a pair member for each member of a mapped to the same
            # element of b
            return list(map(lambda m: NodePair(m[0], m[1]), zip(self.a.get_members(), self.b.get_members())))

        # Check parent-child relationship
        if self.a.is_parent_to(self.b) or self.b.is_parent_to(self.a):
            # a and b have a parent-child relation
            return self.parent_child_member_pairs()

        # a and b are unrelated
        return self.same_level_member_pairs()

    @cache
    def is_parent_to(self, other: Node) -> bool:
        return self.a.is_parent_to(other) and self.b.is_parent_to(other)

    @cache
    def get_parents(self) -> Set[Node]:
        parents = set().union(*map(lambda g: g.get_parents(), self.representation()))

        if len(self.representation()) > 1:
            parents = parents.union(self.representation())

        return parents

    @cache
    def get_children(self) -> Set[Node]:
        return set().intersection(*map(lambda g: g.get_children(), self.representation()))

    @cache
    def representation(self) -> Set[Node]:
        if self.b.is_parent_to(self.a):
            return self.a.representation()

        if self.a.is_parent_to(self.b):
            return self.b.representation()

        return self.a.representation().union(self.b.representation())

    @cache
    def reduce_pair(self) -> Node:
        if self.representation() == self.a.representation():
            return self.a
        if self.representation() == self.b.representation():
            return self.b
        return self

    @cache
    def __str__(self) -> str:
        return '_'.join(map(str, self.representation()))

    @cache
    def map_to(self, other: Node) -> Optional[npt.NDArray[int]]:
        if other.representation() == self.representation():
            return np.arange(len(self))

        if not other.is_parent_to(self):
            return NodePair(self, other).map_to(other)

        if self.is_parent_to(other):
            raise ValueError('Mapping from parent to children not applicable')

        return np.array([
            next(
                j for j, other_member in enumerate(other.get_members()) if other_member.is_parent_to(member)
            ) for member in self.get_members()
        ])


class AtomicNode(Node, ABC):
    """
        Class for a single instance node.
    """

    def __init__(self, name: str, members: List['AtomicNode'], parents: Set['AtomicNode']):
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

    def __str__(self):
        return self.name

    def get_members(self) -> List[Node]:
        return self.members

    def add_child(self, child: 'AtomicNode'):
        self.children.add(child)

    def get_children(self) -> Set[Node]:
        return self.children

    def is_parent_to(self, other: Node) -> bool:
        return self in other.get_parents()

    def get_parents(self) -> Set[Node]:
        return self.parents

    def representation(self) -> Set[Node]:
        return {self}

    def map_to(self, other: Node) -> Optional[npt.NDArray[int]]:
        return NodePair(self, other).map_to(other)


class CellNode(AtomicNode):
    def __init__(self, name: str, parents: Set[AtomicNode]):
        super().__init__(name, list(), parents)

import abc
from abc import ABC
from functools import cache
from typing import Set, List, Any, Tuple, Callable

import numpy as np
import numpy.typing as npt


class Node:
    """
    Abstract class for relational nodes
    """

    @abc.abstractmethod
    def get_key(self) -> Tuple[Any, ...]:
        raise NotImplementedError

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
    def get_members(self) -> npt.NDArray['Node']:
        """
        Get all nodes linked as members of the node
        """

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

    def get_mapping(self, other: 'Node', match_fct: Callable) -> Tuple[npt.NDArray[int], ...]:
        raveled_mapping = np.empty(self.get_members().ravel().shape, dtype=int)
        raveled_self_members = self.get_members().ravel()
        raveled_other_members = other.get_members().ravel()

        for raveled_self_member_index, self_member in enumerate(raveled_self_members):
            raveled_index = np.argmax(match_fct(self_member, raveled_other_members))
            raveled_mapping[raveled_self_member_index] = raveled_index

        return tuple(map(lambda x: x.reshape(self.get_members().shape).tolist(),
                        np.unravel_index(raveled_mapping, other.get_members().shape)))

    @cache
    def map_to(self, other: 'Node') -> Tuple[npt.NDArray[int], ...]:
        """
        Get a mapping from other group to this group.

        Parameters
        ----------
        other: group to get mapping from

        Returns
        -------
        List of the member nodes of other, in an order to be mapped to the correct member of this group (or a pair group)
        """
        if self.representation() == other.representation():
            return self.get_mapping(other,
                                    np.vectorize(
                                        lambda om, sm: om.representation().intersection(sm.representation())))

        if self.is_parent_to(other):
            raise ValueError('Mapping from parent to child is not applicable.')

        if not other.is_parent_to(self):
            raise ValueError('Mapping between unrelated nodes is not applicable.')

        # Other node is parent to self
        return self.get_mapping(other, np.vectorize(
            lambda sm, om: len(om.representation().intersection(sm.get_parents()))))

    @abc.abstractmethod
    def reduced_repr(self) -> 'Node':
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

    def get_key(self) -> Tuple[Any, ...]:
        return self.a.get_key() + self.b.get_key()

    def parent_child_member_pairs(self) -> npt.NDArray['NodePair']:
        """
        Member initialization if a and b has a parent-child relation
        Parameters

        Returns
        -------
        List of members, size same as member of child group
        """
        parent, child = (self.a, self.b) if self.a.is_parent_to(self.b) else (self.b, self.a)
        members = np.empty(child.get_members().shape, dtype=object)
        raveled_members = members.ravel()
        raveled_parents = parent.get_members().ravel()
        for i, child_member in enumerate(child.get_members().ravel()):
            # Find the member of parent group that is parent to this child member
            parent_member = next(filter(lambda p: p.is_parent_to(child_member), raveled_parents))
            raveled_members[i] = NodePair(parent_member, child_member)
        return np.array(members)

    def same_level_member_pairs(self) -> npt.NDArray['NodePair']:
        """
        Member initialization when a and b are not related. Creates a pair member for each pair of (member a, member b)
        where both member_a and member_b must be linked to the same parent member of a common parent of groups a and b.
        If group a and b have no common parent, the pair members will be the carthesian product of members of a and the
        members of b.
        Returns
        -------
        List of members, size dependent on relations to common parent
        """

        unraveled_a = self.a.get_members().ravel()
        unraveled_b = self.b.get_members().ravel()

        semi_raveled = np.empty((len(unraveled_a), len(unraveled_b)), dtype=object)
        for i, a_member in enumerate(unraveled_a):
            for j, b_member in enumerate(unraveled_b):
                semi_raveled[i, j] = NodePair(a_member, b_member)

        return semi_raveled.reshape(self.a.get_members().shape + self.b.get_members().shape)

    @cache
    def get_members(self) -> npt.NDArray[Node]:
        """
        Init the member list, varying strategies depending on relation of a and b.
        Returns
        -------
        List of OrderedGroupPair, where a element of each OrderedGroupPair object corresponds to a member of the a group
        and b corresponds to a member of the b group.
        """
        if len(self.a.get_members()) == 0 and len(self.b.get_members()) == 0:
            return np.array(list())

        # Check if a and b has the same representation
        if self.a.representation() == self.b.representation():
            # a and b have the same representation nodes, possibly different order
            return self.a.get_members()

        # Check parent-child relationship
        if self.a.is_parent_to(self.b) or self.b.is_parent_to(self.a):
            # a and b have a parent-child relation
            return self.parent_child_member_pairs()

        # a and b are unrelated
        return self.same_level_member_pairs()

    @cache
    def is_parent_to(self, other: Node) -> bool:
        if self.representation() == other.representation():
            return False
        if self.a.is_parent_to(self.b):
            return self.b.is_parent_to(other)
        if self.b.is_parent_to(self.a):
            return self.a.is_parent_to(other)
        return self.a.is_parent_to(other) or self.b.is_parent_to(other)

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
    def reduced_repr(self) -> Node:
        if self.representation() == self.a.representation():
            return self.a.reduced_repr()
        if self.representation() == self.b.representation():
            return self.b.reduced_repr()
        return self

    @cache
    def __str__(self) -> str:
        return '_'.join(map(str, self.representation()))


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

    def representation(self) -> Set[Node]:
        return {self}

    def reduced_repr(self) -> 'Node':
        return self


class CellNode(AtomicNode):
    def __init__(self, name: Any, parents: Set[AtomicNode]):
        super().__init__(name, np.array(list()), parents)

from sakkara.relation.node import Node
from sakkara.relation.representation import Representation
from abc import ABC
from typing import Set, Any, Tuple
from functools import cache
import numpy.typing as npt
import numpy as np


class NodePair(Node, ABC):
    """
        Class for pair nodes, to allow for multi-node representations
    """

    def __init__(self, a: Node, b: Node, rep_nodes: 'Representation' = None):
        """

        Parameters
        ----------
        a: First group of pair
        b: Second group of pair
        """
        self.a = a
        self.b = b
        self.rep_nodes = rep_nodes

    def set_rep_nodes(self, rep_nodes):
        self.rep_nodes = rep_nodes

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

    def is_parent_to(self, other: Node) -> bool:
        if self.representation() == other.representation():
            return False
        if self.a.is_parent_to(self.b):
            return self.b.is_parent_to(other)
        if self.b.is_parent_to(self.a):
            return self.a.is_parent_to(other)
        a_par_other = self.a.is_parent_to(other)
        b_par_other = self.b.is_parent_to(other)
        a_twin_other = self.a.is_twin_to(other)
        b_twin_other = self.b.is_twin_to(other)
        return (a_par_other and b_twin_other) or (b_par_other and a_twin_other) or (a_par_other and b_par_other)

    def get_parents(self) -> Set[Node]:
        parents = set().union(*map(lambda g: g.get_parents(), self.representation()))

        if len(self.representation()) > 1:
            parents = parents.union(self.representation())

        return parents

    def get_children(self) -> Set[Node]:
        return set().intersection(*map(lambda g: g.get_children(), self.representation()))

    def get_twins(self) -> Set['Node']:
        twins = set().union(*map(lambda g: g.get_twins(), self.representation()))

        return twins

    def is_twin_to(self, other: Node) -> bool:
        if self.representation().get_nodes() == other.representation().get_nodes():
            return False
        return self.a.is_twin_to(other) and self.b.is_twin_to(other)

    # @cache
    def representation(self) -> Representation:
        if self.rep_nodes is None:
            if self.b.is_parent_to(self.a):
                self.rep_nodes = self.a.representation()
                return self.rep_nodes

            if self.a.is_parent_to(self.b):
                self.rep_nodes = self.b.representation()
                return self.rep_nodes

            self.rep_nodes = self.a.representation().union(self.b.representation())
        return self.rep_nodes

    def reduced_repr(self) -> Node:

        if self.representation() == self.a.representation() and self.a.__len__() == self.representation().__len__():
            return self.a.reduced_repr()
        if self.representation() == self.b.representation() and self.b.__len__() == self.representation().__len__():
            return self.b.reduced_repr()

        if self.__len__() != self.representation().__len__():
            return generate_nodepair(self.representation())
        return self

    # @cache
    def __str__(self) -> str:
        return '_'.join(map(str, self.representation()))

    def __len__(self) -> int:
        return len(self.a) + len(self.b)


def generate_nodepair(nodes: Representation):
    if len(nodes) == 1:
        return list(nodes)[0]
    else:
        lnodes = list(nodes.get_nodes())
        nodepair = NodePair(lnodes[0], lnodes[1])
        for node in lnodes[2:]:
            nodepair = NodePair(nodepair, node)
        nodepair.set_rep_nodes(nodes)
    return nodepair

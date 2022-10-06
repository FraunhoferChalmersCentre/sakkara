import abc
from abc import ABC
from typing import Set, Optional, List


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
            Other node to test if the node is parent to

        Returns
        -------
        True if the node is parent to other, False otherwise

        """
        raise NotImplementedError

    def is_member_of(self, other: 'Node') -> bool:
        return self in other.get_members()

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
    def representation(self) -> Set['Node']:
        """
            Returns the set of nodes that represents the information level of the group
        """
        raise NotImplementedError

    @abc.abstractmethod
    def map_from(self, other: 'Node') -> List[int]:
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

    @abc.abstractmethod
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
        self.members = self.create_members()

    def __len__(self):
        return len(self.members)

    def a_parent_b(self) -> Optional[bool]:
        """
        Get the relation in the pair
        :return: True if self.a is parent to self.b, False if self.b is parent to self.a or None if they are unrelated
        """
        if self.a.is_parent_to(self.b):
            return True
        elif self.b.is_parent_to(self.a):
            return False
        else:
            return None

    def is_parent_to(self, other: Node) -> bool:
        return self.a.is_parent_to(other) and self.b.is_parent_to(other)

    def get_parents(self) -> Set[Node]:
        representation_groups = self.representation()
        parents = set().union(*map(lambda g: g.get_parents(), self.representation()))

        if len(representation_groups) > 1:
            parents = parents.union(representation_groups)

        return parents

    def parent_child_init(self, a_parent: bool) -> List[Node]:
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
        parent = self.a if a_parent else self.b
        child = self.b if a_parent else self.a
        for i, child_member in enumerate(child.get_members()):
            # Find the member of parent group that is parent to this child member
            parent_member = next(filter(lambda p: p.is_parent_to(child_member), parent.get_members()))
            if a_parent:
                members.append(NodePair(parent_member, child_member))
            else:
                members.append(NodePair(child_member, parent_member))
        return members

    def same_level_init(self) -> List[Node]:
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
        common_parents = self.a.get_parents().intersection(self.b.get_parents())
        for a_member in self.a.get_members():
            for b_member in self.b.get_members():
                # Create pair member if a and b members have the same parent members of parent group
                common_parent_members = a_member.get_parents().intersection(b_member.get_parents())
                common_parent_members = list(filter(lambda m: any(map(lambda p: m in p.get_members(), common_parents)), common_parent_members))
                if len(common_parent_members) == len(common_parents):
                    members.append(NodePair(a_member, b_member))
        return members

    def create_members(self) -> List[Node]:
        """
        Init the member list, varying strategies depending on relation of a and b.
        Returns
        -------
        List of OrderedGroupPair, where a element of each OrderedGroupPair object corresponds to a member of the a group
        and b corresponds to a member of the b group.
        """
        # Check if a and b has the same representation
        if self.a.representation() == self.b.representation():
            # a and b have the tha same representation, create a pair member for each member of a mapped to the same
            # element of b
            return list(
                map(lambda m: NodePair(m[0], m[1]), zip(self.a.get_members(), self.b.get_members())))

        # Check relation between a and b
        a_parent = self.a_parent_b()
        if a_parent is not None:
            # a and b have a parent-child relation
            return self.parent_child_init(a_parent)
        else:
            # a and b are unrelated
            return self.same_level_init()

    def representation(self) -> Set[Node]:
        a_parent = self.a_parent_b()
        if a_parent is not None:
            child = self.b if a_parent else self.a
            return child.representation()
        else:
            return self.a.representation().union(self.b.representation())

    def get_members(self) -> List['Node']:
        return self.members

    def reduce_pair(self) -> Node:
        self_repr = self.representation()
        if self_repr == self.a.representation():
            return self.a
        if self_repr == self.b.representation():
            return self.b
        return self

    def map_from(self, other: 'Node') -> List[int]:
        if other == self.a:
            return list(map(lambda m: other.get_members().index(m.a), self.members))
        elif other == self.b:
            return list(map(lambda m: other.get_members().index(m.b), self.members))
        else:
            return NodePair(self, other).map_from(other)

    def __str__(self) -> str:
        return '_'.join(map(str, self.representation()))


class AtomicNode(Node, ABC):
    """
        Class for a single instance group.
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

    def __str__(self):
        return self.name

    def get_members(self) -> List[Node]:
        return self.members

    def is_parent_to(self, other: Node) -> bool:
        return self in other.get_parents()

    def get_parents(self) -> Set[Node]:
        return self.parents

    def representation(self) -> Set['Node']:
        return {self}

    def map_from(self, other: Node) -> List[int]:
        return NodePair(self, other).map_from(other)

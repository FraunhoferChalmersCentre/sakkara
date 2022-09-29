import abc
from typing import Set, TypeVar, Generic, Optional


class Group:
    """
    Interface class
    """

    @abc.abstractmethod
    def is_parent_to(self, other: 'Group') -> bool:
        """
        Test if another group is a child of the group

        Parameters
        ----------
        other : Group
            Other group to test if the group is parent to

        Returns
        -------
        True if the group is parent to other, False otherwise

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_parents(self) -> Set['Group']:
        """
            Get all groups linked as parents of the group
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_representation_groups(self) -> Set['Group']:
        """
            Get the set of groups that represents the information level of the group
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


T = TypeVar('T', bound=Group)


class GroupPair(Group, Generic[T]):
    """
        Class for pair group, to allow for multi-group representations
    """
    def __init__(self, a: T, b: T):
        """

        Parameters
        ----------
        a: First group of pair
        b: Second group of pair
        """
        self.a = a
        self.b = b

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

    def is_parent_to(self, other: Group) -> bool:
        return self.a.is_parent_to(other) and self.b.is_parent_to(other)

    def get_parents(self) -> Set[Group]:
        representation_groups = self.get_representation_groups()
        parents = set().union(*map(lambda g: g.get_parents(), self.get_representation_groups()))

        if len(representation_groups) > 1:
            parents = parents.union(representation_groups)

        return parents

    def get_representation_groups(self) -> Set['Group']:
        a_parent = self.a_parent_b()
        if a_parent is not None:
            child = self.b if a_parent else self.b
            return child.get_representation_groups()
        else:
            return self.a.get_representation_groups().union(self.b.get_representation_groups())

    def __str__(self) -> str:
        return str(self.a) + '_' + str(self.b)


class AtomicGroup(Group, Generic[T]):
    """
        Class for a single instance group.
    """
    def __init__(self, name: str, *parents: 'AtomicGroup[T]'):
        """

        Parameters
        ----------
        name: Name of group
        parents: All (atomic) immediate and non-immediate parents to this atomic group.
        """
        self.name = name
        self.parents = set(parents)

    def __str__(self):
        return self.name

    def is_parent_to(self, other: Group) -> bool:
        return self in other.get_parents()

    def get_parents(self) -> Set[Group]:
        return self.parents

    def get_representation_groups(self) -> Set['Group']:
        return {self}

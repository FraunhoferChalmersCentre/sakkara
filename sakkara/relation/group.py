import abc
from typing import Set, TypeVar, Generic, Tuple, Optional


class Group:

    @abc.abstractmethod
    def is_parent_to(self, child: 'Group') -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_parents(self) -> Set['Group']:
        raise NotImplementedError

    @abc.abstractmethod
    def get_groups(self) -> Set['Group']:
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


T = TypeVar('T', bound=Group)


class GroupPair(Group, Generic[T]):
    def __init__(self, a: T, b: T):
        self.a = a
        self.b = b

    def internal_relation(self) -> Tuple[Optional[T], Optional[T]]:
        """
        Get the relation in the pair
        :return: (parent group, child group) or (None, None) if pair groups are unrelated
        """
        if self.a.is_parent_to(self.b):
            return self.a, self.b
        elif self.b.is_parent_to(self.a):
            return self.b, self.a
        else:
            return None, None

    def get_groups(self) -> Set['Group']:
        return set().union(self.a.get_groups()).union(self.b.get_groups())

    def is_parent_to(self, child: Group) -> bool:
        return self.a.is_parent_to(child) and self.b.is_parent_to(child)

    def get_parents(self) -> Set[Group]:
        parent, child = self.internal_relation()
        if parent is None:
            return self.a.get_parents().intersection(self.b.get_parents()).union(self.get_groups())
        else:
            return parent.get_parents().union(child.get_parents().intersection(self.get_groups()))

    def __str__(self) -> str:
        return str(self.a) + '_' + str(self.b)


class GroupBase(Group, Generic[T]):
    def __init__(self, name: str, *parents: T):
        self.name = name
        self.parents = set(parents)

    def __str__(self):
        return self.name

    def get_groups(self) -> Set['Group']:
        return {self}

    def is_parent_to(self, child: Group) -> bool:
        return self in child.get_parents()

    def get_parents(self) -> Set[Group]:
        return self.parents

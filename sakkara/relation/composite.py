import abc
from abc import ABC
from typing import Set, List

from sakkara.relation.group import Group, GroupPair, GroupBase
from sakkara.relation.ordered import OrderedGroupPair, OrderedGroup


class Composite(Group, ABC):
    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_members(self) -> List[OrderedGroup]:
        raise NotImplementedError


class CompositePair(GroupPair[Composite], Composite, ABC):
    def __init__(self, a: Composite, b: Composite):
        super().__init__(a, b)
        self.members = self.init_members()

    def init_members(self) -> List[OrderedGroupPair]:
        parent, child = self.internal_relation()

        members = list()
        if child is not None:
            for i, child_member in enumerate(child.get_members()):
                parent_member = next(filter(lambda p: p.is_parent_to(child_member), parent.get_members()))
                if parent == self.a:
                    members.append(OrderedGroupPair(i, parent_member, child_member))
                elif parent == self.b:
                    members.append(OrderedGroupPair(i, child_member, parent_member))
        else:
            common_parents = self.a.get_parents().intersection(self.b.get_parents())
            counter = 0
            for ma in self.a.get_members():
                for mb in self.b.get_members():
                    common_parent_members = ma.get_parents().intersection(mb.get_parents())
                    if len(common_parent_members) == len(common_parents):
                        members.append(OrderedGroupPair(counter, ma, mb))
                    counter += 1

        return members

    def __len__(self):
        return len(self.members)

    def get_members(self) -> List[Group]:
        return self.members

    def map_from(self, other: 'Composite') -> List[OrderedGroup]:
        if other == self.a:
            return list(map(lambda m: m.a, self.members))
        elif other == self.b:
            return list(map(lambda m: m.b, self.members))
        else:
            return CompositePair(self, other).map_from(other)


class CompositeBase(GroupBase, Composite, ABC):

    def __init__(self, name: str, members: List[Group], parents: Set[Composite]):
        super().__init__(name, *parents)
        self.members = members

    def get_members(self) -> List[Group]:
        return self.members

    def __len__(self):
        return len(self.members)

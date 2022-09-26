from abc import ABC

from sakkara.relation.group import Group, GroupPair, GroupBase


class OrderedGroup(Group, ABC):
    def __init__(self, index: int):
        self.index = index


class OrderedGroupPair(OrderedGroup, GroupPair[OrderedGroup]):
    def __init__(self, index: int, a: OrderedGroup, b: OrderedGroup):
        OrderedGroup.__init__(self, index)
        GroupPair.__init__(self, a, b)


class OrderedGroupBase(OrderedGroup, GroupBase[OrderedGroup]):
    def __init__(self, index: int, name: str, *parents: OrderedGroup):
        OrderedGroup.__init__(self, index)
        GroupBase.__init__(self, name, *parents)

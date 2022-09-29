from abc import ABC

from sakkara.relation.group import Group, GroupPair, AtomicGroup


class OrderedGroup(Group, ABC):
    """
        Group with an index
    """
    def __init__(self, index: int):
        self.index = index


class OrderedGroupPair(OrderedGroup, GroupPair[OrderedGroup]):
    def __init__(self, index: int, a: OrderedGroup, b: OrderedGroup):
        OrderedGroup.__init__(self, index)
        GroupPair.__init__(self, a, b)


class OrderedAtomicGroup(OrderedGroup, AtomicGroup[OrderedGroup]):
    def __init__(self, index: int, name: str, *parents: AtomicGroup[OrderedGroup]):
        OrderedGroup.__init__(self, index)
        AtomicGroup.__init__(self, name, *parents)

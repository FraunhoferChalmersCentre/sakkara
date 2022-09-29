import abc
from abc import ABC
from typing import Set, List

from sakkara.relation.group import Group, GroupPair, AtomicGroup
from sakkara.relation.ordered import OrderedGroupPair, OrderedGroup, OrderedAtomicGroup


class CompositeGroup(Group, ABC):
    """
        Class for composition of ordered groups
    """

    def __init__(self, members: List[OrderedGroup]):
        self.members = members

    def __len__(self):
        return len(self.members)

    def get_members(self) -> List[OrderedGroup]:
        return self.members

    @abc.abstractmethod
    def map_from(self, other: 'CompositeGroup') -> List[OrderedGroup]:
        """
        Get a mapping from other group to this group. If the other group is unrelated to this group, the mapping will
        be from the other group to a pair group.

        Parameters
        ----------
        other: Composite group to get mapping from

        Returns
        -------
        List of the members of other, in an order to be mapped to the correct member of this group (or a pair group)
        """


class CompositeGroupPair(GroupPair[CompositeGroup], CompositeGroup, ABC):
    """
    Multi-instance class for composite groups
    """

    def __init__(self, a: CompositeGroup, b: CompositeGroup):
        GroupPair.__init__(self, a, b)
        CompositeGroup.__init__(self, self.init_members())

    def parent_child_init(self, a_parent: bool) -> List[OrderedGroupPair]:
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
                members.append(OrderedGroupPair(i, parent_member, child_member))
            else:
                members.append(OrderedGroupPair(i, child_member, parent_member))
        return members

    def same_level_init(self) -> List[OrderedGroupPair]:
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
        counter = 0
        for a_member in self.a.get_members():
            for b_member in self.b.get_members():
                # Create pair member if a and b members have the same parent members of parent group
                common_parent_members = a_member.get_parents().intersection(b_member.get_parents())
                if len(common_parent_members) == len(common_parents):
                    members.append(OrderedGroupPair(counter, a_member, b_member))
                counter += 1
        return members

    def init_members(self) -> List[OrderedGroupPair]:
        """
        Init the member list, varying strategies depending on relation of a and b.
        Returns
        -------
        List of OrderedGroupPair, where a element of each OrderedGroupPair object corresponds to a member of the a group
        and b corresponds to a member of the b group.
        """
        # Check if a and b has the same representation
        if self.a.get_representation_groups() == self.b.get_representation_groups():
            # a and b have the tha same representation, create a pair member for each member of a mapped to the same
            # element of b
            return list(
                map(lambda m: OrderedGroupPair(m[0].index, m[0], m[1]), zip(self.a.get_members(), self.b.get_members())))

        # Check relation between a and b
        a_parent = self.a_parent_b()
        if a_parent is not None:
            # a and b have a parent-child relation
            return self.parent_child_init(a_parent)
        else:
            # a and b are unrelated
            return self.same_level_init()

    def map_from(self, other: 'CompositeGroup') -> List[OrderedGroup]:
        if other == self.a:
            return list(map(lambda m: m.a, self.members))
        elif other == self.b:
            return list(map(lambda m: m.b, self.members))
        else:
            return CompositeGroupPair(self, other).map_from(other)


class CompositeAtomicGroup(AtomicGroup, CompositeGroup, ABC):
    """
    Single instance group for composite groups
    """

    def __init__(self, name: str, members: List[OrderedAtomicGroup], parents: Set['CompositeAtomicGroup']):
        AtomicGroup.__init__(self, name, *parents)
        CompositeGroup.__init__(self, members)

    def map_from(self, other: 'CompositeGroup') -> List[OrderedGroup]:
        return CompositeGroupPair(self, other).map_from(other)

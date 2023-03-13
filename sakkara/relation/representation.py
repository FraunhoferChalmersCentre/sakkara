from typing import Union, Tuple, Any

import numpy as np
import numpy.typing as npt

from sakkara.relation.group import Group


class Representation:
    """
    Class for a tuple of groups, reduced to its minimal representation. E.g., a parent can be represented by its child,
    hence when creating a `Representation` object of the two only the child will be added to the tuple.

    :param \*groups: :class:`Group` objects to include in representation
    """

    def __init__(self, *groups: 'Group'):
        self.groups = []
        for g in groups:
            self.add_group(g)

    def add_group(self, group: Group) -> None:
        """
        Add group to the representation. It will not be added if there already is a twin or a child in
        the representation. If there is a parent to the group in the representation, the group will be added and the
        parent will be removed.
        """
        # Check that group is not already represented by twin or child
        representatives = group.twins.union(group.children)
        if any(map(lambda t: t in representatives, self.groups)):
            return

        # Add group to representation
        self.groups.append(group)

        # Check if newly added group's parents are represented in group
        for parent in group.parents:
            # Remove if they are included (due to redundancy)
            if parent in self.groups:
                self.groups.remove(parent)

    def merge(self, other: 'Representation') -> 'Representation':
        """
        Create a new `Representation` by adding upp groups from this object and from another.

        :param other: Other `Representation` to merge with

        :return: Merged representation. Note that all groups of the merged has either a twin or a parent group in at
            least one of the two input representations.
        """
        merged = Representation(*self.groups)
        for group in other.groups:
            merged.add_group(group)

        return merged

    def is_mappable_to(self, target: 'Representation') -> bool:
        """
        Check if representation is mappable to another representation, i.e., groups of the other representation have
        twins or parents among the groups of this representation.

        :return: `True` is self is mappable to target
        """
        target_represents = set()
        for target_group in target.groups:
            target_represents = target_represents.union(target_group.twins).union(target_group.parents)
        return all(map(lambda t: t in target_represents, self.groups))

    def get_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of a variable corresponding to a component with this representation
        """
        return tuple(map(len, self.groups))

    def __eq__(self, other: 'Representation'):
        return len(self.groups) == len(other.groups) and all(map(lambda x, y: x in y.twins, self.groups, other.groups))

    def get_member_array(self, group: Group) -> npt.NDArray[Any]:
        """
        Get an array shaped as a variable with this representation, where each value correspond to the member of the
        i:th group.
        """
        group_index = self.groups.index(group)
        # Get shape of representation, but target_group excluded
        other_shape = self.get_shape()[:group_index] + self.get_shape()[group_index + 1:]
        # Create flat array or target_group members repeated, as if i == 0
        member_array = np.repeat(group.members, np.prod(other_shape))
        # Reshape target_members to have the same order as target's representation
        member_array = member_array.reshape((len(group),) + other_shape)
        # Swap i:th and 0:th axes to handle the case when i != 0
        member_array = np.swapaxes(member_array, 0, group_index)

        return member_array

    def map_to(self, target: 'Representation') -> Union[Tuple[npt.NDArray[int], ...], slice]:
        """
        Create indexing for a variable of this representation to another target representation.

        :param target: The representation that corresponds to the shape the variable should be in. Each group of self
            representation must have a twin or a child in the target representation.

        :return: Tuple of index arrays, or slice object, to be used for indexing on array object. For more information
            about the format of this tuple, see https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing.

        """
        if self == target:
            return slice(None)
        if not self.is_mappable_to(target):
            raise ValueError(
                'Representation cannot be mapped to target. Instead do mapping to a representation of this merged with target.')

        mapping = []

        for group in self.groups:
            for i, target_group in enumerate(target.groups):
                # Match any group of target representation that is either child or twin to this
                representatives = {group}.union(group.children).union(group.twins)

                if target_group in representatives:
                    # Get the array of members of target group, shaped according to target representation
                    target_members = target.get_member_array(target_group)
                    # Fetch mapping from target_members into members of group
                    this_group_members = target_group.mapping.loc[target_members.ravel(), group.name].values
                    # Append mapping, reshaped to target representation's shape
                    mapping.append(this_group_members.reshape(target.get_shape()))

                    # Finding more representatives not needed
                    break

        return tuple(mapping)

    def get_members(self) -> Tuple[npt.NDArray[Any], ...]:
        """
        Get array of members corresponding to the indices of a variable with the representation

        :return: One array per group of representation filled with corresponding member names, so that the i:th array
            correspond to the i:th group of the representation
        """
        return tuple(map(lambda group: self.get_member_array(group), self.groups))

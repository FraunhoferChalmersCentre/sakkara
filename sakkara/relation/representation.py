import abc
from abc import ABC
from typing import Tuple, Any, List, Dict, Optional

import numpy as np
import numpy.typing as npt

from sakkara.relation.group import Group


class Representation:
    """
    Abstract class for a representation, i.e., class that transforms variables and other elements between different
    shapes.
    """

    @abc.abstractmethod
    def get_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of a variable corresponding to a component with this representation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_groups(self) -> List[Group]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_member_array(self, group: Group) -> npt.NDArray[Any]:
        """
        Get an array shaped as a variable with this representation, where each value correspond to the member of the
        input group.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_members(self) -> Tuple[npt.NDArray, ...]:
        """
        Get member arrays for all groups of this representation. For documentation of a member array, see
        :meth:`Representation.get_member_array`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_member_tuples(self) -> List[Tuple[Any, ...]]:
        """
        Get a (flat) list of all member tuples of this representation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def map(self, element: Any, target_representation: 'Representation') -> Any:
        """
        Map/transform an element (variable or equivalent) corresponding to this representation to a target
        representation.

        :param element: The object/variable to transform.

        :param target_representation: The representation that corresponds to the shape the variable should be in.

        :return: The element but transformed to target representation.
        """
        raise NotImplementedError


def one_to_one_mapping(group: Group, target: Representation) -> Optional[Group]:
    # Get mapping if g is parent or twin to any group of target representation
    for target_group in target.get_groups():
        if group in target_group.twins.union(target_group.parents):
            return target_group

    return None


def subset_prod(remain: int, candidates: npt.NDArray[Group]) -> Tuple[bool, npt.NDArray[Group]]:
    """
    Solve the subset product, e.g., select a subset of candidate values whose product equals remain
    """
    for ix, c in enumerate(candidates):
        if remain % len(c) == 0 and len(c) <= remain:
            # We have found candidate
            break
    else:
        # No candidate can be in solution set
        return False, []

    if remain == len(c):
        # We have found a single candidate satisfying the conditions
        return True, [c]

    # Test whether c can be combined with other candidates to make up for remain
    ok, arr = subset_prod(remain // len(c), candidates[ix + 1:])

    if ok:
        # c could successfully be multiplied together with subset of subsequent candidates to get remain
        arr.append(c)
        return ok, arr
    else:
        # c can not be in solution, start over with reduced list of candidates
        return subset_prod(remain, candidates[ix + 1:])


def one_to_combination_mapping(group: Group, target: Representation) -> Tuple[Group, ...]:
    """
    Find a combination of (parent) groups of target representation that corresponds to the same information as in group
    """

    # Check if any combination of parents to g is equivalent representation
    def is_candidate(target_group):
        return group in target_group.children and not len(group) % len(target_group)

    all_target_groups = np.array(target.get_groups())

    # Filter candidate groups that might form the representation
    candidate_groups = all_target_groups[np.vectorize(is_candidate)(target.get_groups())]

    if len(group) % int(np.prod([len(t) for t in candidate_groups])) != 0:
        raise ValueError('Group is not mappable to target representation')

    # Find a subset of candidates so that the product of their sizes equal the size of g
    _, mapped_groups = subset_prod(len(group), candidate_groups)

    return tuple(mapped_groups)


class UnrepeatableRepresentation(Representation, ABC):
    """
    Simple representation for elements that always have a single value and that can not be repeated.
    """

    def get_shape(self) -> Tuple[int, ...]:
        return 1,

    def get_groups(self) -> List[Group]:
        return []

    def get_member_array(self, group: Group) -> npt.NDArray[Any]:
        raise ValueError('This representation does not hold any groups')

    def get_members(self) -> Tuple[npt.NDArray, ...]:
        raise ValueError('This representation does not hold any groups')

    def get_member_tuples(self) -> List[Tuple[Any, ...]]:
        raise ValueError('This representation does not hold any groups')

    def map(self, element: object, target_representation: 'Representation'):
        return element

    def __eq__(self, other):
        return isinstance(other, UnrepeatableRepresentation)


class TensorRepresentation(Representation, ABC):
    """
    Class for representation shapes corresponding to sequneces of groups.

    :param \*groups: :class:`Group` objects to include in representation
    """

    def __init__(self, *groups: 'Group'):
        self.groups = []
        for g in groups:
            self.add_group(g)

    @abc.abstractmethod
    def add_group(self, group: Group):
        """
        Add group to the representation.
        """
        raise NotImplementedError

    def get_groups(self) -> List[Group]:
        return self.groups

    def get_group_mapping(self, target: 'Representation') -> Dict[Group, Tuple[Group, ...]]:
        """
        Check if representation is mappable to another representation, i.e., groups of the other representation have
        twins or parents among the groups of this representation.

        :return: `True` is self is mappable to target
        """

        mapping_dict = dict()

        for group in self.groups:
            mapped = one_to_one_mapping(group, target)
            if mapped is None:
                mapped = one_to_combination_mapping(group, target)

                if len(mapped) == 0:
                    raise ValueError('Group could not be mapped to any other group or combination of groups')
            else:
                mapped = (mapped,)

            mapping_dict[group] = mapped

        return mapping_dict

    def get_shape(self) -> Tuple[int, ...]:
        return tuple(map(len, self.groups))

    def __eq__(self, other: 'Representation'):
        return len(self.groups) == len(other.get_groups()) and all(
            map(lambda x, y: x in y.twins, self.groups, other.get_groups()))

    def get_member_array(self, group: Group) -> npt.NDArray[Any]:
        if group not in self.groups:
            raise ValueError('Group is not found in this representation')

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

    def map(self, element: Any, target: Representation) -> Any:
        if self == target:
            return element

        mapping_dict = self.get_group_mapping(target)

        mapping = []

        for group in self.groups:
            target_groups = mapping_dict[group]

            if len(target_groups) == 1:
                # One to one mapping means that mapped group is a twin or child to group
                mapped_group = target_groups[0]
                # Get the array of members of target group, shaped according to target representation but ravelled
                target_members = target.get_member_array(mapped_group).ravel()
                mapping_df = mapped_group.mapping

            else:
                # One to combination mapping means that mapped groups are all parent to the group
                mapping_df = group.mapping.copy()
                for m in target_groups:
                    mapping_df[str(m)] = m.mapping.iloc[mapping_df[str(m)].values].index.values
                mapping_df = mapping_df.set_index(list(map(str, target_groups)))
                target_members = [tuple(x) for x in zip(*[target.get_member_array(m).ravel() for m in target_groups])]

            # Fetch mapping from target_members into members of group
            group_indices = np.array([mapping_df.loc[t, group.name] for t in target_members])
            # Append mapping, reshaped to target representation's shape
            mapping.append(group_indices.reshape(target.get_shape()))

        return element[tuple(mapping)]

    def get_members(self) -> Tuple[npt.NDArray, ...]:
        if len(self.groups) == 0:
            raise ValueError('This representation does not hold any groups')
        return tuple(map(lambda group: self.get_member_array(group), self.groups))

    def get_member_tuples(self) -> List[Tuple[Any, ...]]:
        if len(self.groups) == 0:
            raise ValueError('This representation does not hold any groups')
        if len(self.groups) == 1:
            return [(x,) for x in np.ravel(self.get_members()[0])]
        else:
            return [(x,) for x in zip(map(np.ravel, self.get_members()))]


class MinimalTensorRepresentation(TensorRepresentation, ABC):
    """
    Tensor representation that always keeps the minimal set of groups needed. E.g., groups will not be added to this
    representation if there already is a twin or a child in the representation. When adding a group that has a
    parent in the representation, the group will be added and the parent will be removed. This class is currently
    the only implementation of TensorRepresentation.
    """

    def add_group(self, group: Group) -> None:
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

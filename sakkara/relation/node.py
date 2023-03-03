import abc
import time
from functools import cache
from typing import Set, Any, Tuple, Callable, Union

import numpy as np
import numpy.typing as npt


class Node:
    """
    Abstract class for relational nodes
    """

    @abc.abstractmethod
    def get_key(self) -> Tuple[Any, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def is_parent_to(self, other: 'Node') -> bool:
        """
        Test if another node is a child of the node

        Parameters
        ----------
        other : Node
            Other node to test

        Returns
        -------
        True if the node is parent to other, False otherwise

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_members(self) -> npt.NDArray['Node']:
        """
        Get all nodes linked as members of the node
        """

    @abc.abstractmethod
    def get_parents(self) -> Set['Node']:
        """
            Get all nodes linked as parents of the node
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_children(self) -> Set['Node']:
        raise NotImplementedError

    @abc.abstractmethod
    def get_twins(self) -> Set['Node']:
        raise NotImplementedError

    @abc.abstractmethod
    def is_twin_to(self, other: 'Node') -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def representation(self):
        """
            Returns the set of nodes that represents the information level of the group
        """
        raise NotImplementedError

    def get_mapping(self, other: 'Node', match_fct: Callable) -> Tuple[npt.NDArray[int], ...]:
        raveled_mapping = np.empty(self.get_members().ravel().shape, dtype=int)
        raveled_self_members = self.get_members().ravel()
        raveled_other_members = other.get_members().ravel()

        for raveled_self_member_index, self_member in enumerate(raveled_self_members):
            raveled_index = np.argmax(match_fct(self_member, raveled_other_members))
            raveled_mapping[raveled_self_member_index] = raveled_index

        return tuple(map(lambda x: x.reshape(self.get_members().shape).tolist(),
                         np.unravel_index(raveled_mapping, other.get_members().shape)))

    def map_to(self, other: 'Node') -> Union[Tuple[npt.NDArray[int], ...], slice]:
        """
        Get a mapping from other group to this group.

        Parameters
        ----------
        other: Group to get mapping from

        Returns
        -------
        List of the member nodes of other, in an order to be mapped to the correct member of this group (or a pair group)
        """
        if self == other:
            return slice(None)
        if self.representation() == other.representation() or other.is_twin_to(self) or self.is_twin_to(other):
            return self.get_mapping(other,
                                    np.vectorize(lambda om, sm: om.representation() == sm.representation()))

            if self.is_parent_to(other):
                raise ValueError('Mapping from parent to child is not applicable.')

        if not other.is_parent_to(self):
            raise ValueError('Mapping between unrelated nodes is not applicable.')

        # Other node is parent to self
        return self.get_mapping(other, np.vectorize(
            lambda sm, om: len(om.representation().get_nodes().intersection(sm.get_parents()))))

    def member_subset(self, subset: npt.NDArray['Node']) -> Tuple[npt.NDArray[int], ...]:
        vec_repr = np.vectorize(lambda a, b: a.representation() == b.representation())
        representation_match = np.vectorize(lambda node: np.any(vec_repr(node, subset)))

        vec_p2c = np.vectorize(lambda parent, child: parent.is_parent_to(child))
        parent_match = np.vectorize(lambda node: np.any(vec_p2c(subset, node)))

        member_node_match = np.vectorize(
            lambda member: np.logical_or(representation_match(member), parent_match(member)))

        raveled_member_indices = np.argwhere(member_node_match(self.get_members().ravel())).squeeze()

        return np.unravel_index(raveled_member_indices, shape=self.get_members().shape)

    @abc.abstractmethod
    def reduced_repr(self) -> 'Node':
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

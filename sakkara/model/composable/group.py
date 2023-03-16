from abc import ABC
from typing import Tuple, Union, Any, Dict

import pymc as pm
from pytensor import tensor as pt
import numpy as np
from numpy import typing as npt

from sakkara.model.fixed.data import DataComponent
from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import Composable, T
from sakkara.relation.groupset import GroupSet
from sakkara.relation.representation import MinimalTensorRepresentation


class GroupComponent(Composable[Tuple[str, ...], T], ABC):
    """
    Class for specifying components for each member of a group individually

    :param group: Group of which the component is defined for.
    :param membercomponents: Dictionary with key indicating member (corresponding to DataFrame value) and value its
            corresponding value (ModelComponent or other)
    :param name: Name of the corresponding variable to register in PyMC.
    """

    def __init__(self, group: Union[str, Tuple[str, ...]], name: str = None, membercomponents: Dict[Any, Any] = None):
        super().__init__(name, group, dict())
        if membercomponents is not None:
            for k, v in membercomponents.items():
                self.add(k, v)

    def __getitem__(self, item: Any):
        return self.subcomponents[item if isinstance(item, tuple) else (item,)]

    def add(self, member: Any, component: Union[float, npt.NDArray, ModelComponent]) -> None:
        """
        Add component for a member to the GroupComponent

        :param member: Key for the group member, corresponding to Dataframe value.
        :param component: Value or component of the given member.
        """
        if not isinstance(component, ModelComponent):
            component = DataComponent(component, 'global')
        key = member if isinstance(member, tuple) else (member,)
        self.subcomponents[key] = component

    def prebuild(self, groupset: GroupSet) -> None:
        self.build_components(groupset)

    def build_representation(self, groupset: GroupSet):
        self.base_representation = MinimalTensorRepresentation()
        self.components_representation = MinimalTensorRepresentation(groupset['global'])

        for g in self.group:
            self.base_representation.add_group(groupset[g])

        for component in self.subcomponents.values():
            self.components_representation = MinimalTensorRepresentation(*component.representation.get_groups(),
                                                                         *self.components_representation.get_groups())

        self.representation = MinimalTensorRepresentation(*self.base_representation.get_groups(),
                                                          *self.components_representation.get_groups())

    def build_variable(self) -> None:
        member_tuples = self.base_representation.get_member_tuples()

        if not all(m in self.subcomponents for m in member_tuples):
            raise ValueError('All member of group component must be specified.')

        # Get the member array for each group, re-order with mapping
        base_members = self.base_representation.get_members()
        base_members_reordered = tuple(
            map(lambda x: self.base_representation.map(x, self.representation), base_members))

        # Create array for adding
        member_tensor = np.empty(len(member_tuples), dtype=object)

        for i, member_tuple in enumerate(member_tuples):
            # Get the component
            component = self.subcomponents[member_tuple]
            # Get the variable, re-order to the representation and flatten
            member_variable = component.representation.map(component.variable, self.representation)

            # Mask all the entries of the group variable (with self.representation) corresponding to this member
            mask = np.ones(np.prod(self.representation.get_shape()), dtype=bool)
            for j, m in enumerate(member_tuple):
                mask *= base_members_reordered[j].ravel() == m

            member_tensor[i] = member_variable[mask]

        # Stack the member-wise variables into the group variable
        # NOTE! We assume that self.base_representation correspond to the first groups of self.representation
        # This might not be the case if a child of a group in self.base_representation is included in
        # self.components_representation (Hence, this is not supported)
        full_tensor = pt.stack(member_tensor.tolist()).ravel().reshape(self.representation.get_shape())

        # Create the group variable, wrapped with Deterministic
        self.variable = pm.Deterministic(name=self.name, var=full_tensor, dims=self.dims())

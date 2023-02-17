from abc import ABC
from typing import Tuple, Union, Optional, Collection, Any, Dict

import pymc as pm
from pytensor import tensor as pt
from numpy import typing as npt

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import Composable
from sakkara.model.fixed.base import FixedValueComponent
from sakkara.model.composable.group.groupwrapper import GroupWrapper
from sakkara.model.composable.group.memberwrapper import MemberWrapper
from sakkara.relation.groupset import GroupSet


class GroupComponent(Composable[Tuple[str, ...], MemberWrapper], ABC):
    """
    Class for specifying components for each member of a group individually
    """

    def __init__(self, group: Union[str, Tuple[str, ...]], members: Optional[Collection[Any]] = None,
                 name: str = None, membercomponents: Dict[Any, Any] = None):
        """

        Parameters
        ----------

        group: Group of which the component is defined for.
        membercomponents: Dictionary with key indicating member (corresponding to DataFrame value) and value its
            corresponding value (ModelComponent or other)
        name: Name of the corresponding variable to register in PyMC.
        """
        super().__init__(name, group, members, {})
        if membercomponents is not None:
            for k, v in membercomponents.items():
                self.add(k, v)

    def __getitem__(self, item: Any):
        return self.subcomponents[item if isinstance(item, tuple) else (item,)]['component']['component']

    def add(self, member: Any, component: Union[float, npt.NDArray, ModelComponent]) -> None:
        """
        Add component for a member to the GroupComponent

        Parameters
        ----------
        member: Key for the group member, corresponding to Dataframe value.
        component: Value or component of the given member.
        """
        if not isinstance(component, ModelComponent):
            component = FixedValueComponent(component)
        key = member if isinstance(member, tuple) else (member,)
        self.subcomponents[key] = MemberWrapper(GroupWrapper(component, name=member),
                                                group=self.group,
                                                members=[member],
                                                name=member)

    def set_group(self):
        component_groups = set()
        for member_comp in self.subcomponents.values():
            component_groups = component_groups.union(member_comp['component']['component'].retrieve_groups())

        for member_comp in self.subcomponents.values():
            member_comp['component'].group = tuple(component_groups)

    def prebuild(self, groupset: GroupSet) -> None:
        self.set_group()
        self.build_components(groupset)

    def get_built_components(self) -> Dict[Tuple[str, ...], pt.Variable]:
        return {k: v.variable for k, v in self.subcomponents.items()}

    def build_variable(self) -> None:
        undefined_members = list(
            m for m in self.group_node.get_members() if m.get_key() not in list(self.get_built_components().keys()))
        if 0 < len(undefined_members):
            raise KeyError('All members of group must be defined to build concat variable')

        pt_tensor = pt.stack([self.get_built_components()[m.get_key()] for m in self.group_node.get_members().ravel()])
        self.variable = pm.Deterministic(name=self.name,
                                         var=pt_tensor.reshape(self.node.get_members().shape),
                                         dims=self.dims())

from abc import ABC
from typing import Tuple, Union, Optional, Collection, Any, Dict

import pymc as pm
from aesara import tensor as at
from numpy import typing as npt

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import Composable
from sakkara.model.fixed.base import FixedComponent
from sakkara.model.composable.group.groupwrapper import GroupWrapper
from sakkara.model.composable.group.memberwrapper import MemberWrapper
from sakkara.relation.groupset import GroupSet


class GroupComponent(Composable[Tuple[str, ...], MemberWrapper], ABC):
    """
    Class for specifying components for each member of a group individually
    """

    def __init__(self, columns: Union[str, Tuple[str, ...]], members: Optional[Collection[Any]] = None,
                 name: str = None, components: Dict[Any, Any] = None):
        """

        Parameters
        ----------
        components: Dictionary with key indicating member and value its corresponding variable
        columns: The column of the group in the dataframe.
        name: Name of the concatenated variable.
        """
        super().__init__(name, columns, members, {})
        if components is not None:
            for k, v in components.items():
                self.add(k, v)

    def __getitem__(self, item: Any):
        return self.components[item if isinstance(item, tuple) else (item,)]['component']['component']

    def add(self, member: Any, component: Union[float, npt.NDArray, ModelComponent]) -> None:
        if not isinstance(component, ModelComponent):
            component = FixedComponent(component)
        key = member if isinstance(member, tuple) else (member,)
        self.components[key] = MemberWrapper(GroupWrapper(component, name=member),
                                             columns=self.columns,
                                             members=[member],
                                             name=member)

    def set_columns(self):
        component_cols = set()
        for member_comp in self.components.values():
            component_cols = component_cols.union(member_comp['component']['component'].retrieve_columns())

        for member_comp in self.components.values():
            member_comp['component'].columns = tuple(component_cols)

    def prebuild(self, groupset: GroupSet) -> None:
        self.set_columns()
        self.build_components(groupset)

    def get_built_components(self) -> Dict[Tuple[str, ...], at.Variable]:
        return {k: v.variable for k, v in self.components.items()}

    def build_variable(self) -> None:
        undefined_members = list(
            m for m in self.column_node.get_members() if m.get_key() not in list(self.get_built_components().keys()))
        if 0 < len(undefined_members):
            raise KeyError('All members of group must be defined to build concat variable')

        at_tensor = at.stack([self.get_built_components()[m.get_key()] for m in self.column_node.get_members().ravel()])
        self.variable = pm.Deterministic(name=self.name,
                                         var=at_tensor.reshape(self.node.get_members().shape),
                                         dims=self.dims())


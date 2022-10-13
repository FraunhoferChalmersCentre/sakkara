from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from sakkara.relation.node import Node, AtomicNode, CellNode


@dataclass(frozen=True)
class GroupSet:
    """
    Set of group nodes used for PyMC model creation
    """
    groups: Dict[str, AtomicNode]

    def __getitem__(self, item):
        return self.groups[item]

    def coords(self) -> Dict[str, np.ndarray]:
        coords_dict = {}
        for k, v in self.groups.items():
            names = np.empty(len(v), dtype=object)
            for i, m in enumerate(v.get_members()):
                names[i] = str(m)
            coords_dict[k] = names
        return coords_dict


def get_parent_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a matrix of parent mappings between columns in a dataframe
    Parameters
    ----------
    df: Dataframe (with C columns) to base parent mappings of.

    Returns
    -------
    Dataframe of size CxC with parent mappings. Rows are sorted from highest to lowest hierarchy level. Element (i,j)
    True means that j is a parent to i.
    """
    groups = list(df.columns)
    counts_df = pd.DataFrame(index=groups, columns=groups, data=False)
    for i in range(len(groups)):
        counts_df.loc[groups[i], groups[:i] + groups[i + 1:]] = df.groupby(groups[i]).nunique().max() == 1

    counts_df['rank'] = counts_df.sum(axis=1)
    counts_df.sort_values(by='rank', inplace=True, ascending=True)
    return counts_df.loc[:, df.columns]


def fill_member_parents(group_name: str, df: pd.DataFrame, parents: Set[Node],
                        member_parents: Dict[str, Set[Node]]) -> None:
    """
    Add member parents of a group
    Parameters
    ----------
    group_name: Name of the group
    df: Original dataframe
    parents: All composite groups that is a parent to the composite group with group_name
    member_parents: Dictionary with mapping from parent member
    """
    if len(parents) > 0:
        mappings = df.groupby(group_name).first()

        for parent in parents:
            for parent_member in parent.get_members():
                for mn in mappings.index[mappings[str(parent)] == str(parent_member)]:
                    member_parents[str(mn)].add(parent_member)


def add_column_node(name: str, parent_names: List[str], df: pd.DataFrame, groups: Dict[str, AtomicNode]) -> None:
    """
    Init a single composite group.

    Parameters
    ----------
    name: Name of the composite group
    parent_names: Names of the parents
    df: Orginal dataframe
    groups: Dict of names mapped to the pre-created parents

    Returns
    -------

    """
    member_names = list(map(str, df[name].unique()))

    parents = set(map(lambda p: groups[p], parent_names))
    selection_df = df.loc[:, [name] + list(map(str, parents))]

    member_parents = {str(k): set() for k in member_names}
    fill_member_parents(name, selection_df, parents, member_parents)

    members = []
    for member_name in member_names:
        new_member = CellNode(member_name, member_parents[member_name])
        # Add new member as child to member of parent
        for member_parent in member_parents[member_name]:
            member_parent.add_child(new_member)
        members.append(new_member)

    # Create group node
    new_node = AtomicNode(name, members, parents)
    for parent in parents:
        parent.add_child(new_node)

    groups[name] = new_node


def init(df: pd.DataFrame) -> GroupSet:
    """
    Init a group set from a dataframe
    :param df: DataFrame containing only the group columns
    :return: GroupSet created from the input DataFrame
    """
    tmp_df = df.copy().astype(str)

    groups = dict()

    parent_df = get_parent_df(df)
    for i, (group_name, is_parent) in enumerate(parent_df.iterrows()):
        add_column_node(str(group_name), is_parent.index[is_parent], tmp_df, groups)

    return GroupSet(groups)

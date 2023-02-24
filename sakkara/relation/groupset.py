from dataclasses import dataclass
from typing import Dict, List, Set, Any

import numpy as np
import pandas as pd

from sakkara.relation.cellnode import CellNode
from sakkara.relation.atomicnode import AtomicNode


@dataclass(frozen=True)
class GroupSet:
    """
    Set of group nodes used for PyMC model creation
    """
    groups: Dict[str, AtomicNode]

    def __getitem__(self, item: str):
        return self.groups[item]

    def coords(self) -> Dict[str, np.ndarray]:
        coords_dict = {}
        for k, v in self.groups.items():
            names = np.empty(len(v.get_members()), dtype=object)
            for i, m in enumerate(v.get_members()):
                names[i] = m.get_key()[0]
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
    n_uniques = df.nunique(axis=0)
    counts_df = pd.DataFrame(index=groups, columns=groups, data=False)
    for i in range(len(groups)):
        counts_df.loc[groups[i], groups[:i] + groups[i + 1:]] = np.logical_and(
            df.groupby(groups[i]).nunique().max() == 1,
            n_uniques[groups[i]] > n_uniques[groups[:i] + groups[i + 1:]])

    counts_df['rank'] = counts_df.sum(axis=1)
    counts_df.sort_values(by='rank', inplace=True, ascending=True)
    return counts_df.loc[:, df.columns]


def get_member_parents(child_key: str, df: pd.DataFrame, parents: Set[AtomicNode]) -> Dict[Any, Set[CellNode]]:
    """
    Add member parents of the (child) group
    Parameters
    ----------
    child_key: Name of the (child) group
    df: Original dataframe
    parents: All composite groups that is a parent to the composite group with group_name
    """
    member_parents = {k: set() for k in list(df[child_key])}
    if len(parents) > 0:
        mappings = df.groupby(child_key).first().reset_index()

        for parent in parents:
            parent_mappings = mappings[parent.get_key()[0]]
            for parent_member in parent.get_members():
                for matched_child_key in mappings.loc[parent_mappings == parent_member.get_key()[0], child_key]:
                    if matched_child_key not in member_parents:
                        member_parents[matched_child_key] = set()
                    member_parents[matched_child_key].add(parent_member)

    return member_parents


def get_twin_df(df):
    groups = list(df.columns)
    counts_df = pd.DataFrame(index=groups, columns=groups, data=False)
    for i in range(len(groups)):
        counts_df.loc[groups[i], groups[:i] + groups[i + 1:]] = df.groupby(groups[i]).nunique().max() == 1
    return np.logical_and(counts_df, counts_df.T)


def create_column_node(column: str, parent_names: List[str], df: pd.DataFrame,
                       groups: Dict[Any, AtomicNode]) -> AtomicNode:
    """
    Init a single composite group.

    Parameters
    ----------
    column: Name of the composite group
    parent_names: Names of the parents
    df: Orginal dataframe
    groups: Dict of names mapped to the pre-created parents

    Returns
    -------

    """
    parents = set(groups[pn] for pn in parent_names)
    selection_df = df.loc[:, [column] + parent_names]

    member_parents = get_member_parents(column, selection_df, parents)

    members = np.empty(len(member_parents), dtype=object)
    for i, member_name in enumerate(df[column].unique()):
        new_member = CellNode(member_name, member_parents[member_name])
        # Add new member as child to member of parent
        for member_parent in member_parents[member_name]:
            member_parent.add_child(new_member)
        members[i] = new_member

    # Create group node
    new_node = AtomicNode(column, members, parents)
    for parent in parents:
        parent.add_child(new_node)

    return new_node


def init(df: pd.DataFrame) -> GroupSet:
    """
    Init a group set from a dataframe
    :param df: DataFrame containing only the group columns
    :return: GroupSet created from the input DataFrame
    """
    tmp_df = df.copy()

    groups = dict()

    parent_df = get_parent_df(df)
    for i, (group_name, is_parent) in enumerate(parent_df.iterrows()):
        groups[str(group_name)] = create_column_node(str(group_name), list(is_parent.index[is_parent]), tmp_df, groups)

    twin_df = get_twin_df(df)
    for group_name, is_twin in twin_df.iterrows():
        for name, twin_i in is_twin[is_twin].items():
            groups[str(group_name)].add_twin(groups[str(name)])
            for member, twin_member in zip(groups[group_name].get_members(), groups[str(name)].get_members()):
                member.add_twin(twin_member)

    return GroupSet(groups)

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from sakkara.relation.compositegroup import CompositeAtomicGroup, CompositeGroup
from sakkara.relation.ordered import OrderedAtomicGroup, OrderedGroup


@dataclass(frozen=True)
class GroupSet:
    """
    Set of composite groups
    """
    groups: Dict[str, CompositeGroup]

    def __getitem__(self, item):
        return self.groups[item]

    def coords(self) -> Dict[str, np.ndarray]:
        coords_dict = {}
        for k, v in self.groups.items():
            names = np.empty(len(v), dtype=object)
            for m in v.get_members():
                names[m.index] = str(m)
            coords_dict[k] = names
        return coords_dict


def get_parent_df(df) -> pd.DataFrame:
    """
    Create a matrix of parent mappings between columns in a dataframe
    Parameters
    ----------
    df: Dataframe (with C columns) to base parent mappings of.

    Returns
    -------
    Dataframe of size CxC with parent mappings. Rows are sorted with successively lower hierarchy levels.
    """
    groups = list(df.columns)
    counts_df = pd.DataFrame(index=groups, columns=groups, data=False)
    for i in range(len(groups)):
        counts_df.loc[groups[i], groups[:i] + groups[i + 1:]] = df.groupby(groups[i]).nunique().max() == 1

    counts_df['rank'] = counts_df.sum(axis=1)
    counts_df.sort_values(by='rank', inplace=True, ascending=True)
    return counts_df.loc[:, df.columns]


def fill_member_parents(group_name: str, df: pd.DataFrame, parents: Set[CompositeGroup],
                        member_parents: Dict[str, Set[OrderedGroup]]) -> None:
    """
    Add member parents of a composite group
    Parameters
    ----------
    group_name: Name of the composite group
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


def init_composite(name: str, parent_names: List[str], df: pd.DataFrame,
                   groups: Dict[str, CompositeAtomicGroup]) -> CompositeAtomicGroup:
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
    fill_member_parents(str(name), selection_df, parents, member_parents)

    members = list(map(lambda x: OrderedAtomicGroup(x[0], x[1], *member_parents[x[1]]), enumerate(member_names)))
    return CompositeAtomicGroup(str(name), members, parents)


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
        groups[str(group_name)] = init_composite(str(group_name), is_parent.index[is_parent], tmp_df, groups)

    return GroupSet(groups)

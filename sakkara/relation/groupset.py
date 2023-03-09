from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from sakkara.relation.group import Group


@dataclass(frozen=True)
class GroupSet:
    """
    Set of group nodes used for PyMC model creation
    """
    groups: Dict[str, Group]

    def __getitem__(self, item: str):
        return self.groups[item]

    def coords(self) -> Dict[str, np.ndarray]:
        coords_dict = {}
        for k, v in self.groups.items():
            coords_dict[k] = v.members
        return coords_dict


def get_parent_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a matrix of parent mappings between columns in a dataframe

    :param df: Dataframe (with C columns) to base parent mappings of.

    :return Dataframe of size CxC with parent mappings. Rows are sorted from highest to lowest hierarchy level.
    Element (i,j) True means that j is a parent to i.
    """
    groups = list(df.columns)
    n_uniques = df.nunique(axis=0)
    counts_df = pd.DataFrame(index=groups, columns=groups, data=False)
    for i in range(len(groups)):
        # Parent to child is when child value always gives parent value on same row, and there are more unique values of
        # child
        counts_df.loc[groups[i], groups[:i] + groups[i + 1:]] = np.logical_and(
            df.groupby(groups[i]).nunique().max() == 1,
            n_uniques[groups[i]] > n_uniques[groups[:i] + groups[i + 1:]])

    # Sort counts_df from the lowest child in the first row to the highest parent in the last row
    counts_df['rank'] = counts_df.sum(axis=1)
    counts_df.sort_values(by='rank', inplace=True, ascending=True)
    return counts_df.loc[:, df.columns]


def get_twin_df(df):
    """
    Create a matrix of twin relations between columns in a dataframe

    :param df: Dataframe (with C columns) to base twin mappings of.

    :return Dataframe of size CxC with twin mappings.
    """
    groups = list(df.columns)
    counts_df = pd.DataFrame(index=groups, columns=groups, data=False)
    # Twin is when both group a value always gives group b value on corresponding row, and vice versa
    for i in range(len(groups)):
        counts_df.loc[groups[i], groups[:i] + groups[i + 1:]] = df.groupby(groups[i]).nunique().max() == 1
    return np.logical_and(counts_df, counts_df.T)


def init(df: pd.DataFrame) -> GroupSet:
    """
    Init a group set from a dataframe

    :param df: DataFrame containing only the group columns
    :return: GroupSet created from the input DataFrame
    """

    groups = {column: Group(column, df[column].unique()) for column in df.columns}

    parent_df = get_parent_df(df)
    for i, (group_name, is_parent) in enumerate(parent_df.iterrows()):
        # Create data frame with only rows with first appearance of each unique value of group column
        map_df = df.set_index(group_name)
        map_df = map_df.loc[~map_df.index.duplicated(keep='first'), :]
        for parent_name in df.columns[is_parent]:
            parent_mapping = groups[parent_name].mapping.loc[map_df[parent_name], parent_name]
            groups[group_name].add_parent(groups[parent_name], parent_mapping.values)
            groups[parent_name].add_child(groups[group_name])

    twin_df = get_twin_df(df)
    for group_name, is_twin in twin_df.iterrows():
        for twin_name in df.columns[is_twin]:
            groups[group_name].add_twin(groups[twin_name])

    return GroupSet(groups)

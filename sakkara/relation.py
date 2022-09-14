from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupMember:
    name: str
    id: int
    parent: 'GroupMember' = field(default=None)


@dataclass(frozen=True)
class Group:
    name: str
    members: List[GroupMember] = field(default_factory=list)
    parent: Optional['Group'] = field(default=None)

    def get_members(self, add_parents=True) -> pd.DataFrame:
        df = pd.DataFrame({self.name: [m.name for m in self.members], f'{self.name}_id': [m.id for m in self.members]})
        if self.parent is not None and add_parents:
            df.loc[:, self.parent.name] = [m.parent.name for m in self.members]
            df.loc[:, f'{self.parent.name}_id'] = [m.parent.id for m in self.members]

        return df

    def get_parent_mapping(self, parent_name: str) -> pd.DataFrame:
        if self.parent is None:
            return pd.DataFrame()
        df = self.get_members()
        if self.parent.name != parent_name:
            df = df.join(self.parent.get_parent_mapping(parent_name), on=f'{self.parent.name}_id', rsuffix='_l')

        return df

    def is_parent(self, other: str):
        pm = self.get_parent_mapping(other)
        return other in pm.columns


@dataclass(frozen=True)
class GroupSet:
    groups: Dict[str, Group]

    def __getitem__(self, item):
        return self.groups[item]

    def coords(self):
        c = {name: np.array([m.name for m in group.members], dtype=object) for name, group in self.groups.items()}
        return c


def get_relations(df: pd.DataFrame, *groups: str):
    group = groups[0]
    other = groups[1:]
    counts = df.groupby(group)[[g for g in other]].nunique().max()
    parents = [o for o in other if 1 == counts[o]]
    children = [o for o in other if 1 < counts[o]]

    return group, parents, children


def trace_hierarchical_order(df: pd.DataFrame, *groups: str) -> List[str]:
    group, parents, children = get_relations(df, *groups)

    ordered = [group]
    if 0 < len(parents):
        ordered = trace_hierarchical_order(df, *parents) + ordered
    if 0 < len(children):
        ordered = ordered + trace_hierarchical_order(df, *children)

    return ordered


def init_groupset(df: pd.DataFrame, group_cols: Set[str], coeff_cols: Set[str]) -> GroupSet:
    df = df.copy()
    df['obs'] = np.arange(len(df))

    df['global'] = 'global'
    group_col_names = group_cols.union({'global', 'obs'})

    ordered = trace_hierarchical_order(df, *group_col_names)

    first_members = [GroupMember(name, i) for i, name in enumerate(df[ordered[0]].unique())]
    groups = [Group(ordered[0], first_members)]

    for col in ordered[1:]:
        parent = groups[-1]
        mapping = df.groupby(col)[parent.name].first()
        parent_members = {m.name: m for m in parent.members}
        members = [GroupMember(name, i, parent_members[mapping[name]]) for i, name in
                   enumerate(df[col].unique())]
        groups.append(Group(col, members, parent=parent))

    groups.append(Group('column', [GroupMember(c, i) for i, c in enumerate(coeff_cols)]))

    return GroupSet({g.name: g for g in groups})

from itertools import permutations
from typing import List

import numpy as np
import pandas as pd
import pytest

from hierlinreg.relation import trace_hierarchical_order, init_groups


@pytest.fixture
def df():
    return pd.DataFrame(
        {'a': map(str, np.repeat(np.arange(2), 16)),
         'b': map(str, np.repeat(np.arange(4), 8)),
         'c': map(str, np.repeat(np.arange(8), 4)),
         'd': map(str, np.repeat(np.arange(16), 2)),
         'random_values': np.random.randn(32)
         }
    )


@pytest.fixture
def cols():
    return list('abcd')


def test_trace_hierarchical_relations(df: pd.DataFrame, cols: List[str]):
    for p in permutations(cols):
        ordered = trace_hierarchical_order(df, *p)
        assert list('abcd') == ordered


def test_init_groups(df: pd.DataFrame, cols: List[str]):
    groupset = init_groups(df, *cols)
    expected_cols = ['global'] + list('abcd') + ['column', 'observation']
    assert all(e in groupset.groups.keys() for e in expected_cols)

    assert groupset.groups['global'].parent is None
    assert groupset.groups['a'].parent == groupset.groups['global']
    assert list(map(lambda m: m.name, groupset.groups['a'].members)) == list(map(str, [0, 1]))
    assert all(list(map(lambda m: m.parent is groupset['global'].members[0], groupset.groups['a'].members)))

    gnames = 'abcd'
    for i in range(1, 4):
        assert groupset.groups[gnames[i]].parent == groupset.groups[gnames[i - 1]]
        assert list(map(lambda m: m.name, groupset.groups[gnames[i]].members)) == list(map(str, range(2 ** (i + 1))))
        assert all(
            list(map(lambda m: m.parent in groupset.groups[gnames[i - 1]].members, groupset.groups[gnames[i]].members)))


def test_set_get_coords(df: pd.DataFrame, cols: List[str]):
    groupset = init_groups(df, *cols)
    coords = groupset.coords()

    assert list(coords['a']) == list(range(2))
    assert list(coords['b']) == list(range(4))
    assert list(coords['c']) == list(range(8))
    assert list(coords['d']) == list(range(16))


def test_get_parent_mapping(df: pd.DataFrame, cols: List[str]):
    groupset = init_groups(df, *cols)
    assert list(groupset['a'].get_parent_mapping('global')['global']) == [0, 0]
    assert list(groupset['b'].get_parent_mapping('a')['a']) == list(map(str, [0, 0, 1, 1]))
    assert list(groupset['b'].get_parent_mapping('a')['a_id']) == [0, 0, 1, 1]
    assert list(groupset['c'].get_parent_mapping('b')['b']) == list(map(str, [0, 0, 1, 1, 2, 2, 3, 3]))
    assert list(groupset['c'].get_parent_mapping('b')['b_id']) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert list(groupset['c'].get_parent_mapping('a')['a']) == list(map(str, [0, 0, 0, 0, 1, 1, 1, 1]))
    assert list(groupset['c'].get_parent_mapping('a')['a_id']) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert list(groupset['d'].get_parent_mapping('c')['c']) == list(
        map(str, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]))
    assert list(groupset['d'].get_parent_mapping('c')['c_id']) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    assert list(groupset['d'].get_parent_mapping('b')['b']) == list(
        map(str, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]))
    assert list(groupset['d'].get_parent_mapping('b')['b_id']) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert list(groupset['d'].get_parent_mapping('a')['a']) == list(
        map(str, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
    assert list(groupset['d'].get_parent_mapping('a')['a_id']) == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

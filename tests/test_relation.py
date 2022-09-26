import numpy as np
import pandas as pd
import pytest

from sakkara.relation import groupset
from sakkara.relation.composite import CompositePair, CompositeBase


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            'g': map(str, np.repeat('global', 32)),
            'a': map(str, np.repeat(np.arange(2), 16)),
            'b': map(str, np.repeat(np.arange(4), 8)),
            'c': map(str, np.repeat(np.arange(8), 4)),
            'd': map(str, np.repeat(np.arange(3), [16, 5, 11])),
            'e': map(str, np.repeat(np.arange(2), [21, 11])),
            'o': map(str, np.arange(32))
        }
    )


@pytest.fixture
def graph_dict():
    return {
        'g': {'children': list('abcdeo'), 'parents': [], 'neither': []},
        'a': {'children': list('bcdo'), 'parents': 'g', 'neither': []},
        'b': {'children': list('co'), 'parents': list('ga'), 'neither': list('de')},
        'c': {'children': 'o', 'parents': list('gab'), 'neither': list('de')},
        'd': {'children': 'o', 'parents': ['g', 'a', 'e'], 'neither': list('bc')},
        'e': {'children': list('od'), 'parents': 'g', 'neither': list('abc')},
        'o': {'children': [], 'parents': list('gabcde'), 'neither': []}
    }


def test_get_parent_df(df):
    parent_df = groupset.get_parent_df(df)
    group_order = list(parent_df.index)

    assert 'g' == group_order[0]
    assert all(map(lambda k: k in group_order[1:3], list('ae')))
    assert all(map(lambda k: k in group_order[3:6], list('bdc')))
    assert 'o' == group_order[6]

    # Group should not be a parent to itself
    for g in group_order:
        assert not parent_df.loc[g, g]

    # Global group
    for g in list('abcdeo'):
        assert parent_df.loc[g, 'g']

    # Observation group
    for g in list('abcdeg'):
        assert not parent_df.loc[g, 'o']

    # Group a
    assert not any(parent_df.loc['a', list('bcde')])

    # group b
    assert all(parent_df.loc['b', list('a')])
    assert not any(parent_df.loc['b', list('cdeo')])

    # group c
    assert all(parent_df.loc['c', list('ab')])
    assert not any(parent_df.loc['c', list('de')])

    # group d
    assert all(parent_df.loc['d', list('ae')])
    assert not any(parent_df.loc['d', list('bc')])

    # group e
    assert not any(parent_df.loc['e', list('abcd')])


def test_init_groups(df):
    gs = groupset.init(df)

    expected_groups = list('gabcdeo')
    assert len(expected_groups) == len(gs.groups)
    assert all(g in expected_groups for g in gs.groups.keys())

    assert len(gs['g']) == 1
    assert len(gs['a']) == 2
    assert len(gs['b']) == 4
    assert len(gs['c']) == 8
    assert len(gs['d']) == 3
    assert len(gs['e']) == 2
    assert len(gs['o']) == 32


def test_get_coords(df):
    gs = groupset.init(df)
    coords = gs.coords()

    assert coords['g'] == ['global']
    assert list(coords['a']) == list(map(str, np.arange(2)))
    assert list(coords['b']) == list(map(str, np.arange(4)))
    assert list(coords['c']) == list(map(str, np.arange(8)))
    assert list(coords['d']) == list(map(str, np.arange(3)))
    assert list(coords['e']) == list(map(str, np.arange(2)))
    assert list(coords['o']) == list(map(str, np.arange(32)))


def test_relations(df, graph_dict):
    gs = groupset.init(df)

    for k, v in graph_dict.items():
        for child in v['children']:
            assert gs[k].is_parent_to(gs[child])
            assert not gs[child].is_parent_to(gs[k])
        for parent in v['parents']:
            assert not gs[k].is_parent_to(gs[parent])
            assert gs[parent].is_parent_to(gs[k])
        for other in v['neither']:
            assert not gs[k].is_parent_to(gs[other])
            assert not gs[other].is_parent_to(gs[k])


def test_internal_relation(df, graph_dict):
    gs = groupset.init(df)

    for k, v in graph_dict.items():
        for child in v['children']:
            cp = CompositePair(gs[k], gs[child])
            assert cp.internal_relation() == (gs[k], gs[child])
            assert gs[k].is_parent_to(cp)
            assert not gs[child].is_parent_to(cp)
            assert not cp.is_parent_to(gs[child])
            assert not cp.is_parent_to(gs[k])
        for parent in v['parents']:
            cp = CompositePair(gs[parent], gs[k])
            assert cp.internal_relation() == (gs[parent], gs[k])
            assert not gs[k].is_parent_to(cp)
            assert gs[parent].is_parent_to(cp)
            assert not cp.is_parent_to(gs[parent])
            assert not cp.is_parent_to(gs[k])
        for other in v['neither']:
            cp = CompositePair(gs[k], gs[other])
            assert cp.internal_relation() == (None, None)
            assert gs[k].is_parent_to(cp)
            assert gs[other].is_parent_to(cp)
            assert not cp.is_parent_to(gs[k])
            assert not cp.is_parent_to(gs[other])

    assert CompositePair(CompositePair(gs['a'], gs['b']), gs['e']).internal_relation() == (None, None)
    assert CompositePair(CompositePair(gs['b'], gs['e']), gs['a']).internal_relation() == (None, None)
    parent, child = CompositePair(CompositePair(gs['a'], gs['b']), gs['c']).internal_relation()
    assert isinstance(parent, CompositePair)
    assert isinstance(child, CompositeBase)
    assert parent.a == gs['a']
    assert parent.b == gs['b']
    assert child == gs['c']


def test_mapping(df):
    gs = groupset.init(df)

    ab = CompositePair(gs['a'], gs['b'])
    assert len(ab) == 4
    assert list(map(str, ab.map_from(gs['a']))) == list(map(str, [0, 0, 1, 1]))
    assert list(map(str, ab.map_from(gs['b']))) == list(map(str, [0, 1, 2, 3]))

    bd = CompositePair(gs['b'], gs['d'])
    assert len(bd) == 6
    assert list(map(str, bd.map_from(gs['b']))) == list(map(str, [0, 1, 2, 2, 3, 3]))
    assert list(map(str, bd.map_from(gs['d']))) == list(map(str, [0, 0, 1, 2, 1, 2]))
    assert list(map(str, bd.map_from(gs['a']))) == list(map(str, [0, 0, 1, 1, 1, 1]))

    abd = CompositePair(ab, bd)
    assert len(abd) == 6
    assert list(map(str, abd.map_from(gs['b']))) == list(map(str, [0, 1, 2, 2, 3, 3]))
    assert list(map(str, abd.map_from(gs['d']))) == list(map(str, [0, 0, 1, 2, 1, 2]))
    assert list(map(str, abd.map_from(gs['a']))) == list(map(str, [0, 0, 1, 1, 1, 1]))

    abdo = CompositePair(abd, gs['o'])
    assert len(abdo) == 32
    assert list(map(str, abdo.map_from(gs['b']))) == list(map(str, np.repeat(np.arange(4), 8)))
    assert list(map(str, abdo.map_from(gs['d']))) == list(map(str, np.repeat(np.arange(3), [16, 5, 11])))
    assert list(map(str, abdo.map_from(gs['a']))) == list(map(str, np.repeat(np.arange(2), 16)))
    assert list(map(str, abdo.map_from(gs['o']))) == list(map(str, np.arange(32)))

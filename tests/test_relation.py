import numpy as np
import pandas as pd
import pytest

from sakkara.relation import groupset
from sakkara.relation.node import NodePair


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            'g': map(str, np.repeat('global', 32)),
            'a': map(str, np.repeat(np.arange(2), 16)),
            'b': map(str, np.repeat(np.arange(4), 8)),
            'c': map(str, np.repeat(np.arange(8), 4)),
            'd': map(str, np.repeat(np.arange(3), [16, 5, 11])),
            'e': np.repeat(np.arange(2), [21, 11]),
            'o': np.arange(32),
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
    df['o_copy'] = df['o'].copy()
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

    assert not parent_df.loc['o_copy', 'o']
    assert not parent_df.loc['o', 'o_copy']


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
    assert list(coords['e']) == list(np.arange(2))
    assert list(coords['o']) == list(np.arange(32))


def test_relations(df, graph_dict):
    gs = groupset.init(df)

    for k, v in graph_dict.items():
        assert not gs[k].is_parent_to(gs[k])
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
        cp = NodePair(gs[k], gs[k])
        assert not cp.is_parent_to(gs[k])
        assert not gs[k].is_parent_to(cp)
        assert gs[k].representation() == cp.representation()

        for child in v['children']:
            assert gs[child] in gs[k].get_children()
            assert not gs[k] in gs[child].get_children()
            cp = NodePair(gs[k], gs[child])
            assert gs[k].is_parent_to(cp)
            assert not gs[child].is_parent_to(cp)
            assert not cp.is_parent_to(gs[child])
            assert not cp.is_parent_to(gs[k])
        for parent in v['parents']:
            cp = NodePair(gs[parent], gs[k])
            assert gs[k] in gs[parent].get_children()
            assert not gs[parent] in gs[k].get_children()
            assert not gs[k].is_parent_to(cp)
            assert gs[parent].is_parent_to(cp)
            assert not cp.is_parent_to(gs[parent])
            assert not cp.is_parent_to(gs[k])
        for other in v['neither']:
            assert not gs[other] in gs[k].get_children()
            assert not gs[k] in gs[other].get_children()
            cp = NodePair(gs[k], gs[other])
            assert gs[k].is_parent_to(cp)
            assert gs[other].is_parent_to(cp)
            assert not cp.is_parent_to(gs[k])
            assert not cp.is_parent_to(gs[other])


def test_mapping(df):
    gs = groupset.init(df)

    aa = NodePair(gs['a'], gs['a'])
    assert aa.reduced_repr() == gs['a']
    assert len(aa) == 2
    assert aa.map_to(gs['a']).tolist() == [0, 1]
    aaa = NodePair(aa, gs['a'])
    assert aaa.reduced_repr() == gs['a']
    assert len(aaa) == 2
    assert aaa.map_to(gs['a']).tolist() == [0, 1]
    assert aaa.map_to(aa).tolist() == [0, 1]

    with pytest.raises(ValueError):
        gs['a'].map_to(gs['b'])
        gs['a'].map_to(gs['c'])
        gs['a'].map_to(gs['e'])

    ab = NodePair(gs['a'], gs['b'])
    assert len(ab) == 4
    assert ab.map_to(gs['a']).tolist() == [0, 0, 1, 1]
    assert ab.map_to(gs['b']).tolist() == [0, 1, 2, 3]
    assert ab.reduced_repr() == gs['b']

    with pytest.raises(ValueError):
        ab.map_to(gs['c'])
        ab.map_to(gs['e'])

    bd = NodePair(gs['b'], gs['d'])
    assert len(bd) == 5
    assert bd.map_to(gs['b']).tolist() == [0, 1, 2, 2, 3]
    assert bd.map_to(gs['d']).tolist() == [0, 0, 1, 2, 2]
    assert bd.map_to(gs['a']).tolist() == [0, 0, 1, 1, 1]
    assert bd.map_to(gs['e']).tolist() == [0, 0, 0, 1, 1]
    assert bd.reduced_repr() == bd

    with pytest.raises(ValueError):
        bd.map_to(gs['c'])

    abd = NodePair(ab, bd)
    assert len(abd) == 5
    assert abd.map_to(gs['b']).tolist() == [0, 1, 2, 2, 3]
    assert abd.map_to(gs['d']).tolist() == [0, 0, 1, 2, 2]
    assert abd.map_to(gs['a']).tolist() == [0, 0, 1, 1, 1]
    assert abd.map_to(gs['e']).tolist() == [0, 0, 0, 1, 1]
    assert abd.reduced_repr() == bd

    with pytest.raises(ValueError):
        abd.map_to(gs['c'])

    abdo = NodePair(abd, gs['o'])
    assert len(abdo) == 32
    assert abdo.map_to(gs['b']).tolist() == np.repeat(np.arange(4), 8).tolist()
    assert abdo.map_to(gs['d']).tolist() == np.repeat(np.arange(3), [16, 5, 11]).tolist()
    assert abdo.map_to(gs['a']).tolist() == np.repeat(np.arange(2), 16).tolist()
    assert abdo.map_to(gs['o']).tolist() == np.arange(32).tolist()
    assert abdo.reduced_repr() == gs['o']

    ac = NodePair(gs['a'], gs['c'])
    acb = NodePair(ac, gs['b'])
    assert ac.representation() == acb.representation()

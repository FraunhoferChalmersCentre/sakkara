import numpy as np
import pandas as pd
import pytest

from sakkara.relation import groupset
from sakkara.relation.compositegroup import CompositeGroupPair, CompositeAtomicGroup


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
        cp = CompositeGroupPair(gs[k], gs[k])
        assert cp.a_parent_b() is None
        assert not cp.is_parent_to(gs[k])
        assert not gs[k].is_parent_to(cp)
        assert gs[k].get_representation_groups() == cp.get_representation_groups()

        for child in v['children']:
            cp = CompositeGroupPair(gs[k], gs[child])
            assert cp.a_parent_b()
            assert gs[k].is_parent_to(cp)
            assert not gs[child].is_parent_to(cp)
            assert not cp.is_parent_to(gs[child])
            assert not cp.is_parent_to(gs[k])
        for parent in v['parents']:
            cp = CompositeGroupPair(gs[parent], gs[k])
            assert cp.a_parent_b()
            assert not gs[k].is_parent_to(cp)
            assert gs[parent].is_parent_to(cp)
            assert not cp.is_parent_to(gs[parent])
            assert not cp.is_parent_to(gs[k])
        for other in v['neither']:
            cp = CompositeGroupPair(gs[k], gs[other])
            assert cp.a_parent_b() is None
            assert gs[k].is_parent_to(cp)
            assert gs[other].is_parent_to(cp)
            assert not cp.is_parent_to(gs[k])
            assert not cp.is_parent_to(gs[other])

    assert CompositeGroupPair(CompositeGroupPair(gs['a'], gs['b']), gs['e']).a_parent_b() is None
    assert not CompositeGroupPair(CompositeGroupPair(gs['b'], gs['e']), gs['a']).a_parent_b()
    assert CompositeGroupPair(CompositeGroupPair(gs['a'], gs['b']), gs['c']).a_parent_b()


def test_mapping(df):
    gs = groupset.init(df)

    aa = CompositeGroupPair(gs['a'], gs['a'])
    assert len(aa) == 2
    assert list(map(str, aa.map_from(gs['a']))) == list(map(str, [0, 1]))
    aaa = CompositeGroupPair(aa, gs['a'])
    assert len(aaa) == 2
    assert list(map(lambda m: m.index, aaa.map_from(gs['a']))) == [0, 1]
    assert list(map(lambda m: m.index, aaa.map_from(aa))) == [0, 1]

    ab = CompositeGroupPair(gs['a'], gs['b'])
    assert len(ab) == 4
    assert list(map(lambda m: m.index, ab.map_from(gs['a']))) == [0, 0, 1, 1]
    assert list(map(lambda m: m.index, ab.map_from(gs['b']))) == [0, 1, 2, 3]

    bd = CompositeGroupPair(gs['b'], gs['d'])
    assert len(bd) == 6
    assert list(map(lambda m: m.index, bd.map_from(gs['b']))) == [0, 1, 2, 2, 3, 3]
    assert list(map(lambda m: m.index, bd.map_from(gs['d']))) == [0, 0, 1, 2, 1, 2]
    assert list(map(lambda m: m.index, bd.map_from(gs['a']))) == [0, 0, 1, 1, 1, 1]

    abd = CompositeGroupPair(ab, bd)
    assert len(abd) == 6
    assert list(map(lambda m: m.index, abd.map_from(gs['b']))) == [0, 1, 2, 2, 3, 3]
    assert list(map(lambda m: m.index, abd.map_from(gs['d']))) == [0, 0, 1, 2, 1, 2]
    assert list(map(lambda m: m.index, abd.map_from(gs['a']))) == [0, 0, 1, 1, 1, 1]

    abdo = CompositeGroupPair(abd, gs['o'])
    assert len(abdo) == 32
    assert list(map(lambda m: m.index, abdo.map_from(gs['b']))) == np.repeat(np.arange(4), 8).tolist()
    assert list(map(lambda m: m.index, abdo.map_from(gs['d']))) == np.repeat(np.arange(3), [16, 5, 11]).tolist()
    assert list(map(lambda m: m.index, abdo.map_from(gs['a']))) == np.repeat(np.arange(2), 16).tolist()
    assert list(map(lambda m: m.index, abdo.map_from(gs['o']))) == np.arange(32).tolist()

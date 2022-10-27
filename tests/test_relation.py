import numpy as np
from numpy import testing
import pandas as pd
import pytest

from sakkara.relation import groupset
from sakkara.relation.node import NodePair, Node

"""
Testing relations is performed on the following graph (parents in top):

      g (global)
      |\
      | \
      |  \
      a   e
      |\  |
      | \ |
      |  \|
      b   d
      |   |
      c   |
       \ /
        o (observation)
      

"""


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
def gs(df):
    return groupset.init(df)


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


def test_init_groups(gs):
    expected_groups = list('gabcdeo')
    assert len(expected_groups) == len(gs.groups)
    assert all(g in expected_groups for g in gs.groups.keys())

    assert len(gs['g'].get_members()) == 1
    assert len(gs['a'].get_members()) == 2
    assert len(gs['b'].get_members()) == 4
    assert len(gs['c'].get_members()) == 8
    assert len(gs['d'].get_members()) == 3
    assert len(gs['e'].get_members()) == 2
    assert len(gs['o'].get_members()) == 32


def test_get_coords(gs):
    coords = gs.coords()

    assert coords['g'] == ['global']
    assert list(coords['a']) == list(map(str, np.arange(2)))
    assert list(coords['b']) == list(map(str, np.arange(4)))
    assert list(coords['c']) == list(map(str, np.arange(8)))
    assert list(coords['d']) == list(map(str, np.arange(3)))
    assert list(coords['e']) == list(np.arange(2))
    assert list(coords['o']) == list(np.arange(32))


def test_relations(gs, graph_dict):
    for k, v in graph_dict.items():
        assert not gs[k].is_parent_to(gs[k])
        for child in v['children']:
            assert gs[k].is_parent_to(gs[child])
            assert not gs[child].is_parent_to(gs[k])
            assert gs[child] in gs[k].get_children()
            assert not gs[k] in gs[child].get_children()
        for parent in v['parents']:
            assert not gs[k].is_parent_to(gs[parent])
            assert gs[parent].is_parent_to(gs[k])
            assert gs[k] in gs[parent].get_children()
            assert not gs[parent] in gs[k].get_children()
        for other in v['neither']:
            assert not gs[k].is_parent_to(gs[other])
            assert not gs[other].is_parent_to(gs[k])
            assert not gs[other] in gs[k].get_children()
            assert not gs[k] in gs[other].get_children()


def test_pairing(gs, graph_dict):
    for k, v in graph_dict.items():
        self_pair = NodePair(gs[k], gs[k])
        assert not self_pair.is_parent_to(gs[k])
        assert not gs[k].is_parent_to(self_pair)
        assert self_pair.representation() == {gs[k]}
        assert self_pair.reduced_repr() == gs[k]

        for child in v['children']:
            for pair in (NodePair(gs[k], gs[child]), NodePair(gs[child], gs[k])):
                assert gs[k].is_parent_to(pair)
                assert not gs[child].is_parent_to(pair)
                assert not pair.is_parent_to(gs[child])
                assert not pair.is_parent_to(gs[k])
                assert pair.representation() == {gs[child]}
                assert pair.reduced_repr() == gs[child]

        for parent in v['parents']:
            for pair in (NodePair(gs[k], gs[parent]), NodePair(gs[parent], gs[k])):
                assert not gs[k].is_parent_to(pair)
                assert gs[parent].is_parent_to(pair)
                assert not pair.is_parent_to(gs[parent])
                assert not pair.is_parent_to(gs[k])
                assert pair.representation() == {gs[k]}
                assert pair.reduced_repr() == gs[k]

        for other in v['neither']:
            pair = NodePair(gs[k], gs[other])
            assert gs[k].is_parent_to(pair)
            assert gs[other].is_parent_to(pair)
            assert not pair.is_parent_to(gs[k])
            assert not pair.is_parent_to(gs[other])
            assert pair.representation() == {gs[k], gs[other]}
            assert pair.reduced_repr() == pair

            pair = NodePair(gs[other], gs[k])
            assert gs[k].is_parent_to(pair)
            assert gs[other].is_parent_to(pair)
            assert not pair.is_parent_to(gs[k])
            assert not pair.is_parent_to(gs[other])
            assert pair.representation() == {gs[other], gs[k]}
            assert pair.reduced_repr() == pair


def test_mapping(gs, graph_dict):
    def test_child_parent(child: Node, parent: Node):
        mappings = child.map_to(parent)
        assert len(mappings) == len(parent.get_members().shape)
        for i, mapping in enumerate(mappings):
            assert np.array(mapping).shape == child.get_members().shape
            assert np.all(np.isin(mapping, np.arange(parent.get_members().shape[i])))

        with pytest.raises(ValueError):
            parent.map_to(child)

    def test_same_representation(a: Node, b: Node):
        for i, mapping in enumerate(a.map_to(b)):
            testing.assert_array_equal(mapping, np.arange(a.get_members().shape[i]))

    def test_restructured_representation(a: Node, b: Node):
        for i, mapping in enumerate(a.map_to(b)):
            assert np.array(mapping).shape == a.get_members().shape
            testing.assert_array_equal(np.unique(mapping), np.arange(b.get_members().shape[i]))

    def test_unrelated(a: Node, b: Node):
        with pytest.raises(ValueError):
            a.map_to(b)

        with pytest.raises(ValueError):
            b.map_to(a)

    for k, v in graph_dict.items():
        test_same_representation(gs[k], gs[k])
        test_same_representation(gs[k], NodePair(gs[k], gs[k]))

        for child_name in v['children']:
            test_child_parent(gs[child_name], gs[k])
            test_child_parent(NodePair(gs[k], gs[child_name]), gs[k])
            test_same_representation(NodePair(gs[k], gs[child_name]), gs[child_name])
            test_same_representation(NodePair(gs[k], gs[child_name]), NodePair(gs[child_name], gs[k]))

        for parent_name in v['parents']:
            test_child_parent(gs[k], gs[parent_name])
            test_child_parent(NodePair(gs[k], gs[parent_name]), gs[parent_name])
            test_same_representation(NodePair(gs[k], gs[parent_name]), gs[k])
            test_same_representation(NodePair(gs[k], gs[parent_name]), NodePair(gs[parent_name], gs[k]))

        for other_name in v['neither']:
            test_unrelated(gs[k], gs[other_name])
            test_child_parent(NodePair(gs[k], gs[other_name]), gs[k])
            test_child_parent(NodePair(gs[k], gs[other_name]), gs[other_name])
            test_restructured_representation(NodePair(gs[k], gs[other_name]), NodePair(gs[other_name], gs[k]))

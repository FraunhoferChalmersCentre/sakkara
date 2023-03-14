import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from sakkara.relation import groupset
from sakkara.relation.representation import Representation, TensorRepresentation as TR

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

    assert len(gs['g']) == 1
    assert len(gs['a']) == 2
    assert len(gs['b']) == 4
    assert len(gs['c']) == 8
    assert len(gs['d']) == 3
    assert len(gs['e']) == 2
    assert len(gs['o']) == 32


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
        assert gs[k] in gs[k].twins
        for child in v['children']:
            assert gs[k] in gs[child].parents
            assert gs[child] not in gs[k].parents
            assert gs[child] in gs[k].children
            assert gs[k] not in gs[child].children
            assert gs[k] not in gs[child].twins
            assert gs[child] not in gs[k].twins
        for parent in v['parents']:
            assert gs[k] not in gs[parent].parents
            assert gs[parent] in gs[k].parents
            assert gs[k] in gs[parent].children
            assert gs[parent] not in gs[k].children
            assert gs[k] not in gs[parent].twins
            assert gs[parent] not in gs[k].twins
        for other in v['neither']:
            assert gs[k] not in gs[other].parents
            assert gs[other] not in gs[k].parents
            assert gs[other] not in gs[k].children
            assert gs[k] not in gs[other].children
            assert gs[k] not in gs[other].twins
            assert gs[other] not in gs[k].twins


def test_pairing(gs, graph_dict):
    for k, v in graph_dict.items():
        self_repr = TR(gs[k], gs[k])
        assert self_repr.groups == [gs[k]]

        for child in v['children']:
            for pair in (TR(gs[k], gs[child]), TR(gs[child], gs[k])):
                assert pair.groups == [gs[child]]

        for parent in v['parents']:
            for pair in (TR(gs[k], gs[parent]), TR(gs[parent], gs[k])):
                assert pair.groups == [gs[k]]

        for other in v['neither']:
            pair = TR(gs[k], gs[other])
            assert pair.groups == [gs[k], gs[other]]

            pair = TR(gs[other], gs[k])
            assert pair.groups == [gs[other], gs[k]]


def test_mapping(df, gs, graph_dict):
    def test_child_parent(child: npt.NDArray, child_repr: Representation, parent: npt.NDArray,
                          parent_repr: Representation):

        with pytest.raises(ValueError):
            child_repr.map(child, parent_repr)

        mapped = parent_repr.map(parent, child_repr)
        assert mapped.shape == child_repr.get_shape()
        assert mapped.shape == child.shape
        assert all(v in mapped.ravel() for v in parent.ravel())

    def test_permuted_representation(x: npt.NDArray, a: Representation, y: npt.NDArray, b: Representation):
        mapped = a.map(x, b)
        assert mapped.shape == b.get_shape()
        assert mapped.shape == y.shape

        mapped = b.map(y, a)
        assert mapped.shape == a.get_shape()
        assert mapped.shape == x.shape

    def test_same_representation(x: npt.NDArray, a: Representation, y: npt.NDArray, b: Representation):
        mapped = a.map(x, b)
        assert all(m == xi for m, xi in zip(mapped.ravel(), x.ravel()))

        mapped = b.map(y, a)
        assert all(m == yi for m, yi in zip(mapped.ravel(), y.ravel()))

    def test_unrelated(x: npt.NDArray, a: Representation, y: npt.NDArray, b: Representation):
        with pytest.raises(ValueError):
            a.map(x, b)

        with pytest.raises(ValueError):
            b.map(y, a)

    for k, v in graph_dict.items():
        test_same_representation(df[k].unique(), TR(gs[k]), df[k].unique(), TR(gs[k]))
        test_same_representation(df[k].unique(), TR(gs[k]), df[k].unique(), TR(gs[k], gs[k]))

        for child_name in v['children']:
            test_child_parent(df[child_name].unique(), TR(gs[child_name]), df[k].unique(), TR(gs[k]))
            test_child_parent(df[child_name].unique(), TR(gs[k], gs[child_name]), df[k].unique(), TR(gs[k]))
            test_same_representation(df[child_name].unique(), TR(gs[k], gs[child_name]), df[child_name].unique(), TR(gs[child_name]))
            test_same_representation(df[child_name].unique(), TR(gs[k], gs[child_name]), df[child_name].unique(),
                                     TR(gs[child_name], gs[k]))

        for parent_name in v['parents']:
            test_child_parent(df[k].unique(), TR(gs[k]), df[parent_name].unique(), TR(gs[parent_name]))
            test_child_parent(df[k].unique(), TR(gs[k], gs[parent_name]), df[parent_name].unique(), TR(gs[parent_name]))
            test_same_representation(df[k].unique(), TR(gs[k], gs[parent_name]), df[k].unique(), TR(gs[k]))
            test_same_representation(df[k].unique(), TR(gs[k], gs[parent_name]), df[k].unique(), TR(gs[parent_name], gs[k]))

        for other_name in v['neither']:
            test_unrelated(df[k].unique(), TR(gs[k]), df[other_name].unique(), TR(gs[other_name]))

            ko = np.arange(len(df[k].unique()) * len(df[other_name].unique())).reshape(len(df[k].unique()),
                                                                                       len(df[other_name].unique()))
            ok = np.arange(len(df[k].unique()) * len(df[other_name].unique())).reshape(len(df[other_name].unique()),
                                                                                       len(df[k].unique())) + 1000

            test_child_parent(ko, TR(gs[k], gs[other_name]), df[k].unique(), TR(gs[k]))
            test_child_parent(ko, TR(gs[k], gs[other_name]), df[other_name].unique(), TR(gs[other_name]))
            test_same_representation(ko, TR(gs[k], gs[other_name]), ko, TR(gs[k], gs[other_name]))
            test_permuted_representation(ko, TR(gs[k], gs[other_name]), ok, TR(gs[other_name], gs[k]))

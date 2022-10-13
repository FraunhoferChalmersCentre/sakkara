import numpy as np
import pandas as pd
import pymc as pm
import pytest

from sakkara.model.components import Distribution
from sakkara.relation.groupset import init, GroupSet


@pytest.fixture
def groupset() -> GroupSet:
    df = pd.DataFrame(
        {
            'global': 'global',
            'building': np.repeat(list('ab'), 10),
            'sensor': np.repeat(list('stuv'), 5),
            'time': np.tile(np.arange(5), 4),
            'obs': np.arange(20)
        })

    return init(df)


def test_build_variables(groupset):
    coords = groupset.coords()

    rv_global = Distribution(pm.Normal, mu=0)
    rv_building = Distribution(pm.Normal, column='building', mu=rv_global)
    rv_sensor = Distribution(pm.Normal, column='sensor', mu=rv_building, name='rv')

    with pm.Model(coords=coords):
        rv_sensor.build(groupset)
        assert pm.draw(rv_global.variable).shape == ()
        assert pm.draw(rv_building.variable).shape == (2,)
        assert pm.draw(rv_sensor.variable).shape == (4,)


def test_retrieve_groups():
    hv = Distribution(pm.Normal, column='room', mu=Distribution(pm.Normal, column='building', mu=Distribution(pm.Normal)))
    assert all(c in hv.retrieve_columns() for c in ['room', 'building', 'global'])

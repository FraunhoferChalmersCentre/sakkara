import numpy as np
import pandas as pd
import pymc as pm
import pytest

from hierlinreg.hierarchical import HierarchicalVariable
from hierlinreg.relation import init_groups


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            'building': np.repeat(list('ab'), 10),
            'sensor': np.repeat(list('stuv'), 5),
            'time': np.tile(np.arange(5), 4)
        })

    bidx, _ = df['building'].factorize()
    sidx, _ = df['sensor'].factorize()

    df['value'] = bidx * 100 + sidx * 10 + np.random.randn(20)

    return df


def test_build_variables(df: pd.DataFrame):
    groupset = init_groups(df, 'building', 'sensor')
    coords = groupset.coords()

    rv_global = HierarchicalVariable(pm.Normal, mu=0)
    rv_building = HierarchicalVariable(pm.Normal, group_name='building', mu=rv_global)
    rv_sensor = HierarchicalVariable(pm.Normal, group_name='sensor', mu=rv_building)

    with pm.Model(coords=coords):
        rv_global.build('rv_global', groupset)
        assert rv_global.variable is not None
        assert rv_building.variable is None
        assert rv_sensor.variable is None

        rv_global.variable = None

        rv_building.build('rv_building', groupset)
        assert rv_global.variable is not None
        assert rv_building.variable is not None
        assert rv_sensor.variable is None

        rv_global.variable = None
        rv_building.variable = None

        rv_sensor.build('rv_sensor', groupset)
        assert rv_sensor.variable is not None
        assert rv_building.variable is not None
        assert rv_global.variable is not None

        assert pm.draw(rv_sensor.variable).shape == (4,)
        assert pm.draw(rv_building.variable).shape == (2,)
        assert pm.draw(rv_global.variable).shape == (1,)

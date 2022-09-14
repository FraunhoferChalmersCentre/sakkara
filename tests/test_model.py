import numpy as np
import pandas as pd
import pymc as pm
import pytest

from sakkara.model import HierarchicalVariable as HV
from sakkara.relation import init_groupset


@pytest.fixture
def groupset():
    df = pd.DataFrame(
        {
            'building': np.repeat(list('ab'), 10),
            'sensor': np.repeat(list('stuv'), 5),
            'time': np.tile(np.arange(5), 4),
            'random_values': np.random.randn(20)
        })

    bidx, _ = df['building'].factorize()
    sidx, _ = df['sensor'].factorize()

    df['value'] = bidx * 100 + sidx * 10 + np.random.randn(20)

    return init_groupset(df, {'building', 'sensor'}, {'random_values'})


def test_build_variables(groupset):
    coords = groupset.coords()

    rv_global = HV(pm.Normal, mu=0)
    rv_building = HV(pm.Normal, group='building', mu=rv_global)
    rv_sensor = HV(pm.Normal, group='sensor', mu=rv_building, name='rv')

    with pm.Model(coords=coords):
        rv_sensor.build(groupset)
        assert pm.draw(rv_global.variable).shape == ()
        assert pm.draw(rv_building.variable).shape == (2,)
        assert pm.draw(rv_sensor.variable).shape == (4,)


def test_retrieve_groups():
    hv = HV(pm.Normal, group='room', mu=HV(pm.Normal, group='building', mu=HV(pm.Normal)))

    assert all(c in hv.retrieve_groups() for c in ['room', 'building', 'global'])

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy.stats.distributions import norm, binom

from sakkara.model.components import Distribution
from sakkara.relation.groupset import init, GroupSet


@pytest.fixture
def groupset() -> GroupSet:
    df = pd.DataFrame(
        {
            'global': 'global',
            'building': np.repeat(list('ab'), 10),
            'sensor': np.repeat(list('stuv'), 5),
            'time': np.tile(pd.date_range("2018-01-01 00:00:00", tz='utc', periods=5, freq="H"), 4),
            'obs': np.arange(20)
        })

    return init(df)


def test_build_variables(groupset):
    coords = groupset.coords()

    rv_global = Distribution(pm.Normal, mu=0)
    rv_building = Distribution(pm.Normal, column='building', mu=rv_global)
    rv_sensor = Distribution(pm.Normal, column='sensor', mu=rv_building, name='rv')

    sensor_building_sum = - rv_building + 1 / rv_sensor
    N_DRAWS = 1000

    with pm.Model(coords=coords):
        sensor_building_sum.build(groupset)
        sb, s, b, g = pm.draw(
            [sensor_building_sum.variable, rv_sensor.variable, rv_building.variable, rv_global.variable], N_DRAWS, 1)
        assert sb.shape == (N_DRAWS, 4)
        assert s.shape == (N_DRAWS, 4)
        assert b.shape == (N_DRAWS, 2)
        assert g.shape == (N_DRAWS,)

        p = norm.cdf(1) - norm.cdf(-1)
        abs_error = binom.ppf(.9, N_DRAWS, p) - binom.mean(N_DRAWS, p)
        assert sum(np.abs(b.mean(axis=1) - g) <= 1 / np.sqrt(2)) == pytest.approx(p * N_DRAWS, abs=abs_error)

        assert sum(np.abs(s[:, :2].mean(axis=1) - b[:, 0]) <= 1 / np.sqrt(2)) == pytest.approx(p * N_DRAWS,
                                                                                               abs=abs_error)
        assert sum(np.abs(s[:, 2:].mean(axis=1) - b[:, 1]) <= 1 / np.sqrt(2)) == pytest.approx(p * N_DRAWS,
                                                                                               abs=abs_error)

        assert all(a == pytest.approx(b) for a, b, in zip(sb.flatten(), (1 / s - b[:, [0, 0, 1, 1]]).flatten()))


def test_retrieve_groups():
    hv = Distribution(pm.Normal, column='room',
                      mu=Distribution(pm.Normal, column='building', mu=Distribution(pm.Normal)))
    assert all(c in hv.retrieve_columns() for c in ['room', 'building', 'global'])

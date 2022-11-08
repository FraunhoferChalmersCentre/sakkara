import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy.stats.distributions import norm, binom

from sakkara.model import FixedComponent, RandomVariable
from sakkara.relation.groupset import init, GroupSet


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            'global': 'global',
            'building': np.repeat(list('ab'), 10),
            'sensor': np.repeat(list('stuv'), 5),
            'time': np.tile(pd.date_range("2018-01-01 00:00:00", tz='utc', periods=5, freq="H"), 4),
            'obs': np.arange(20)
        })


@pytest.fixture
def groupset(df) -> GroupSet:
    return init(df)


def test_retrieve_groups():
    hv = RandomVariable(pm.Normal, columns='room',
                        mu=RandomVariable(pm.Normal, columns='building', mu=RandomVariable(pm.Normal)))
    assert all(c in hv.retrieve_columns() for c in ['room', 'building', 'global'])


def test_build_variables(groupset):
    rv_global = RandomVariable(pm.Normal, mu=0)
    rv_building = RandomVariable(pm.Normal, columns='building', mu=rv_global)
    rv_sensor = RandomVariable(pm.Normal, columns='sensor', mu=rv_building, name='rv')

    sensor_building_sum = - rv_building + 1 / rv_sensor
    N_DRAWS = 1000

    with pm.Model(coords=groupset.coords()):
        sensor_building_sum.build(groupset)
        sb, s, b, g = pm.draw(
            [sensor_building_sum.variable, rv_sensor.variable, rv_building.variable, rv_global.variable], N_DRAWS, 1)
        assert sb.shape == (N_DRAWS, 4)
        assert s.shape == (N_DRAWS, 4)
        assert b.shape == (N_DRAWS, 2)
        assert g.shape == (N_DRAWS, 1)

        p = norm.cdf(1) - norm.cdf(-1)
        abs_error = binom.ppf(.9, N_DRAWS, p) - binom.mean(N_DRAWS, p)
        assert sum(np.abs(b.mean(axis=1) - g.squeeze()) <= 1 / np.sqrt(2)) == pytest.approx(p * N_DRAWS, abs=abs_error)

        assert sum(np.abs(s[:, :2].mean(axis=1) - b[:, 0]) <= 1 / np.sqrt(2)) == pytest.approx(p * N_DRAWS,
                                                                                               abs=abs_error)
        assert sum(np.abs(s[:, 2:].mean(axis=1) - b[:, 1]) <= 1 / np.sqrt(2)) == pytest.approx(p * N_DRAWS,
                                                                                               abs=abs_error)

        assert all(a == pytest.approx(b) for a, b, in zip(sb.flatten(), (1 / s - b[:, [0, 0, 1, 1]]).flatten()))


def test_tuple_column(df, groupset):
    sigma = 1e-15
    rv_time = RandomVariable(pm.Normal, mu=FixedComponent(np.arange(5), columns='time'), sigma=sigma, columns='time')
    rv_sensor = RandomVariable(pm.Normal, mu=FixedComponent(np.arange(4) * 10, columns='sensor'), sigma=sigma,
                               columns='sensor')
    rv_combined = RandomVariable(pm.Normal, mu=rv_time + rv_sensor, sigma=sigma, name='combined')

    rv_tuple_col = RandomVariable(pm.Normal,
                                  mu=FixedComponent(np.sum(np.meshgrid(np.arange(5), np.arange(4) * 10), axis=0),
                                                    columns=('time', 'sensor')),
                                  sigma=sigma,
                                  columns=('time', 'sensor'),
                                  name='tuple')

    rv_sum_1 = rv_tuple_col + rv_combined
    rv_sum_2 = rv_combined + rv_tuple_col

    rv_obs = rv_sum_1 + rv_sum_2 + RandomVariable(pm.Normal, sigma=sigma, columns='obs', name='observed')
    with pm.Model(coords=groupset.coords()):
        rv_obs.build(groupset)

    assert rv_combined.node.representation() == rv_tuple_col.node.representation()
    assert pm.draw(rv_time.variable).shape == (5,)
    assert pm.draw(rv_sensor.variable).shape == (4,)
    assert pm.draw(rv_tuple_col.variable).shape == (4, 5)
    assert pm.draw(rv_combined.variable).shape == (5, 4)
    assert pm.draw(rv_sum_1.variable).shape == (4, 5)
    assert pm.draw(rv_sum_2.variable).shape == (5, 4)
    assert pm.draw(rv_obs.variable).shape == (20,)

    time_codes, _ = pd.factorize(df['time'])
    sensor_codes, _ = pd.factorize(df['sensor'])
    assert pm.draw(rv_obs.variable) == pytest.approx(
        4 * np.arange(5)[time_codes] + 4 * (np.arange(4) * 10)[sensor_codes])

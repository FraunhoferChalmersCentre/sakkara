import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy.stats.distributions import norm, binom

from sakkara.model import FixedValueComponent, DistributionComponent
from sakkara.relation.groupset import init, GroupSet


@pytest.fixture
def df():
    return pd.DataFrame(
            {
                'global': 'global',
                'building': np.repeat(list('ab'), 10),
                'sensor': np.repeat(list('stuv'), 5),
                'time': np.tile(pd.date_range("2018-01-01 00:00:00", tz='utc', periods=5, freq="H"), 4),
                'time_2': pd.date_range("2020-01-01 00:00:00", tz='utc', periods=20, freq="H"),
                'obs': np.arange(20)
            })


@pytest.fixture
def groupset(df) -> GroupSet:
    return init(df)


def test_retrieve_groups():
    hv = DistributionComponent(pm.Normal, group='room',
                               mu=DistributionComponent(pm.Normal, group='building', mu=DistributionComponent(pm.Normal)))
    assert all(c in hv.retrieve_groups() for c in ['room', 'building', 'global'])


def test_build_variables(groupset):
    rv_global = DistributionComponent(pm.Normal, mu=0)
    rv_building = DistributionComponent(pm.Normal, group='building', mu=rv_global)
    rv_sensor = DistributionComponent(pm.Normal, group='sensor', mu=rv_building, name='rv')

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
    rv_time = DistributionComponent(pm.Normal, mu=FixedValueComponent(np.arange(5), group='time'), sigma=sigma, group='time')
    rv_sensor = DistributionComponent(pm.Normal, mu=FixedValueComponent(np.arange(4) * 10, group='sensor'), sigma=sigma,
                                      group='sensor')
    rv_combined = DistributionComponent(pm.Normal, mu=rv_time + rv_sensor, sigma=sigma, name='combined')

    rv_tuple_col = DistributionComponent(pm.Normal,
                                         mu=FixedValueComponent(np.sum(np.meshgrid(np.arange(5), np.arange(4) * 10), axis=0),
                                                                group=('time', 'sensor')),
                                         sigma=sigma,
                                         group=('time', 'sensor'),
                                         name='tuple')

    rv_sum_1 = rv_tuple_col + rv_combined
    rv_sum_2 = rv_combined + rv_tuple_col

    rv_obs = rv_sum_1 + rv_sum_2 + DistributionComponent(pm.Normal, sigma=sigma, group='obs', name='observed')
    with pm.Model(coords=groupset.coords()):
        rv_obs.build(groupset)

    assert rv_combined.node.representation() == rv_tuple_col.node.representation()
    assert pm.draw(rv_time.variable).shape == (5,)
    assert pm.draw(rv_sensor.variable).shape == (4,)
    # 'sensor' dim put before 'time' due to alphabetical order
    assert pm.draw(rv_tuple_col.variable).shape == (5, 4)
    assert pm.draw(rv_combined.variable).shape == (5, 4)
    assert pm.draw(rv_sum_1.variable).shape == (5, 4)
    assert pm.draw(rv_sum_2.variable).shape == (5, 4)
    assert pm.draw(rv_obs.variable).shape == (20,)

    time_codes, _ = pd.factorize(df['time'])
    sensor_codes, _ = pd.factorize(df['sensor'])
    assert pm.draw(rv_obs.variable) == pytest.approx(
        4 * np.arange(5)[time_codes] + 4 * (np.arange(4) * 10)[sensor_codes])


def test_one_to_one_groups(groupset):
    x = DistributionComponent(pm.Normal, mu=DistributionComponent(pm.Normal, group='obs'), group='obs', name='x')
    y = DistributionComponent(pm.Normal, mu=DistributionComponent(pm.Normal, group='time_2'), group='obs', name='y')
    ll = DistributionComponent(pm.Normal, mu=DistributionComponent(pm.Normal, group='time_2'), observed=np.random.randn(20),
                               group='obs', name='ll')

    with pm.Model(coords=groupset.coords()):
        x.build(groupset)
        assert pm.draw(x.variable).shape == (20,)

        y.build(groupset)
        assert pm.draw(y.variable).shape == (20,)

        ll.build(groupset)
        assert pm.draw(ll.variable).shape == (20,)


def test_reduction():
    tdf = pd.DataFrame(
        {
            'global': 'global',
            'a': np.array([0, 0, 0, 1, 1, 1]),
            'b': np.array([2, 2, 3, 3, 3, 3]),
            'c': np.array([0, 0, 1, 1, 1, 1]),
            'd': np.array([0, 0, 0, 1, 2, 2]),
            'e': np.array([4, 4, 4, 4, 5, 4]),
            'f': np.array([6, 6, 6, 6, 0, 6]),
            'obs': np.array([1, 2, 3, 4, 5, 6])
        })

    gs = init(tdf)

    sigma = 1e-15
    a = DistributionComponent(pm.Normal, mu=np.ones((tdf['a'].nunique())), sigma=sigma, group='a', name='dista')
    b = DistributionComponent(pm.Normal, mu=np.ones((tdf['b'].nunique())), sigma=sigma, group='b', name='distb')
    c = DistributionComponent(pm.Normal, mu=np.ones((tdf['c'].nunique())), sigma=sigma, group='c', name='distc')
    d = DistributionComponent(pm.Normal, mu=np.ones((tdf['d'].nunique())), sigma=sigma, group='d', name='distd')
    e = DistributionComponent(pm.Normal, mu=np.ones((tdf['e'].nunique())), sigma=sigma, group='e', name='diste')
    f = DistributionComponent(pm.Normal, mu=np.ones((tdf['f'].nunique())), sigma=sigma, group='f', name='distf')

    ab = DistributionComponent(pm.Normal, mu=a + b, sigma=sigma, name='ab')
    ac = DistributionComponent(pm.Normal, mu=a + c, sigma=sigma, name='ac')
    ef = DistributionComponent(pm.Normal, mu=e + f, sigma=sigma, name='ef')
    cd = DistributionComponent(pm.Normal, mu=c + d, sigma=sigma, name='cd')
    dd = DistributionComponent(pm.Normal, mu=d + d, sigma=sigma, name='dd')

    abcd = DistributionComponent(pm.Normal, mu=ab + cd, sigma=sigma, name='abcd')
    abac = DistributionComponent(pm.Normal, mu=a + b + a + c, sigma=sigma, name='abac')

    abce = DistributionComponent(pm.Normal, mu=a + b + c + e, sigma=sigma, name='abce')
    acef = DistributionComponent(pm.Normal, mu=a + c + e + f, sigma=sigma, name='acef')
    acef2 = DistributionComponent(pm.Normal, mu=ac + ef, sigma=sigma, name='acef2')
    abceacef = DistributionComponent(pm.Normal, mu=abce + acef, sigma=sigma, name='abceacef')

    with pm.Model(coords=gs.coords()):
        dd.build(gs)
        assert pm.draw(dd.variable).shape == (3,)

        acef.build(gs)
        assert pm.draw(acef.variable).shape == (2, 2, 2)
        assert pm.draw(acef.variable) == pytest.approx(4 * np.ones((2, 2, 2)))
        acef2.build(gs)
        assert pm.draw(acef2.variable).shape == (2, 2, 2)
        assert pm.draw(acef2.variable) == pytest.approx(4 * np.ones((2, 2, 2)))
        abce.build(gs)
        assert pm.draw(abce.variable).shape == (2, 2, 2)
        abceacef.build(gs)
        assert pm.draw(abceacef.variable).shape == (2, 2, 2)

        abac.build(gs)
        assert pm.draw(abac.variable).shape == (2, 2)

        abcd.build(gs)
        sample = pm.draw(abcd.variable)
        assert sample.shape == (3, 2) or sample.shape == (2, 3)

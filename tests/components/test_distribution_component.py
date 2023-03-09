import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy.stats.distributions import norm, binom

from sakkara.model import FixedValueComponent as FVC, DistributionComponent as DC, build
from sakkara.relation.groupset import init


def test_retrieve_groups():
    hv = DC(pm.Normal, group='room', mu=DC(pm.Normal, group='building', mu=DC(pm.Normal)))
    assert all(c in hv.retrieve_groups() for c in ['room', 'building'])


@pytest.mark.usefixtures('simple_df')
def test_build_variables(simple_df):
    groupset = init(simple_df)
    rv_global = DC(pm.Normal, mu=0)
    rv_building = DC(pm.Normal, group='building', mu=rv_global)
    rv_sensor = DC(pm.Normal, group='sensor', mu=rv_building, name='rv')

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


@pytest.mark.usefixtures('simple_df')
def test_twin_groups(simple_df):
    a = DC(pm.Normal, group='obs', mu=FVC(np.arange(len(simple_df)), group='time_2'), sigma=1e-15, name='a')

    _ = build(simple_df, a)

    assert all(x == pytest.approx(y) for x, y in zip(pm.draw(a.variable), np.arange(len(simple_df))))


@pytest.mark.usefixtures('simple_df')
def test_tuple_column(simple_df):
    groupset = init(simple_df)
    sigma = 1e-15
    rv_time = DC(pm.Normal, mu=FVC(np.arange(5), group='time'), sigma=sigma,
                 group='time')
    rv_sensor = DC(pm.Normal, mu=FVC(np.arange(4) * 10, group='sensor'), sigma=sigma,
                   group='sensor')
    rv_combined = DC(pm.Normal, mu=rv_sensor + rv_time, sigma=sigma, name='combined')

    rv_tuple_col = DC(pm.Normal,
                      mu=FVC(
                          np.sum(np.meshgrid(np.arange(5), np.arange(4) * 10), axis=0),
                          group=('sensor', 'time')),
                      sigma=sigma,
                      group=('time', 'sensor'),
                      name='tuple')

    rv_sum_1 = rv_tuple_col + rv_combined
    rv_sum_2 = rv_combined + rv_tuple_col

    rv_obs = rv_sum_1 + rv_sum_2 + DC(pm.Normal, sigma=sigma, group='obs', name='observed')
    with pm.Model(coords=groupset.coords()):
        rv_obs.build(groupset)

    assert rv_combined.representation != rv_tuple_col.representation
    assert pm.draw(rv_time.variable).shape == (5,)
    assert pm.draw(rv_sensor.variable).shape == (4,)
    assert pm.draw(rv_tuple_col.variable).shape == (5, 4)
    assert pm.draw(rv_combined.variable).shape == (4, 5)
    assert pm.draw(rv_sum_1.variable).shape == (5, 4)
    assert pm.draw(rv_sum_2.variable).shape == (4, 5)
    assert pm.draw(rv_obs.variable).shape == (20,)

    time_codes, _ = pd.factorize(simple_df['time'])
    sensor_codes, _ = pd.factorize(simple_df['sensor'])
    assert pm.draw(rv_obs.variable) == pytest.approx(
        4 * np.arange(5)[time_codes] + 4 * (np.arange(4) * 10)[sensor_codes])


@pytest.mark.usefixtures('simple_df')
def test_one_to_one_groups(simple_df):
    groupset = init(simple_df)
    x = DC(pm.Normal, mu=DC(pm.Normal, group='obs'), group='obs', name='x')
    y = DC(pm.Normal, mu=DC(pm.Normal, group='time_2'), group='obs', name='y')
    ll = DC(pm.Normal, mu=DC(pm.Normal, group='time_2'),
            observed=np.random.randn(20),
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
    a = DC(pm.Normal, mu=np.ones((tdf['a'].nunique())), sigma=sigma, group='a', name='dista')
    b = DC(pm.Normal, mu=np.ones((tdf['b'].nunique())), sigma=sigma, group='b', name='distb')
    c = DC(pm.Normal, mu=np.ones((tdf['c'].nunique())), sigma=sigma, group='c', name='distc')
    d = DC(pm.Normal, mu=np.ones((tdf['d'].nunique())), sigma=sigma, group='d', name='distd')
    e = DC(pm.Normal, mu=np.ones((tdf['e'].nunique())), sigma=sigma, group='e', name='diste')
    f = DC(pm.Normal, mu=np.ones((tdf['f'].nunique())), sigma=sigma, group='f', name='distf')

    ab = DC(pm.Normal, mu=a + b, sigma=sigma, name='ab')
    ac = DC(pm.Normal, mu=a + c, sigma=sigma, name='ac')
    ef = DC(pm.Normal, mu=e + f, sigma=sigma, name='ef')
    cd = DC(pm.Normal, mu=c + d, sigma=sigma, name='cd')
    dd = DC(pm.Normal, mu=d + d, sigma=sigma, name='dd')

    abcd = DC(pm.Normal, mu=ab + cd, sigma=sigma, name='abcd')
    abac = DC(pm.Normal, mu=a + b + a + c, sigma=sigma, name='abac')

    abce = DC(pm.Normal, mu=a + b + c + e, sigma=sigma, name='abce')
    acef = DC(pm.Normal, mu=a + c + e + f, sigma=sigma, name='acef')
    acef2 = DC(pm.Normal, mu=ac + ef, sigma=sigma, name='acef2')
    abceacef = DC(pm.Normal, mu=abce + acef, sigma=sigma, name='abceacef')

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

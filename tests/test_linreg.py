import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
import arviz as az
import pymc as pm

from sakkara.model.utils import build, Likelihood, data_components
from sakkara.model.components import Distribution


@pytest.fixture
def df():
    rng = default_rng(100)

    N = 5

    heating_power = np.repeat(1, N)
    outdoor_temperature = np.cos(np.linspace(0, 2 * np.pi, N)) - 1 + rng.uniform(-1, 1, N)
    time = np.arange(N)

    df = pd.DataFrame({'building': np.repeat(['a', 'b'], 2 * N),
                       'room': np.repeat(['a1', 'a2', 'b1', 'b2'], N),
                       'time': np.tile(time, 4),
                       'heating_power': np.tile(heating_power, 4),
                       'outdoor_temperature': np.tile(outdoor_temperature, 4),
                       'indoor_temperature': 20
                       })

    df.loc[df['room'] == 'a1', 'indoor_temperature'] += (
            heating_power * 1 + outdoor_temperature * 1).cumsum()
    df.loc[df['room'] == 'a2', 'indoor_temperature'] += (
            heating_power * .8 + outdoor_temperature * 1).cumsum()
    df.loc[df['room'] == 'b1', 'indoor_temperature'] += (
            heating_power * .6 + outdoor_temperature * .4).cumsum()
    df.loc[df['room'] == 'b2', 'indoor_temperature'] += (
            heating_power * .4 + outdoor_temperature * .4).cumsum()

    df['y'] = df.groupby('room')['indoor_temperature'].diff()
    df = df.dropna()

    return df


@pytest.fixture
def likelihood(df):
    coeff = Distribution(pm.Normal,
                         name='outdoor_temperature',
                         column='building',
                         mu=Distribution(
                             pm.Normal
                         ),
                         sigma=Distribution(
                             pm.Exponential,
                             lam=1000
                         )
                         )
    intercept = Distribution(pm.Normal,
                             name='heating_power',
                             column='room',
                             mu=Distribution(
                                 pm.Normal,
                                 column='building',
                                 mu=Distribution(pm.Normal),
                                 sigma=Distribution(pm.Exponential, lam=10)
                             ),
                             sigma=Distribution(pm.Exponential, lam=1000)
                             )

    data = data_components(df)

    likelihood = Likelihood(pm.Normal, mu=coeff * data['outdoor_temperature'] + intercept,
                            sigma=Distribution(pm.Exponential, lam=1000),
                            data=data['y'])
    return likelihood


def test_minimal_linreg():
    x = np.random.randn(10)
    group = np.repeat([0, 1], 5)
    k = np.array([1, -1])
    y = x * k[group] + np.random.randn() * 1e-3
    df = pd.DataFrame({'x': x, 'y': y, 'group': group})

    data = data_components(df)
    coeff = Distribution(pm.Normal, name='x', column='group', mu=Distribution(pm.Normal),
                         sigma=Distribution(pm.Exponential, lam=1))
    intercept = Distribution(pm.Normal, name='intercept')

    likelihood = Likelihood(pm.Normal, name='est', mu=coeff * data['x'] + intercept,
                            sigma=Distribution(pm.Exponential, lam=1),
                            data=data['y'])

    built_model = build(df, likelihood)
    with built_model:
        idata = pm.fit(1, 'advi')
    assert isinstance(built_model, pm.Model)


def test_sampling(likelihood, df):
    with build(df, likelihood):
        idata = pm.fit(100000, method='advi', random_seed=1000).sample(1000)

    result = az.summary(idata)
    assert pytest.approx(result.loc['mu_outdoor_temperature', 'mean'], abs=5e-2) == .7
    assert pytest.approx(result.loc['outdoor_temperature[a]', 'mean']) == 1.
    assert pytest.approx(result.loc['outdoor_temperature[b]', 'mean']) == .4
    assert pytest.approx(result.loc['mu_mu_heating_power', 'mean'], abs=5e-2) == .7
    assert pytest.approx(result.loc['mu_heating_power[a]', 'mean'], abs=5e-2) == .9
    assert pytest.approx(.9, result.loc['mu_heating_power[b]', 'mean'], abs=5e-2) == .5
    assert pytest.approx(result.loc['heating_power[a1]', 'mean']) == 1.
    assert pytest.approx(result.loc['heating_power[a2]', 'mean']) == .8
    assert pytest.approx(result.loc['heating_power[b1]', 'mean']) == .6
    assert pytest.approx(result.loc['heating_power[b2]', 'mean']) == .4
    assert pytest.approx(result.loc['sigma_mu_heating_power', 'mean'], abs=5e-2) == .2

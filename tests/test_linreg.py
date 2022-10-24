import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
import arviz as az
import pymc as pm

from sakkara.model.composite import CompositeComponent
from sakkara.model.utils import build, Likelihood, data_components
from sakkara.model.components import Distribution, FixedComponent


@pytest.fixture
def df():
    def rng(seed=100): return default_rng(seed)

    N = 5

    heating_power = np.repeat(1, N)
    outdoor_temperature = np.cos(np.linspace(0, 2 * np.pi, N)) - 1 + rng().uniform(-1, 1, N)
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


def test_build_linreg_model(df, likelihood):
    assert isinstance(likelihood, Likelihood)

    assert isinstance(likelihood['mu'], CompositeComponent)

    assert isinstance(likelihood['mu'].a, CompositeComponent)

    assert isinstance(likelihood['mu'].a.a, Distribution)
    assert isinstance(likelihood['mu'].a.a['mu'], Distribution)
    assert isinstance(likelihood['mu'].a.a['sigma'], Distribution)
    assert isinstance(likelihood['mu'].a.a['sigma']['lam'], FixedComponent)

    assert isinstance(likelihood['mu'].a.b, Distribution)
    assert isinstance(likelihood['mu'].a.b['value'], FixedComponent)

    assert isinstance(likelihood['mu'].b, Distribution)
    assert isinstance(likelihood['mu'].b['mu'], Distribution)
    assert isinstance(likelihood['mu'].b['sigma'], Distribution)
    assert isinstance(likelihood['mu'].b['mu']['mu'], Distribution)
    assert isinstance(likelihood['mu'].b['mu']['sigma'], Distribution)
    assert isinstance(likelihood['mu'].b['mu']['sigma']['lam'], FixedComponent)

    built_model = build(df, likelihood)
    assert isinstance(built_model, pm.Model)

    assert pm.draw(likelihood.variable).shape == (16,)

    assert pm.draw(likelihood['mu'].variable).shape == (16,)
    assert pm.draw(likelihood['mu'].a.variable).shape == (16,)
    assert pm.draw(likelihood['mu'].a.b.variable).shape == (16,)
    assert pm.draw(likelihood['mu'].a.a.variable).shape == (2,)
    assert pm.draw(likelihood['mu'].a.a['mu'].variable).shape == ()
    assert pm.draw(likelihood['mu'].a.a['sigma'].variable).shape == ()
    assert pm.draw(likelihood['mu'].b['mu'].variable).shape == (2,)
    assert pm.draw(likelihood['mu'].b['sigma'].variable).shape == ()
    assert pm.draw(likelihood['mu'].b.variable).shape == (4,)
    assert pm.draw(likelihood['mu'].b['mu']['mu'].variable).shape == ()
    assert pm.draw(likelihood['mu'].b['mu']['sigma'].variable).shape == ()

    assert pm.draw(likelihood['sigma'].variable).shape == ()


def test_rebuild_model(df, likelihood):
    with build(df, likelihood):
        saved_variable = likelihood.variable
        _ = pm.fit(1, method='advi')
    with build(df, likelihood):
        _ = pm.fit(1, method='advi')
        assert likelihood.variable != saved_variable


def test_sampling(df, likelihood):
    with build(df, likelihood):
        idata = pm.fit(50000, method='advi', random_seed=1000).sample(1000, random_seed=1000)

    result = az.summary(idata)
    assert pytest.approx(result.loc['mu_outdoor_temperature', 'mean'], abs=5e-2) == .7
    assert pytest.approx(result.loc['outdoor_temperature[a]', 'mean']) == 1.
    assert pytest.approx(result.loc['outdoor_temperature[b]', 'mean']) == .4
    assert pytest.approx(result.loc['mu_mu_heating_power', 'mean'], abs=5e-2) == .7
    assert pytest.approx(result.loc['mu_heating_power[a]', 'mean'], abs=5e-2) == .9
    assert pytest.approx(result.loc['mu_heating_power[b]', 'mean'], abs=5e-2) == .5
    assert pytest.approx(result.loc['heating_power[a1]', 'mean']) == 1.
    assert pytest.approx(result.loc['heating_power[a2]', 'mean']) == .8
    assert pytest.approx(result.loc['heating_power[b1]', 'mean']) == .6
    assert pytest.approx(result.loc['heating_power[b2]', 'mean']) == .4
    assert pytest.approx(result.loc['sigma_mu_heating_power', 'mean'], abs=5e-2) == .2

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
import arviz as az
import pymc as pm

from sakkara.model import build, FunctionComponent, RandomVariable, SeriesComponent, FixedComponent, Likelihood, \
    data_components

N = 2


@pytest.fixture
def df():
    def rng(seed=100): return default_rng(seed)

    df = pd.DataFrame({'g1': np.repeat(['a', 'b'], 2 * N),
                       'g2': np.repeat(['a1', 'a2', 'b1', 'b2'], N),
                       'x': rng().normal(0, 1, 4 * N),
                       'y': 0
                       })

    df.loc[df['g2'] == 'a1', 'y'] = 1 + df.loc[df['g2'] == 'a1', 'x'] * 1
    df.loc[df['g2'] == 'a2', 'y'] = .8 + df.loc[df['g2'] == 'a2', 'x'] * 1
    df.loc[df['g2'] == 'b1', 'y'] = .6 + df.loc[df['g2'] == 'b1', 'x'] * .4
    df.loc[df['g2'] == 'b2', 'y'] = .4 + df.loc[df['g2'] == 'b2', 'x'] * .4

    return df


@pytest.fixture
def likelihood(df):
    coeff = RandomVariable(pm.Normal,
                           name='coeff',
                           columns='g1',
                           mu=RandomVariable(
                               pm.Normal
                           ),
                           sigma=RandomVariable(pm.HalfNormal, sigma=1e-10)
                           )

    intercept = RandomVariable(pm.Normal,
                               name='intercept',
                               columns='g2',
                               mu=RandomVariable(
                                   pm.Normal,
                                   columns='g1',
                                   mu=RandomVariable(pm.Normal),
                                   sigma=RandomVariable(pm.HalfNormal, sigma=1e-5)
                               ),
                               sigma=RandomVariable(pm.HalfNormal, sigma=1e-10)
                               )

    data = data_components(df)

    return Likelihood(pm.Normal, mu=coeff * data['x'] + intercept, sigma=1e-15, obs_data=data['y'])


def test_build_linreg_model(df, likelihood):
    assert isinstance(likelihood, Likelihood)

    assert isinstance(likelihood['mu'], FunctionComponent)

    assert isinstance(likelihood['mu'].args[0], FunctionComponent)

    assert isinstance(likelihood['mu'].args[0].args[0], RandomVariable)
    assert isinstance(likelihood['mu'].args[0].args[0]['mu'], RandomVariable)
    assert isinstance(likelihood['mu'].args[0].args[0]['sigma'], RandomVariable)
    assert isinstance(likelihood['mu'].args[0].args[0]['sigma']['sigma'], FixedComponent)

    assert isinstance(likelihood['mu'].args[0].args[1], SeriesComponent)

    assert isinstance(likelihood['mu'].args[1], RandomVariable)
    assert isinstance(likelihood['mu'].args[1]['mu'], RandomVariable)
    assert isinstance(likelihood['mu'].args[1]['sigma'], RandomVariable)
    assert isinstance(likelihood['mu'].args[1]['mu']['mu'], RandomVariable)
    assert isinstance(likelihood['mu'].args[1]['mu']['sigma'], RandomVariable)
    assert isinstance(likelihood['mu'].args[1]['mu']['sigma']['sigma'], FixedComponent)

    assert isinstance(likelihood['sigma'], FixedComponent)

    built_model = build(df, likelihood)
    assert isinstance(built_model, pm.Model)

    assert pm.draw(likelihood.variable).shape == (N * 4,)

    assert pm.draw(likelihood['mu'].variable).shape == (N * 4,)
    assert pm.draw(likelihood['mu'].args[0].variable).shape == (N * 4,)
    assert likelihood['mu'].args[0].args[1].variable.shape == (N * 4,)
    assert pm.draw(likelihood['mu'].args[0].args[0].variable).shape == (2,)
    assert pm.draw(likelihood['mu'].args[0].args[0]['mu'].variable).shape == (1,)
    assert pm.draw(likelihood['mu'].args[0].args[0]['sigma'].variable).shape == (1,)
    assert pm.draw(likelihood['mu'].args[1]['mu'].variable).shape == (2,)
    assert pm.draw(likelihood['mu'].args[1]['sigma'].variable).shape == (1,)
    assert pm.draw(likelihood['mu'].args[1].variable).shape == (4,)
    assert pm.draw(likelihood['mu'].args[1]['mu']['mu'].variable).shape == (1,)
    assert pm.draw(likelihood['mu'].args[1]['mu']['sigma'].variable).shape == (1,)


def test_rebuild_model(df, likelihood):
    with build(df, likelihood):
        saved_variable = likelihood.variable
        _ = pm.fit(1, method='advi')

    with pytest.raises(TypeError):
        with build(df, likelihood):
            _ = pm.fit(1, method='advi')

    likelihood.clear()
    with build(df, likelihood):
        _ = pm.fit(1, method='advi')
        assert likelihood.variable != saved_variable


def test_sampling(df, likelihood):
    with build(df, likelihood):
        approx = pm.fit(n=10000, method='advi', random_seed=1000)

    idata = approx.sample(1000, random_seed=1000)

    result = az.summary(idata)
    assert pytest.approx(result.loc['mu_coeff[global]', 'mean'], abs=5e-2) == .7
    assert pytest.approx(result.loc['coeff[a]', 'mean'], abs=5e-2) == 1.
    assert pytest.approx(result.loc['coeff[b]', 'mean'], abs=5e-2) == .4
    assert pytest.approx(result.loc['mu_mu_intercept[global]', 'mean'], abs=5e-2) == .7
    assert pytest.approx(result.loc['mu_intercept[a]', 'mean'], abs=5e-2) == .9
    assert pytest.approx(result.loc['mu_intercept[b]', 'mean'], abs=5e-2) == .5
    assert pytest.approx(result.loc['intercept[a1]', 'mean'], abs=5e-2) == 1.
    assert pytest.approx(result.loc['intercept[a2]', 'mean'], abs=5e-2) == .8
    assert pytest.approx(result.loc['intercept[b1]', 'mean'], abs=5e-2) == .6
    assert pytest.approx(result.loc['intercept[b2]', 'mean'], abs=5e-2) == .4

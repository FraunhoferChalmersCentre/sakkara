import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
import arviz as az
import pymc as pm

from sakkara.model import build, FunctionComponent, DistributionComponent, DataComponent, Likelihood, data_components, \
    UnrepeatableComponent

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
    coeff = DistributionComponent(pm.Normal,
                                  name='coeff',
                                  group='g1',
                                  mu=DistributionComponent(
                                      pm.Normal
                                  ),
                                  sigma=DistributionComponent(pm.HalfNormal, sigma=1e-10)
                                  )

    intercept = DistributionComponent(pm.Normal,
                                      name='intercept',
                                      group='g2',
                                      mu=DistributionComponent(
                                          pm.Normal,
                                          group='g1',
                                          mu=DistributionComponent(pm.Normal),
                                          sigma=DistributionComponent(pm.HalfNormal, sigma=1e-5)
                                      ),
                                      sigma=DistributionComponent(pm.HalfNormal, sigma=1e-10)
                                      )

    data = data_components(df)

    return Likelihood(pm.Normal, mu=coeff * data['x'] + intercept, sigma=1e-15, observed=data['y'])


def test_build_linreg_model(df, likelihood):
    assert isinstance(likelihood, Likelihood)

    assert isinstance(likelihood['mu'], FunctionComponent)

    assert isinstance(likelihood['mu'].args[0], FunctionComponent)

    assert isinstance(likelihood['mu'].args[0].args[0], DistributionComponent)
    assert isinstance(likelihood['mu'].args[0].args[0]['mu'], DistributionComponent)
    assert isinstance(likelihood['mu'].args[0].args[0]['sigma'], DistributionComponent)
    assert isinstance(likelihood['mu'].args[0].args[0]['sigma']['sigma'], UnrepeatableComponent)

    assert isinstance(likelihood['mu'].args[0].args[1], DataComponent)

    assert isinstance(likelihood['mu'].args[1], DistributionComponent)
    assert isinstance(likelihood['mu'].args[1]['mu'], DistributionComponent)
    assert isinstance(likelihood['mu'].args[1]['sigma'], DistributionComponent)
    assert isinstance(likelihood['mu'].args[1]['mu']['mu'], DistributionComponent)
    assert isinstance(likelihood['mu'].args[1]['mu']['sigma'], DistributionComponent)
    assert isinstance(likelihood['mu'].args[1]['mu']['sigma']['sigma'], UnrepeatableComponent)

    assert isinstance(likelihood['sigma'], UnrepeatableComponent)

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

    with pytest.raises(ValueError):
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

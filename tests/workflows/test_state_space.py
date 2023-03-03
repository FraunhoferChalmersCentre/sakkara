import pytest
import pymc as pm
import numpy as np
import pandas as pd
from numpy.random import default_rng
from numpy import testing
import pytensor.tensor as pt

from sakkara.model import DistributionComponent, data_components, GroupComponent, Likelihood, FunctionComponent, \
    DeterministicComponent
from sakkara.model.utils import build

N = 20

SIGMA = 1e-15


def rng(seed=100): return default_rng(seed)


@pytest.fixture
def parallel_process_df():
    k = 1.005
    x0 = 1
    x = x0 * np.power(k, np.arange(N))

    epsilon = rng().normal(0, 1, N)
    random_process = np.array([np.sum(np.power(k, i - np.arange(i)) * epsilon[:i]) for i in range(N)])

    latent_process = x + random_process

    measure_error = np.array([0, 1])

    y = (latent_process.reshape(-1, 1) + rng().normal(0, measure_error, size=(N, len(measure_error)))).T.flatten()

    timesteps = pd.date_range("2018-01-01 00:00:00", tz='utc', periods=N, freq="H")
    full_df = pd.DataFrame(
        {
            'time': np.tile(timesteps, len(measure_error)),
            'group': np.repeat(np.arange(len(measure_error)), N),
            'y': y
        })
    return full_df.loc[
        (full_df.group == 1) | (np.isin(full_df.time, timesteps[np.linspace(0, N - 1, 3, endpoint=True, dtype=int)]))]


@pytest.fixture
def partially_observed_df():
    true_k = 2
    time = np.arange(N)
    all_x = rng().normal(size=N)
    true_state = (all_x * true_k).cumsum()
    all_y = true_state + rng().normal(scale=SIGMA, size=N)

    df = pd.DataFrame({'x': all_x, 'y': all_y, 'time': time}, index=time)

    random_sel = rng(10).choice([True, False], size=len(df), p=[.9, .1])

    df['sample'] = df['y'].copy()
    df.loc[random_sel, 'sample'] = float('nan')
    return df


def test_state_space_model(parallel_process_df):
    R = GroupComponent(
        name='R',
        group='group',
        membercomponents={
            0: 1e-3,
            1: DistributionComponent(pm.Exponential, lam=10)
        }
    )

    data = data_components(parallel_process_df)

    X = DistributionComponent(pm.GaussianRandomWalk, name='X', group='time')

    likelihood = Likelihood(pm.Normal, mu=X, sigma=R, observed=data['y'])

    built_model = build(parallel_process_df, likelihood)
    assert pm.draw(X.variable).shape == (20,)
    assert pm.draw(R.variable).shape == (2,)
    assert pm.draw(likelihood.variable).shape == (23,)


def test_partially_observed_state_space(partially_observed_df):
    dc = data_components(partially_observed_df)

    k = DistributionComponent(pm.Normal, name='diff')

    state = DeterministicComponent('state', FunctionComponent(pt.cumsum, k * dc['x']))

    likelihood = Likelihood(pm.Normal,
                            mu=state,
                            sigma=SIGMA,
                            observed=dc['sample'],
                            nan_data_mask=0,
                            nan_param_mask={'mu': 0, 'sigma': 1}
                            )

    built_model = build(partially_observed_df, likelihood)
    approx = pm.fit(model=built_model, method='advi', random_seed=100)

    assert not np.any(np.isnan(approx.hist))

    means = approx.sample(1000, random_seed=100)['posterior']['state'].to_dataframe().groupby(level=2).mean()

    testing.assert_allclose(means['state'], partially_observed_df['y'], atol=.1)

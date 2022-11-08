import pytest
import pymc as pm
import numpy as np
import pandas as pd
from numpy.random import default_rng
from numpy import testing

from sakkara.model import RandomVariable, data_components, GroupComponent, Likelihood
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
        columns='group',
        components={
            0: 0,
            1: RandomVariable(pm.Normal, columns='time', sigma=RandomVariable(pm.Exponential, lam=1))
        }
    )

    data = data_components(parallel_process_df)

    x_sigma = RandomVariable(pm.Exponential, lam=1)

    timesteps = parallel_process_df['time'].unique()

    X = GroupComponent(name='X', columns='time')
    X.add(timesteps[0], RandomVariable(pm.Normal, sigma=x_sigma))

    for i in range(1, N):
        X.add(timesteps[i], RandomVariable(pm.Normal, mu=X[timesteps[i - 1]], sigma=x_sigma))

    likelihood = Likelihood(pm.Normal, mu=X + R, sigma=RandomVariable(pm.Exponential, lam=10), obs_data=data['y'])

    built_model = build(parallel_process_df, likelihood)
    assert pm.draw(X.variable).shape == (20,)
    assert pm.draw(R.variable).shape == (2, 20)
    assert pm.draw(likelihood.variable).shape == (23,)


def test_partially_observed_state_space(partially_observed_df):
    k = RandomVariable(pm.Uniform, lower=0, upper=4, name='k')
    dc = data_components(partially_observed_df)

    state = GroupComponent(columns='obs', name='state')
    state.add(0, k * partially_observed_df.loc[0, 'x'])
    for i in range(1, len(partially_observed_df)):
        state.add(i, state[i - 1] + k * partially_observed_df.loc[i, 'x'])

    likelihood = Likelihood(pm.Normal, mu=state, sigma=SIGMA, obs_data=dc['sample'], nan_data_mask=1,
                            nan_var_mask={'mu': 0, 'sigma': 1})

    built_model = build(partially_observed_df, likelihood)
    approx = pm.fit(model=built_model, method='advi',
                    callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute")], random_seed=100)

    assert not np.any(np.isnan(approx.hist))
    posterior = approx.sample(1000)['posterior']['state'].to_dataframe()
    means = []
    for o in range(len(partially_observed_df)):
        means.append(posterior.loc[:, :, o].mean().values)

    means = np.array(means).squeeze()

    testing.assert_allclose(means, partially_observed_df['y'], rtol=1e-3)

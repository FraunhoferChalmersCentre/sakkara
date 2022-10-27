import pytest
import pymc as pm
import numpy as np
import pandas as pd
from numpy.random import default_rng

from sakkara.model.components import Distribution, Deterministic, Concat
from sakkara.model.utils import Likelihood, data_components, build

N = 20


@pytest.fixture
def df():
    def rng(seed=100): return default_rng(seed)

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


def test_state_space_model(df):
    R = Concat(
        name='R',
        columns='group',
        components={
            0: 0,
            1: Distribution(pm.Normal, columns='time', sigma=Distribution(pm.Exponential, lam=1))
        }
    )

    data = data_components(df)

    x_sigma = Distribution(pm.Exponential, lam=1)

    timesteps = df['time'].unique()

    X = Concat(name='X', columns='time')
    X.add(timesteps[0], Distribution(pm.Normal, sigma=x_sigma))

    for i in range(1, N):
        X.add(timesteps[i], Distribution(pm.Normal, mu=X[timesteps[i - 1]], sigma=x_sigma))

    likelihood = Likelihood(pm.Normal, mu=X + R, sigma=Distribution(pm.Exponential, lam=10), data=data['y'])

    built_model = build(df, likelihood)
    assert pm.draw(X.variable).shape == (20,)
    assert pm.draw(R.variable).shape == (2, 20)
    assert pm.draw(likelihood.variable).shape == (23,)

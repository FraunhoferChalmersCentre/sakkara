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
    full_df = pd.DataFrame(
        {
            'time': np.tile(np.arange(N), len(measure_error)),
            'group': np.repeat(np.arange(len(measure_error)), N),
            'y': y
        })
    return full_df.loc[
        (full_df.group == 1) | (np.isin(full_df.time, np.linspace(0, N - 1, 3, endpoint=True, dtype=int)))]


def test_state_space_model(df):
    R = Concat(
        name='R',
        column='group',
        components={
            0: Deterministic(0.),
            1: Distribution(pm.Normal, column='time', sigma=Distribution(pm.Exponential, lam=1))
        }
    )

    data = data_components(df)

    x_sigma = Distribution(pm.Exponential, lam=1)

    X = Concat(name='X', column='time')
    X.add(0, Distribution(pm.Normal, sigma=x_sigma))

    for i in range(1, N):
        X.add(i, Distribution(pm.Normal, mu=X[i - 1], sigma=x_sigma))

    likelihood = Likelihood(pm.Normal, mu=X + R, sigma=Distribution(pm.Exponential, lam=10), data=data['y'])

    built_model = build(df, likelihood)
    assert len(pm.draw(X.variable)) == 20
    assert len(pm.draw(R.variable)) == 23
    assert len(pm.draw(likelihood.variable)) == 23

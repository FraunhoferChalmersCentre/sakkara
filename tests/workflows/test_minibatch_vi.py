import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng
import pymc as pm

from sakkara.model import data_components, DistributionComponent as DC, MinibatchLikelihood, build

rng = default_rng(100)


@pytest.fixture
def udf():
    return pd.DataFrame({'time': np.arange(30), 'u': rng.normal(0, 1, 30)})


@pytest.fixture
def xdf(udf):
    xdf = pd.DataFrame({'g': ['a', 'b'], 'k': [-1, 1]})

    merged = xdf.merge(udf, how='cross')
    merged['y'] = merged['u'] * merged['k']
    merged['obs'] = np.arange(len(merged))

    return merged


def test_hierarchical_minibatch(udf, xdf):
    udc = data_components(udf, 'time')
    xdc = data_components(xdf)

    k = DC(pm.Normal, name='k', group='g')
    predicted = k * udc['u']

    likelihood = MinibatchLikelihood(pm.Normal, xdc['y'], batch_size=5, mu=predicted, sigma=1e-15)

    with build(xdf, likelihood):
        approx = pm.fit(n=10000, random_seed=100)

    idata = approx.sample(10000, random_seed=100)

    k_posterior = idata.posterior['k'].to_dataframe().reset_index()
    assert k_posterior.loc[k_posterior.g == 'a', 'k'].mean() == pytest.approx(-1, abs=1e-2)
    assert k_posterior.loc[k_posterior.g == 'b', 'k'].mean() == pytest.approx(1, abs=1e-2)

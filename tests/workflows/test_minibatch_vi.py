import pytest
import pymc as pm

from sakkara.model import data_components, DistributionComponent as DC, MinibatchLikelihood, build, \
    DeterministicComponent


@pytest.mark.usefixtures('udf', 'xdf')
def test_hierarchical_minibatch(udf, xdf):
    udc = data_components(udf, 'time')
    xdc = data_components(xdf)

    k = DC(pm.Normal, name='k', group='g')
    predicted = DeterministicComponent('predicted', k * udc['u'])
    mbl = MinibatchLikelihood(pm.Normal, observed=xdc['y'], batch_size=10, mu=predicted, sigma=1e-15)

    model = build(xdf, mbl)
    approx = pm.fit(model=model, n=10000, random_seed=100)

    idata = approx.sample(10000, random_seed=100)

    k_posterior = idata.posterior['k'].to_dataframe().reset_index()
    assert k_posterior.loc[k_posterior.g == 'a', 'k'].mean() == pytest.approx(-1, abs=1e-2)
    assert k_posterior.loc[k_posterior.g == 'b', 'k'].mean() == pytest.approx(1, abs=1e-2)

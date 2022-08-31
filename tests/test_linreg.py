import numpy as np
import pandas as pd
import pytest
import pymc as pm

from hierlinreg.hierarchical import HierarchicalVariable, Likelihood
from hierlinreg.linreg import retrieve_group_names, init_model


@pytest.fixture
def df():
    df = pd.DataFrame(
        {
            'building': np.repeat(list('ab'), 10),
            'sensor': np.repeat(list('stuv'), 5),
            'time': np.tile(np.arange(5), 4),
            'a': np.random.randn(20),
            'b': np.random.randn(20),
            'c': np.random.randn(20)
        })

    bidx, _ = df['building'].factorize()
    sidx, _ = df['sensor'].factorize()

    df['value'] = bidx * 100 + sidx * 10 + np.random.randn(20)

    return df


def test_retrieve_groups():
    rv_global = HierarchicalVariable(pm.Normal, mu=0)
    rv_building = HierarchicalVariable(pm.Normal, group_name='building', mu=rv_global)
    rv_sensor = HierarchicalVariable(pm.Normal, group_name='sensor', mu=rv_building)

    group_cols, coeff_cols = retrieve_group_names({'a': rv_sensor})
    assert all(n in group_cols for n in ['sensor', 'building', 'global'])
    assert coeff_cols == ['a']


def test_create_linear_model(df: pd.DataFrame):
    model_spec = {
        'a': HierarchicalVariable(pm.Normal, 'sensor', mu=HierarchicalVariable(pm.Normal),
                                  sigma=HierarchicalVariable(pm.HalfCauchy, beta=1)),
        'b': HierarchicalVariable(pm.Normal, 'sensor',
                                  mu=HierarchicalVariable(pm.Normal, 'building',
                                                          mu=HierarchicalVariable(pm.Normal))
                                  )
    }

    likelihood = Likelihood(pm.Normal, 'mu', sigma=HierarchicalVariable(pm.Exponential, lam=1))

    model = init_model(df, 'c', model_spec, likelihood)
    assert pm.draw(model.a_sensor).shape == (4,)
    assert pm.draw(model.mu_a_global).shape == (1,)
    assert pm.draw(model.sigma_a_global).shape == (1,)
    assert pm.draw(model.b_sensor).shape == (4,)
    assert pm.draw(model.mu_b_building).shape == (2,)
    assert pm.draw(model.mu_mu_b_global).shape == (1,)
    assert pm.draw(model.likelihood_observation).shape == (20,)
    assert pm.draw(model.sigma_likelihood_global).shape == (1,)

import numpy as np
import pytest
import pymc as pm

from sakkara.model import DistributionComponent as DC, build, Likelihood, DataComponent, \
    data_components


@pytest.mark.usefixtures('simple_df')
def test_basic_feature(simple_df):
    x = DC(pm.Uniform, lower=1, upper=1)

    ll = Likelihood(pm.Normal, mu=x, sigma=1e-15, observed=DataComponent(np.repeat(0, 20), group='obs'))

    _ = build(simple_df, ll)

    assert pm.draw(ll.variable).shape == (20,)
    assert all(l == pytest.approx(1.) for l in pm.draw(ll.variable))


@pytest.mark.usefixtures('simple_df')
def test_nan_data(simple_df):
    c = DC(pm.Uniform, lower=[0, 1, 2, 3, 4], upper=[0, 1, 2, 3, 4], group='time')

    simple_df['nan_obs'] = np.repeat(np.arange(5), 4)
    simple_df['nan_obs'].iloc[np.arange(4) * 6] = float('Nan')

    dc = data_components(simple_df)

    with pytest.raises(ValueError):
        ll = Likelihood(pm.Normal, mu=c, sigma=1e-15, observed=dc['nan_obs'])
        _ = build(simple_df, ll)

    ll = Likelihood(pm.Normal, mu=c, sigma=1e-15, observed=dc['nan_obs'], nan_param_mask={'mu': 0, 'sigma': 1e-15},
                    nan_data_mask=0)
    _ = build(simple_df, ll)

    assert pm.draw(ll.variable).shape == (20,)
    assert all(
        l == pytest.approx(i % 5) if i % 6 != 0 else l == pytest.approx(0) for i, l in enumerate(pm.draw(ll.variable)))

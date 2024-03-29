import pytest

import pymc as pm

from sakkara.model import data_components, DistributionComponent as DC, MinibatchLikelihood, \
    build, Likelihood, DeterministicComponent, FunctionComponent
from sakkara.model.deterministic import MinibatchDeterministic
from sakkara.model.minibatch import MinibatchComponent


@pytest.mark.usefixtures('udf', 'xdf')
def test_minibatch(udf, xdf):
    udc = data_components(udf, 'time')
    xdc = data_components(xdf)

    k = DC(pm.Normal, name='k', group='g')
    predicted = DeterministicComponent('predicted', k * udc['u'])
    mbl = MinibatchLikelihood(pm.Normal, observed=xdc['y'], batch_size=1, mu=predicted, sigma=1e-15)

    _ = build(xdf, mbl)

    assert mbl['total_size'].variable == 60

    assert pm.draw(mbl.variable).shape == (60,)

    assert pm.draw(mbl['mu'].variable).shape == (1,)
    assert isinstance(mbl['mu'], MinibatchDeterministic)
    assert isinstance(mbl['mu'].component, FunctionComponent)
    assert isinstance(mbl['mu'].component.args[0], MinibatchComponent)
    assert mbl['mu'].component.args[0].component == k
    assert isinstance(mbl['mu'].component.args[1], MinibatchComponent)
    assert mbl['mu'].component.args[1].component == udc['u']

    assert pm.draw(mbl['observed'].variable).shape == (1,)
    assert isinstance(mbl['observed'], MinibatchComponent)
    assert mbl['observed'].component == xdc['y']

    assert predicted.variable is None

    assert pm.draw(xdc['y'].variable).shape == (60,)

    assert pm.draw(udc['u'].variable).shape == (30,)

    assert pm.draw(k.variable).shape == (2,)

    ll = Likelihood(pm.Normal, observed=xdc['y'], mu=predicted, sigma=1e-15)
    ll.clear()
    _ = build(xdf, ll)

    assert pm.draw(ll['mu'].variable).shape == (2, 30)
    assert pm.draw(ll.variable).shape == (60,)
    assert pm.draw(predicted.variable).shape == (2, 30)
    assert pm.draw(k.variable).shape == (2,)

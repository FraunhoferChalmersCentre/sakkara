import pytest
import pymc as pm
import numpy as np

from sakkara.model import DistributionComponent as DC, DeterministicComponent as DetC, build, FixedValueComponent as FVC


@pytest.mark.usefixtures('simple_df')
def test_deterministic_component(simple_df):
    a = DC(pm.Normal, group='time', mu=FVC(np.arange(5), group='time'), sigma=1e-15)
    b = DC(pm.Normal, mu=1, sigma=1e-15)

    c = DetC('c', a + b)

    d = DC(pm.Normal, name='d', mu=c)

    _ = build(simple_df, d)

    assert a.get_name().split('<')[0] == 'var_c_'
    assert a.get_name().split('>')[-1] == '_arg0'
    assert b.get_name().split('<')[0] == 'var_c_'
    assert b.get_name().split('>')[-1] == '_arg1'
    assert str(c.node.representation()) == '{time}'
    assert all(x == pytest.approx(y) for x, y in zip(pm.draw(c.variable), np.arange(1, 6)))


@pytest.mark.usefixtures('simple_df')
def test_twin(simple_df):
    a = DC(pm.Normal, group='obs', mu=FVC(np.arange(len(simple_df)), group='time_2'), sigma=1e-15)
    b = DC(pm.Normal, group='time_2', mu=FVC(np.arange(10, 10 + len(simple_df)), group='obs'), sigma=1e-15)

    c = DetC('c', a + b)

    d = DC(pm.Normal, name='d', mu=c)

    _ = build(simple_df, d)

    assert c.dims() == ('obs', )
    assert all(x == pytest.approx(y) for x, y in
               zip(pm.draw(c.variable), np.arange(len(simple_df)) + np.arange(10, 10 + len(simple_df))))

    a.clear()
    b.clear()

    c = DetC('c', a + b, group='time_2')

    d = DC(pm.Normal, name='d', mu=c)

    _ = build(simple_df, d)

    assert c.dims() == ('time_2', )
    assert all(x == pytest.approx(y) for x, y in zip(pm.draw(c.variable), np.arange(5) + np.arange(10, 15)))

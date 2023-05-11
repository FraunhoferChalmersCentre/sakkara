import pytest
import pymc as pm
import numpy as np
import pytensor.tensor as pt

from sakkara.model import DistributionComponent as DC, Reshaper, DeterministicComponent, build, DataComponent, f_


@pytest.mark.usefixtures('simple_df')
def test_deterministic_component(simple_df):
    a = DC(pm.Normal, group='time', mu=DataComponent(np.arange(5), group='time'), sigma=1e-15)
    b = DC(pm.Normal, mu=1, sigma=1e-15)

    c = DeterministicComponent('c', a + b)

    d = DC(pm.Normal, name='d', mu=c)

    _ = build(simple_df, d)

    assert a.get_name().split('<')[0] == 'mu_d_'
    assert a.get_name().split('>')[-1] == '_arg0'
    assert b.get_name().split('<')[0] == 'mu_d_'
    assert b.get_name().split('>')[-1] == '_arg1'
    assert tuple(map(str, c.representation.groups)) == ('time',)
    assert all(x == pytest.approx(y) for x, y in zip(pm.draw(c.variable), np.arange(1, 6)))


@pytest.mark.usefixtures('simple_df')
def test_twin(simple_df):
    a = DC(pm.Normal, group='obs', mu=DataComponent(np.arange(len(simple_df)), group='time_2'), sigma=1e-15)
    b = DC(pm.Normal, group='time_2', mu=DataComponent(np.arange(10, 10 + len(simple_df)), group='obs'), sigma=1e-15)

    c = DeterministicComponent('c', a + b)

    d = DC(pm.Normal, name='d', mu=c)

    _ = build(simple_df, d)

    assert tuple(map(str, c.representation.get_groups())) == ('obs',)
    assert all(x == pytest.approx(y) for x, y in
               zip(pm.draw(c.variable), np.arange(len(simple_df)) + np.arange(10, 10 + len(simple_df))))

    a.clear()
    b.clear()

    c = DeterministicComponent('c', Reshaper(a + b, group='time_2'))

    d = DC(pm.Normal, name='d', mu=c)

    _ = build(simple_df, d)

    assert tuple(map(str, c.representation.get_groups())) == ('time_2',)
    assert all(x == pytest.approx(y) for x, y in zip(pm.draw(c.variable), np.arange(5) + np.arange(10, 15)))


@pytest.mark.usefixtures('simple_df')
def test_fct_wrapping(simple_df):
    x = DC(pm.Uniform, name='x', group='time', lower=[1, 2, 3, 4, 5], upper=[1, 2, 3, 4, 5])

    c = DC(pm.Uniform, name='c', lower=1, upper=1)

    y = DeterministicComponent('y', c + f_(pt.cumsum)(x))

    _ = build(simple_df, y)

    assert pm.draw(c.variable).shape == (1,)
    assert tuple(map(str, c.representation.get_groups())) == ('global',)
    assert tuple(map(str, y.representation.get_groups())) == ('time',)
    assert pm.draw(y.variable).shape == (5,)
    assert all(pytest.approx(pm.draw(y.variable)[i - 1]) == 1 + i * (1 + i) / 2 for i in range(1, 6))


@pytest.mark.usefixtures('simple_df')
def test_reshaping(simple_df):
    x = DataComponent(np.repeat(np.arange(4), 5).reshape(4, 5), group=('sensor', 'time'))

    c = DC(pm.Uniform, name='c', lower=np.arange(20), upper=np.arange(20), group='obs')

    y = DeterministicComponent('y', Reshaper(c + x, group=('time', 'sensor')))

    _ = build(simple_df, y)

    assert pm.draw(c.variable).shape == (20,)
    assert tuple(map(str, c.representation.get_groups())) == ('obs',)
    assert tuple(map(str, y.representation.get_groups())) == ('time', 'sensor')
    assert pm.draw(y.variable).shape == (5, 4)
    assert all(pm.draw(y.variable)[j, i] == i * 6 + j for j in range(5) for i in range(4))

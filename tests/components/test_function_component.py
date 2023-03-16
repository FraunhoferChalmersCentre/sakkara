import numpy as np
import pytest
import pymc as pm

from sakkara.model import DistributionComponent as DC, build, f_, DataComponent


@pytest.mark.usefixtures('simple_df')
def test_args_only(simple_df):
    def fct(x, y):
        return 10 * x + y

    a = DC(pm.Normal, mu=2, sigma=1e-15, group='building')
    b = DC(pm.Normal, mu=4, sigma=1e-15, group='sensor')

    with pytest.raises(ValueError):
        _ = build(simple_df, f_(fct)(a, b))

    a_named = DC(pm.Normal, mu=2, sigma=1e-15, group='building', name='a')
    b_named = DC(pm.Normal, mu=4, sigma=1e-15, group='sensor', name='b')

    _ = build(simple_df, f_(fct)(a_named, b_named))

    c = DC(pm.Normal, name='c', mu=f_(fct)(a, b), sigma=1e-15)

    _ = build(simple_df, c)

    assert tuple(map(str, a.representation.groups)) == ('building',)
    assert tuple(map(str, b.representation.groups)) == ('sensor',)
    assert tuple(map(str, c.representation.groups)) == ('sensor',)
    assert a.get_name().split('<')[0] == 'mu_c_'
    assert a.get_name().split('>')[-1] == '_arg0'
    assert b.get_name().split('<')[0] == 'mu_c_'
    assert b.get_name().split('>')[-1] == '_arg1'
    assert all([pytest.approx(24) == z for z in pm.draw(c.variable)])


@pytest.mark.usefixtures('simple_df')
def test_kwargs_only(simple_df):
    def fct(x, y):
        return 10 * x + y

    a = DC(pm.Normal, mu=2, sigma=1e-15, group='building')
    b = DC(pm.Normal, mu=4, sigma=1e-15, group='sensor')

    with pytest.raises(ValueError):
        _ = build(simple_df, f_(fct)(y=b, x=a))

    a_named = DC(pm.Normal, mu=2, sigma=1e-15, group='building', name='a')
    b_named = DC(pm.Normal, mu=4, sigma=1e-15, group='sensor', name='b')

    _ = build(simple_df, f_(fct)(y=b_named, x=a_named))

    c = DC(pm.Normal, name='c', mu=f_(fct)(y=b, x=a), sigma=1e-15)

    _ = build(simple_df, c)

    assert tuple(map(str, a.representation.groups)) == ('building',)
    assert tuple(map(str, b.representation.groups)) == ('sensor',)
    assert tuple(map(str, c.representation.groups)) == ('sensor',)
    assert a.get_name().split('<')[0] == 'mu_c_'
    assert a.get_name().split('>')[-1] == '_x'
    assert b.get_name().split('<')[0] == 'mu_c_'
    assert b.get_name().split('>')[-1] == '_y'
    assert all([pytest.approx(24) == z for z in pm.draw(c.variable)])


@pytest.mark.usefixtures('simple_df')
def test_args_and_kwargs(simple_df):
    def fct(x, y, z):
        return 100 * x + 10 * y + z

    a = DC(pm.Normal, mu=2, sigma=1e-15, group='building')
    b = DC(pm.Normal, mu=4, sigma=1e-15, group='sensor')
    c = DC(pm.Normal, mu=6, sigma=1e-15)

    with pytest.raises(ValueError):
        _ = build(simple_df, f_(fct)(a, z=c, y=b))

    a_named = DC(pm.Normal, mu=2, sigma=1e-15, group='building', name='a')
    b_named = DC(pm.Normal, mu=4, sigma=1e-15, group='sensor', name='b')
    c_named = DC(pm.Normal, mu=6, sigma=1e-15, name='c')

    _ = build(simple_df, f_(fct)(a_named, z=c_named, y=b_named))

    d = DC(pm.Normal, name='r', mu=f_(fct)(a, z=c, y=b), sigma=1e-15)

    _ = build(simple_df, d)

    assert tuple(map(str, a.representation.groups)) == ('building',)
    assert tuple(map(str, b.representation.groups)) == ('sensor',)
    assert tuple(map(str, c.representation.groups)) == ('global',)
    assert tuple(map(str, d.representation.groups)) == ('sensor',)
    assert a.get_name().split('<')[0] == 'mu_r_'
    assert a.get_name().split('>')[-1] == '_arg0'
    assert b.get_name().split('<')[0] == 'mu_r_'
    assert b.get_name().split('>')[-1] == '_y'
    assert c.get_name().split('<')[0] == 'mu_r_'
    assert c.get_name().split('>')[-1] == '_z'
    assert all([pytest.approx(246) == z for z in pm.draw(d.variable)])


@pytest.mark.usefixtures('simple_df')
def test_tensor_fct(simple_df):
    x = DataComponent(np.repeat(np.arange(5), 4).reshape(5, 4), group=('time', 'sensor'))

    c = DC(pm.Uniform, name='c', lower=1, upper=1)

    y = c + f_(np.cumsum)(x, axis=1)

    _ = build(simple_df, y)

    assert pm.draw(c.variable).shape == (1,)
    assert tuple(map(str, c.representation.get_groups())) == ('global',)
    assert tuple(map(str, y.representation.get_groups())) == ('time', 'sensor')
    assert pm.draw(y.variable).shape == (5, 4)
    assert all(pm.draw(y.variable)[i, j] == 1 + i * (j + 1) for j in range(4) for i in range(5))

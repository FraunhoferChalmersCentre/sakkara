import pytest
import pymc as pm

from sakkara.model import GroupComponent as GC, DistributionComponent as DC, build


@pytest.mark.usefixtures('simple_df')
def test_fixed_values(simple_df):
    gc = GC('sensor', 'gc', {k: i for i, k in enumerate('stuv')})

    x = DC(pm.Uniform, name='x', lower=gc, upper=gc, group='sensor')

    _ = build(simple_df, x)

    assert list(pm.draw(x.variable)) == [0, 1, 2, 3]
    assert [str(g) for g in gc.representation.get_groups()] == ['sensor']
    assert pm.draw(gc.variable).shape == (4,)
    assert all(pm.draw(gc.variable)[i] == i for i in range(4))


@pytest.mark.usefixtures('simple_df')
def test_unrelated_groups(simple_df):
    gc = GC('sensor', 'gc', {k: DC(pm.Uniform, group='time', lower=i, upper=i) for i, k in enumerate('stuv')})

    x = gc + DC(pm.Uniform, 'x', lower=1, upper=1, group='building')

    _ = build(simple_df, x)

    assert [str(g) for g in gc.representation.get_groups()] == ['sensor', 'time']
    assert pm.draw(gc.variable).shape == (4, 5)
    assert all(pm.draw(gc.variable)[i, j] == i for i in range(4) for j in range(5))


@pytest.mark.usefixtures('simple_df')
def test_mixed_groups(simple_df):
    gc = GC('sensor', 'gc', {k: DC(pm.Uniform, group='time', lower=i, upper=i) for i, k in enumerate('st')})
    gc.add('u', DC(pm.Uniform, lower=2, upper=2, group='global'))
    gc.add('v', 3)

    _ = build(simple_df, gc)

    assert [str(g) for g in gc.representation.get_groups()] == ['sensor', 'time']
    assert pm.draw(gc.variable).shape == (4, 5)
    assert all(pm.draw(gc.variable)[i, j] == i for i in range(4) for j in range(5))


@pytest.mark.usefixtures('simple_df')
def test_parent_child(simple_df):
    with pytest.raises(ValueError):
        gc = GC('building', 'gc', {k: DC(pm.Uniform, group='sensor', lower=i, upper=i) for i, k in enumerate('ab')})
        _ = build(simple_df, gc)

    c = DC(pm.Uniform, group='building', lower=[0, 1], upper=[0, 1])
    gc = GC('sensor', 'gc', {k: c for i, k in enumerate('stuv')})
    _ = build(simple_df, gc)

    assert pm.draw(gc.variable).tolist() == [0, 0, 1, 1]

@pytest.mark.usefixtures('simple_df')
def test_misc_errors(simple_df):
    with pytest.raises(ValueError):
        gc = GC('sensor', 'gc', {k: DC(pm.Uniform, group='time', lower=i, upper=i) for i, k in enumerate('st')})
        _ = build(simple_df, gc)



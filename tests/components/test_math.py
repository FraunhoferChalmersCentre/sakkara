import operator

import pymc as pm
import pytest

from sakkara.model import DistributionComponent, DataComponent, build


@pytest.fixture
def variables():
    a = DistributionComponent(pm.Normal, 'a', mu=2, sigma=1e-15)
    b = DistributionComponent(pm.Normal, 'b', mu=3, sigma=1e-15)
    c = DataComponent(5, group='global', name='c')

    return {2: a, 3: b, 5: c, 7: 7}


def operator_test(op, variables, simple_df):
    for k1, v1 in variables.items():
        for k2, v2 in variables.items():
            x = op(v1, v2)
            if isinstance(x, int) or isinstance(x, float):
                continue

            with build(simple_df, x):
                assert pm.draw(x.variable).flatten()[0] == pytest.approx(op(k1, k2))

            x = op(v2, v1)
            with build(simple_df, x):
                assert pm.draw(x.variable).flatten()[0] == pytest.approx(op(k2, k1))


def test_operators(variables, simple_df):
    operator_test(operator.add, variables, simple_df)
    operator_test(operator.sub, variables, simple_df)
    operator_test(operator.mul, variables, simple_df)
    operator_test(operator.truediv, variables, simple_df)
    operator_test(operator.floordiv, variables, simple_df)
    operator_test(operator.pow, variables, simple_df)
    operator_test(operator.mod, variables, simple_df)

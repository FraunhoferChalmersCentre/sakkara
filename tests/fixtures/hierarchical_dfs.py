import pytest
import pandas as pd
import numpy as np
from numpy.random import default_rng

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

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            'global': 'global',
            'building': np.repeat(list('ab'), 10),
            'sensor': np.repeat(list('stuv'), 5),
            'time': np.tile(pd.date_range("2018-01-01 00:00:00", tz='utc', periods=5, freq="H"), 4),
            'time_2': pd.date_range("2020-01-01 00:00:00", tz='utc', periods=20, freq="H"),
            'obs': np.arange(20)
        })

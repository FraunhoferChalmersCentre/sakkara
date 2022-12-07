import warnings
from typing import Callable, Dict

import numpy as np
import pandas as pd
import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.relation.groupset import init


def build(df: pd.DataFrame, likelihood: ModelComponent):

    tmp_df = df.copy()
    tmp_df.loc[:, 'global'] = 'global'
    tmp_df.loc[:, 'obs'] = np.arange(len(df))

    columns = likelihood.retrieve_columns().union({'global', 'obs'})
    groupset = init(tmp_df.loc[:, list(columns)])

    with pm.Model(coords=groupset.coords()) as model:
        likelihood.build(groupset)
    return model

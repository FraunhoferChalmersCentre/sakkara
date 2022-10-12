from typing import Callable, Dict

import numpy as np
import pandas as pd
import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.model.components import Distribution, Deterministic
from sakkara.relation.groupset import init


class Likelihood(Distribution):
    def __init__(self, generator: Callable, data: ModelComponent, name=None, **kwargs):
        super().__init__(generator, column='obs', name=name, **kwargs)
        self.components['observed'] = data
        self.name = 'likelihood' if name is None else name


def data_components(df: pd.DataFrame) -> Dict[str, ModelComponent]:
    return {k: Distribution(pm.Data, 'data_' + k, value=df.loc[:, k].values, column='obs', mutable=False) for k in df}


def build(df: pd.DataFrame, likelihood: Likelihood):
    likelihood.clear()

    tmp_df = df.copy()
    tmp_df.loc[:, 'global'] = 'global'
    tmp_df.loc[:, 'obs'] = np.arange(len(df))

    groupset = init(tmp_df.loc[:, list(likelihood.retrieve_columns())])

    with pm.Model(coords=groupset.coords()) as model:
        likelihood.build(groupset)
    return model

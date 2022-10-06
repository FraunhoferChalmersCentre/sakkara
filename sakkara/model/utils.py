from abc import ABC
from typing import Callable

import numpy as np
import pandas as pd
import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.model.components import HierarchicalComponent, Concat, Deterministic, Distribution, Stacked
from sakkara.relation.groupset import GroupSet, init


class Data(Concat, ABC):
    def __init__(self, df: pd.DataFrame, group_name='obs'):
        super().__init__({k: Deterministic(value=df[k], group_name=group_name) for k in df})


class Likelihood(Distribution):
    def __init__(self, generator: Callable, data: ModelComponent, name=None, **kwargs):
        super().__init__(generator, group_name='obs', name=name, **kwargs)
        self.components['observed'] = data
        self.name = 'likelihood' if name is None else name


def build(df: pd.DataFrame, likelihood: Likelihood):
    likelihood.clear()

    tmp_df = df.copy()
    tmp_df['global'] = 'global'
    tmp_df['obs'] = np.arange(len(df))

    groupset = init(tmp_df.loc[:, list(likelihood.retrieve_group_names())])

    with pm.Model(coords=groupset.coords()) as model:
        likelihood.build(groupset)
    return model

from abc import ABC
from typing import Callable

import numpy as np
import pandas as pd
import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.model.multi_component import ConcatComponent
from sakkara.model.single_component import HierarchicalComponent
from sakkara.relation.groupset import GroupSet, init


class DataConcatComponent(ConcatComponent, ABC):
    def __init__(self, df: pd.DataFrame):
        keys = list(df.columns)
        components = [
            HierarchicalComponent(pm.Data, value=df.loc[:, k], name=f'{k}_data', group_name='obs', mutable=False) for k
            in keys]
        super().__init__(components, keys)


class Likelihood(HierarchicalComponent):
    def __init__(self, distribution: Callable, data: ModelComponent, name=None, **kwargs):
        super().__init__(distribution, **kwargs)
        self.data = data
        self.group_name = 'obs'
        self.name = 'likelihood' if name is None else name

    def prebuild(self, groupset: GroupSet) -> None:
        self.params['observed'] = self.data
        super(Likelihood, self).prebuild(groupset)


def build(df: pd.DataFrame, likelihood: Likelihood):
    likelihood.clear()

    tmp_df = df.copy()
    tmp_df['global'] = 'global'
    tmp_df['obs'] = np.arange(len(df))

    groupset = init(tmp_df.loc[:, list(likelihood.retrieve_group_names())])

    with pm.Model(coords=groupset.coords()) as model:
        likelihood.build(groupset)
    return model

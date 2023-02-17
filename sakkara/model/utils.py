import numpy as np
import pandas as pd
import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.relation.groupset import init


def build(df: pd.DataFrame, component: ModelComponent):
    """
    Build a complete PyMC model based on a single Sakkara ModelComponent (typically a Likelihood object). Sakkara
    will trace all underlying components, and their respective groupings, necessary for creating the model.

    Parameters
    ----------
    df: DataFrame containing columns defining groups used among Sakkara ModelComponent objects.
    component: ModelComponent object (of the lowest hierarchy, typically a Likelihood) to init creation of PyMC model
        from.

    Returns
    -------
    A PyMC model (https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html#pymc.Model) ready to apply
    inference algorithms on.

    """
    tmp_df = df.copy()
    tmp_df.loc[:, 'global'] = 'global'
    tmp_df.loc[:, 'obs'] = np.arange(len(df))

    groups = component.retrieve_groups().union({'global', 'obs'})
    groupset = init(tmp_df.loc[:, list(groups)])

    with pm.Model(coords=groupset.coords()) as model:
        component.build(groupset)
    return model

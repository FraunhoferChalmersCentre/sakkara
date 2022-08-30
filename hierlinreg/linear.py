from typing import Dict, Set

import pymc as pm
import pandas as pd
import aesara.tensor as at

from hierlinreg.hierarchical import HierarchicalVariable, Likelihood
from hierlinreg.relation import init_groups


def retrieve_group_names(model_scheme: Dict[str, HierarchicalVariable]) -> Set[str]:
    groups = set()
    for v in model_scheme.values():
        groups = groups.union(v.retrieve_variable_groups())
    return groups


def create_linear_model(df: pd.DataFrame, target: str, model_spec: Dict[str, HierarchicalVariable],
                        likelihood: Likelihood):
    col_names = retrieve_group_names(model_spec)
    groupset = init_groups(df, *col_names)
    with pm.Model(coords=groupset.coords()) as model:
        coeffs = []

        for k, v in model_spec.items():
            v.build(k, groupset)
            mapping = groupset['observation'].get_parent_mapping(v.group_name)[f'{v.group_name}_id']
            coeffs.append(v.variable[mapping])

        data = pm.Data('data', df.loc[:, model_spec.keys()], dims=('observation', 'column'))
        coeffs_t = at.concatenate(coeffs)

        likelihood.set_data(df[target])
        likelihood.set_estimated(at.diag(data * coeffs_t))

        likelihood.build('likelihood', groupset)

    return model

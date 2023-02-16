import warnings
from abc import ABC
from typing import Callable, Any, Dict, Union

import numpy as np

from sakkara.model.base import ModelComponent
from sakkara.model.fixed.base import FixedComponent
from sakkara.model.fixed.data import SeriesComponent
from sakkara.model.composable.group.component import GroupComponent
from sakkara.model.composable.hierarchical.random_variable import RandomVariable


class Likelihood(RandomVariable, ABC):
    def __init__(self, generator: Callable, obs_data: SeriesComponent, name='likelihood', columns='obs',
                 nan_var_mask: Dict[str, Any] = None, nan_data_mask: Any = None, **kwargs: Any):

        if np.any(np.isnan(obs_data.values)):
            warnings.warn('Observables contains NaN values, these will be ignored in likelihood computation.',
                          UserWarning)

            components = {}

            for k, v in kwargs.items():
                var_mask = nan_var_mask[k] if isinstance(nan_var_mask, dict) else nan_var_mask

                component = GroupComponent(columns=columns, name=k + '_masked')
                for m in np.argwhere(~np.isnan(obs_data.values)).flatten():
                    component.add(m, v)
                for m in np.argwhere(np.isnan(obs_data.values)).flatten():
                    component.add(m, FixedComponent(value=var_mask, name=f'{k}_{m}_mask'))
                components[k] = component

            non_nandata = np.array([o if not np.isnan(o) else nan_data_mask for o in obs_data.values])
            components['observed'] = SeriesComponent(non_nandata, 'masked_' + obs_data.name)
        else:
            components = kwargs
            components['observed'] = obs_data

        super().__init__(generator, name, columns, **components)

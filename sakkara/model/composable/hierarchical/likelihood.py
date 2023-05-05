import warnings
from abc import ABC
from typing import Callable, Any, Dict, Union, Tuple

import numpy as np

from sakkara.model.fixed.data import DataComponent
from sakkara.model.composable.group import GroupComponent
from sakkara.model.composable.hierarchical.distribution import DistributionComponent


class Likelihood(DistributionComponent, ABC):
    """
    Component for a likelihood distribution, e.g, with observed data.

    :param generator: PyMC callable for distribution to use.
    :param observed: Data to input as observed keyword in PyMC.
    :param name: Name of the corresponding variable to register in PyMC.
    :param group: Group of which the component is defined for.
    :param nan_param_mask: Masked distribution parameters to use for rows with `Nan`, must be defined for each keyword argument entered. Required if there are `Nan` in observed.
    :param nan_data_mask: Masked observed value to use for rows with `Nan`. Required if there are `Nan` in observed.
    """

    def __init__(self,
                 generator: Callable,
                 observed: DataComponent,
                 name: str = 'likelihood',
                 group: Union[str, Tuple[str, ...]] = 'obs',
                 nan_param_mask: Dict[str, Any] = None,
                 nan_data_mask: Any = None,
                 **kwargs: Any):
        if np.any(np.isnan(observed.values)):
            if nan_param_mask is None or nan_data_mask:
                raise ValueError(
                    'Both nan_param_mask and nan_data_mask must be defined when there is Nan in the observed data.')

            warnings.warn('Observables contains NaN values, these will be ignored in likelihood computation.',
                          UserWarning)

            components = {}

            for k, v in kwargs.items():
                if k not in nan_param_mask:
                    raise ValueError('Mask values for all parameters must be defined in nan_param_mask')

                var_mask = nan_param_mask[k] if isinstance(nan_param_mask, dict) else nan_param_mask

                component = GroupComponent(group=group, name=k + '_masked')
                mask_component = DataComponent(var_mask, 'global')
                for m in np.argwhere(~np.isnan(observed.values)).flatten():
                    component.add(m, v)
                for m in np.argwhere(np.isnan(observed.values)).flatten():
                    component.add(m, mask_component)
                components[k] = component

            non_nandata = np.array([o if not np.isnan(o) else nan_data_mask for o in observed.values])
            components['observed'] = DataComponent(non_nandata, group, name='masked_' + observed.name)
        else:
            components = kwargs
            components['observed'] = observed

        super().__init__(generator, name, group, **components)


class MinibatchLikelihood(Likelihood):
    def __init__(self, generator: Callable,
                 observed: DataComponent,
                 batch_size: int,
                 name: str = 'likelihood',
                 group: str = 'obs',
                 nan_param_mask: Dict[str, Any] = None,
                 nan_data_mask: Any = None,
                 **kwargs: Any):
        kwargs['total_size'] = len(observed.values)
        super().__init__(generator, observed, name, group, nan_param_mask, nan_data_mask, **kwargs)

        for k, v in self.subcomponents.items():
            self.subcomponents[k] = v.to_minibatch(batch_size, group)

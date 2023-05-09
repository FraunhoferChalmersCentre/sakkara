from abc import ABC
from typing import Dict, Union, Tuple, Optional

import pandas as pd
import numpy as np
import numpy.typing as npt
import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.model.fixed.base import FixedValueComponent
from sakkara.model.minibatch import MinibatchComponent
from sakkara.relation.groupset import GroupSet
from sakkara.relation.representation import MinimalTensorRepresentation


class DataComponent(FixedValueComponent, ABC):
    """
    Wrap data into a component

    :param data: Array of data to wrap.
    :param group: Group(s) of which the component is defined for. The number of elements should correspond to the
        order of the data array.
    :param name: Name of the component.
    """

    def __init__(self, data: Union[npt.NDArray, float, int], group: Union[str, Tuple[str, ...]], name: str = None):
        if isinstance(data, np.ndarray):
            super().__init__(data, group, name)
        else:
            super().__init__(np.array([data]), group, name)

    def get_name(self) -> Optional[str]:
        return self.name

    def build_representation(self, groupset: GroupSet) -> None:
        self.representation = MinimalTensorRepresentation()
        for g in self.group:
            self.representation.add_group(groupset[g])

    def build_variable(self) -> None:
        self.variable = pm.Data(self.name, self.values)

    def to_minibatch(self, batch_size: int, group: str) -> 'ModelComponent':
        return MinibatchComponent(self, batch_size, group)


def data_components(df: pd.DataFrame, group: Union[str, Tuple[str, ...]] = 'obs') -> Dict[str, DataComponent]:
    """
    Generate :class:`DataComponent` objects from a :class:`pandas.DataFrame`

    :param df: DataFrame to generate components from.
    :param group: Group to apply, i.e., each member of the given group corresponds to one row of the dataframe.


    :return: Dictionary of {<column name in DataFrame>: :class:`DataComponent`}

    """
    return {k: DataComponent(df[k].values, group, k) for k in df}

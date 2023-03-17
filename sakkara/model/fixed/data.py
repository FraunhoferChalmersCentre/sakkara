from abc import ABC
from typing import Dict, Union, Tuple

import pandas as pd
import numpy as np
import numpy.typing as npt

from sakkara.model.fixed.base import FixedValueComponent
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

    def build_representation(self, groupset: GroupSet) -> None:
        self.representation = MinimalTensorRepresentation()
        for g in self.group:
            self.representation.add_group(groupset[g])


def data_components(df: pd.DataFrame, group: Union[str, Tuple[str, ...]] = 'obs') -> Dict[str, DataComponent]:
    """
    Generate :class:`DataComponent` objects from a :class:`pandas.DataFrame`

    :param df: DataFrame to generate components from.
    :param group: Group to apply, i.e., each member of the given group corresponds to one row of the dataframe.


    :return: Dictionary of {<column name in DataFrame>: :class:`DataComponent`}

    """
    return {k: DataComponent(df[k].values, group, k) for k in df}

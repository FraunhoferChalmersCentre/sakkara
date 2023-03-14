from abc import ABC
from typing import Dict, Union, Tuple

import pandas as pd
import numpy.typing as npt

from sakkara.model.fixed.base import FixedValueComponent
from sakkara.relation.groupset import GroupSet
from sakkara.relation.representation import TensorRepresentation


class DataComponent(FixedValueComponent, ABC):
    """
    Helper component for wrapping Pandas Series objects

    :param data: Series to wrap as a component.
    :param name: Name of the corresponding variable to register in PyMC.
    :param group: Group of which the component is defined for.
    """

    def __init__(self, data: Union[npt.NDArray, pd.Series], group: Union[str, Tuple[str, ...]], name: str = None):
        super().__init__(data.values if isinstance(data, pd.Series) else data, group, name)

    def build_representation(self, groupset: GroupSet) -> None:
        self.representation = TensorRepresentation()
        for g in self.group:
            self.representation.add_group(groupset[g])


def data_components(df: pd.DataFrame) -> Dict[str, DataComponent]:
    """
    Generate :class:`SeriesComponent` objects from a :class:`pandas.DataFrame`

    :param df: DataFrame to generate components from.


    :return: Dictionary of {<column name in DataFrame>: :class:`SeriesComponent`}

    """
    return {k: DataComponent(df[k], 'obs', k) for k in df}

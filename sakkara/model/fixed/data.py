from abc import ABC
from typing import Dict, Union

import pandas as pd
import numpy.typing as npt

from sakkara.model.fixed.base import FixedValueComponent


class SeriesComponent(FixedValueComponent, ABC):
    """
    Helper component for wrapping Pandas Series objects

    :param data: Series to wrap as a component.
    :param name: Name of the corresponding variable to register in PyMC.
    :param group: Group of which the component is defined for.
    """
    def __init__(self, data: Union[npt.NDArray, pd.Series], name: str, group: str = 'obs'):
        super().__init__(data.values if isinstance(data, pd.Series) else data, name, group)


def data_components(df: pd.DataFrame) -> Dict[str, SeriesComponent]:
    """
    Generate :class:`SeriesComponent` objects from a :class:`pandas.DataFrame`

    :param df: DataFrame to generate components from.


    :return: Dictionary of {<column name in DataFrame>: :class:`SeriesComponent`}

    """
    return {k: SeriesComponent(df[k], k) for k in df}

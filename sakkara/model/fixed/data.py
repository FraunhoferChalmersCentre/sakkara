from abc import ABC
from typing import Dict, Union

import pandas as pd
import numpy.typing as npt

from sakkara.model.fixed.base import FixedValueComponent


class SeriesComponent(FixedValueComponent, ABC):
    """
    Helper component for wrapping Pandas Series objects
    """
    def __init__(self, data: Union[npt.NDArray, pd.Series], name: str, group: str = 'obs'):
        """

        Parameters
        ----------
        data: Series to wrap as a component.
        name: Name of the corresponding variable to register in PyMC.
        group: Group of which the component is defined for.
        """
        super().__init__(data.values if isinstance(data, pd.Series) else data, name, group)


def data_components(df: pd.DataFrame) -> Dict[str, SeriesComponent]:
    """
    Generate SeriesCmponent objects automatically from a DataFrame

    Parameters
    ----------
    df: DataFrame to generate components from.

    Returns
    -------
    Dictionary of {[column name in DataFrame]: SeriesComponent}

    """
    return {k: SeriesComponent(df[k], k) for k in df}

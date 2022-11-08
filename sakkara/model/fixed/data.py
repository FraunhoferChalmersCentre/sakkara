from abc import ABC
from typing import Dict, Union

import pandas as pd
import numpy.typing as npt

from sakkara.model.fixed.base import FixedComponent


class SeriesComponent(FixedComponent, ABC):
    def __init__(self, data: Union[npt.NDArray, pd.Series], name: str, columns='obs'):
        super().__init__(data.values if isinstance(data, pd.Series) else data, columns, name)


def data_components(df: pd.DataFrame) -> Dict[str, SeriesComponent]:
    return {k: SeriesComponent(df[k], k) for k in df}

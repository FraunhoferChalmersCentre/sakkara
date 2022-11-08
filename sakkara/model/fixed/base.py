from copy import deepcopy
from typing import Any, Union, Tuple, Optional, Set

import numpy as np

from sakkara.model.math_op import MathOpBase
from sakkara.relation.groupset import GroupSet
from sakkara.relation.node import NodePair


class FixedComponent(MathOpBase):
    """
    Class for fixed variables.
    """

    def __init__(self, value: Any, columns: Union[str, Tuple[str, ...]] = 'global', name: Optional[str] = None):
        super().__init__()
        self.columns = (columns,) if isinstance(columns, str) else columns
        self.name = name
        self.values = np.array(value).reshape(1) if isinstance(value, (float, int)) else value

    def set_name(self, name: str) -> None:
        self.name = name

    def get_name(self) -> Optional[str]:
        return self.name if self.name is not None else 'fixed'

    def clear(self):
        self.variable = None
        self.node = None

    def prebuild(self, groupset: GroupSet) -> None:
        pass

    def build_node(self, groupset: GroupSet) -> None:
        self.node = groupset[self.columns[0]]
        for column in self.columns[1:]:
            self.node = NodePair(groupset[column], self.node).reduced_repr()

    def build_variable(self) -> None:
        self.variable = deepcopy(self.values)

    def retrieve_columns(self) -> Set[str]:
        return set(self.columns)
from abc import ABC
from typing import Optional, Union, Tuple

import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import T
from sakkara.model.composable.hierarchical.base import HierarchicalComponent


class DeterministicComponent(HierarchicalComponent[T], ABC):
    """
    Wrapper for PyMC's Deterministic variables
    """

    def __init__(self, name: str, component: ModelComponent, group: Optional[Union[str, Tuple[str, ...]]] = None):
        """

        Parameters
        ----------
        name: Name that will be applied to the Deterministic variable in PyMC.
        component: ModelComponent whose corresponding PyMC variable wil be wrapped in the Deterministic variable
        group: Group of which the component is defined for.
        """
        super().__init__(name, group, None, subcomponents={'var': component})

    def build_variable(self) -> None:
        self.variable = pm.Deterministic(self.name, **self.get_built_components(), dims=self.dims())

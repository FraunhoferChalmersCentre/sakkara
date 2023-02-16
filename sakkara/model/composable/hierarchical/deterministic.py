from abc import ABC

import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import T
from sakkara.model.composable.hierarchical.base import HierarchicalComponent


class DeterminisitcComponent(HierarchicalComponent[T], ABC):
    """
    Wrapper for PyMC's Deterministic
    """

    def __init__(self, name: str, component: ModelComponent):
        super().__init__(name, 'global', None, components={'var': component})

    def build_variable(self) -> None:
        self.variable = pm.Deterministic(self.name, **self.get_built_components(), dims=self.dims())

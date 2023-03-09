from abc import ABC
from typing import Optional, Union, Tuple

import pymc as pm

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import T
from sakkara.model.composable.hierarchical.base import HierarchicalComponent


class DeterministicComponent(HierarchicalComponent[T], ABC):
    """
    Wrapper for :class:`pymc.Deterministic`

    :param name: Name that will be applied to the :class:`pymc.Deterministic` object.
    :param component: :class:`ModelComponent` whose corresponding PyMC variable wil be wrapped into :class:`pymc.Deterministic`
    :param group: Group of which the component is defined for.
    """

    def __init__(self, name: str, component: ModelComponent, group: Optional[Union[str, Tuple[str, ...]]] = None):
        super().__init__(name, group, subcomponents={'var': component})

    def build_variable(self) -> None:
        self.variable = pm.Deterministic(self.name, **self.get_built_components(), dims=self.dims())

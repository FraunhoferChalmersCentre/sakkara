from abc import ABC
from typing import Callable, Optional, Union, Tuple, Any, Iterable

from sakkara.model.base import ModelComponent
from sakkara.model.composable.base import T
from sakkara.model.fixed.base import FixedComponent
from sakkara.model.composable.hierarchical.base import HierarchicalComponent


class RandomVariable(HierarchicalComponent[T], ABC):
    """
    Class for components whose variable is generated by a PyMC distribution
    """

    def __init__(self, generator: Callable, name: Optional[str] = None, columns: Union[str, Tuple[str, ...]] = 'global',
                 members: Optional[Iterable[Any]] = None, **components: Any):
        super().__init__(name, columns, members,
                         components={k: v if isinstance(v, ModelComponent) else FixedComponent(v) for k, v in
                                     components.items()})
        self.generator = generator

    def build_variable(self) -> None:
        self.variable = self.generator(self.name, **self.get_built_components(), shape=self.node.get_members().shape,
                                       dims=self.dims())

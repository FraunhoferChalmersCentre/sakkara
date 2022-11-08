from abc import ABC
from typing import Union, Tuple

from sakkara.model.base import ModelComponent
from sakkara.model.composable.hierarchical.wrapper import ComponentWrapper


class GroupWrapper(ComponentWrapper[ModelComponent], ABC):
    def __init__(self, component: ModelComponent, columns: Union[str, Tuple[str, ...]] = None, name: str = None):
        super().__init__(component, columns, None, name)

    def __getitem__(self, item) -> ModelComponent:
        return self.components[item]

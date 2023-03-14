from typing import Callable
from sakkara.model.function import FunctionComponent
from sakkara.model.fixed.base import UnrepeatableComponent as UC
from sakkara.model.base import ModelComponent


class FunctionWrapper:
    def __init__(self, fct: Callable):
        self.fct = fct

    def __call__(self, *args, **kwargs):
        fct_args = [m if isinstance(m, ModelComponent) else UC(m) for m in args]
        fct_kwargs = {k: m if isinstance(m, ModelComponent) else UC(m) for k, m in kwargs.items()}

        return FunctionComponent(self.fct, *fct_args, **fct_kwargs)


def f(fct: Callable):
    return FunctionWrapper(fct)
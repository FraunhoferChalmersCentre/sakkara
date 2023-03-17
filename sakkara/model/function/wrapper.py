from typing import Callable, Optional, Union, Tuple
from sakkara.model.function.base import FunctionComponent
from sakkara.model.fixed.base import UnrepeatableComponent as UC
from sakkara.model.base import ModelComponent


class f_:
    """
    Wraps a generic function with associated Sakkara components. After wrapped with this, inputs can be
    a subclass of :class:`ModelComponent` or any other type (which will be wrapped with :class:`UnrepeatableComponent`)

    **Example**

    .. highlight:: python
    .. code-block:: python

        import pymc as pm
        from sakkara.model import DistributionComponent as DC, f

        def g(a,b):
            return a + b

        comp = DC(pm.Normal)
        # Wrap g with f, pass arguments as we would have done directly to g
        res = f(g)(comp, 100)
        # res corresponds to N(0, 1) + 100

    :param fct: Function callable
    :param output_group: Group to assume on output use if input and output shapes do not match
    """

    def __init__(self, fct: Callable, output_group: Optional[Union[str, Tuple[str, ...]]] = None):
        self.fct = fct
        self.output_group = (output_group,) if isinstance(output_group, str) else output_group

    def __call__(self, *args, **kwargs):
        fct_args = [m if isinstance(m, ModelComponent) else UC(m) for m in args]
        fct_kwargs = {k: m if isinstance(m, ModelComponent) else UC(m) for k, m in kwargs.items()}

        return FunctionComponent(self.fct, self.output_group, *fct_args, **fct_kwargs)

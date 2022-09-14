import abc
import operator
from abc import ABC
from functools import cache
from typing import Callable, Set, Any, Dict, Iterable

import numpy as np
import pandas as pd
import pymc as pm

from sakkara.relation import GroupSet, init_groupset


class ModelComponent:
    def __init__(self, name: str = None, group: str = None):
        self.group = group
        self.name = name

    @abc.abstractmethod
    def retrieve_groups(self) -> Set[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_names(self) -> Set[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def build(self, groupset: GroupSet):
        raise NotImplementedError

    def __add__(self, other):
        return CompositeComponent(self, other, operator.add)

    def __sub__(self, other):
        return CompositeComponent(self, other, operator.sub)

    def __mul__(self, other):
        return CompositeComponent(self, other, operator.mul)

    def __rmul__(self, other):
        return CompositeComponent(self, other, operator.mul)

    def __truediv__(self, other):
        return CompositeComponent(self, other, operator.truediv)


class CompositeComponent(ModelComponent, ABC):
    def __init__(self, a: ModelComponent, b: ModelComponent, op: Callable[[Any, Any], Any], name: str = None):
        super().__init__(name, None)
        self.a = a
        self.b = b
        self.op = op

    def build(self, groupset: GroupSet):
        _ = self.a.build(groupset)
        _ = self.b.build(groupset)

        a_parent_to_b = groupset[self.b.group].is_parent(self.a.group)
        if a_parent_to_b:
            child = self.b
            parent = self.a
        else:
            child = self.a
            parent = self.b

        self.group = child.group

        mapping = groupset[child.group].get_parent_mapping(parent.group)[f'{parent.group}_id'].values
        if 1 < len(np.unique(mapping)):
            return self.op(child.build(groupset), parent.build(groupset)[mapping])
        else:
            return self.op(child.build(groupset), parent.build(groupset))

    def retrieve_groups(self) -> Set[str]:
        return self.a.retrieve_groups().union(self.b.retrieve_groups())

    def retrieve_names(self) -> Set[str]:
        return self.a.retrieve_names().union(self.b.retrieve_names())


class HierarchicalVariable(ModelComponent):

    def __init__(self, distribution: Callable, group: str = 'global', name=None, **kwargs):
        super().__init__(name, group)
        self.distribution = distribution
        self.params = kwargs
        self.variable = None

    def build(self, groupset: GroupSet):
        if self.variable is not None:
            return self.variable

        built_params = {}

        if self.group is None or self.group == 'global':
            dims = None
        else:
            dims = self.group

        for k, v in self.params.items():
            if isinstance(v, ModelComponent):
                v.name = f'{k}_{self.name}' if v.name is None else v.name
                built_component = v.build(groupset)
                if v.group is not None:
                    group = groupset[self.group]
                    parent_mapping = group.get_parent_mapping(v.group)[f'{v.group}_id'].values
                    if 1 < len(np.unique(parent_mapping)):
                        built_params[k] = built_component[parent_mapping]
                    else:
                        built_params[k] = built_component
                else:
                    built_params[k] = built_component
            else:
                built_params[k] = v

        self.variable = self.distribution(self.name, **built_params, dims=dims)
        return self.variable

    def retrieve_groups(self) -> Set[str]:
        groups = set()
        for k, v in self.params.items():
            if isinstance(v, ModelComponent):
                parent_groups = v.retrieve_groups()
                groups = groups.union(parent_groups)
        if self.group is not None:
            groups.add(self.group)

        return groups

    def retrieve_names(self) -> Set[str]:
        names = set()
        for k, v in self.params.items():
            if isinstance(v, ModelComponent):
                parent_names = v.retrieve_names()
                names = names.union(parent_names)
        if self.name is not None:
            names.add(self.name)

        return names


class Container:
    def __init__(self, components: Dict[str, ModelComponent]):
        self.components = components

    def __getitem__(self, item):
        return self.components[item]

    def sum(self) -> ModelComponent:
        s = None
        for k, v in self.components.items():
            if s is None:
                s = v
            else:
                s = s + v
        return s

    def math_operation(self, other: 'Container', op):
        common_keys = set(self.components.keys()).intersection(other.components.keys())
        return Container({k: CompositeComponent(self.components[k], other[k], op, k) for k in common_keys})

    def __add__(self, other: 'Container') -> 'Container':
        return self.math_operation(other, operator.add)

    def __sub__(self, other) -> 'Container':
        return self.math_operation(other, operator.sub)

    def __mul__(self, other) -> 'Container':
        return self.math_operation(other, operator.mul)

    def __rmul__(self, other) -> 'Container':
        return self.math_operation(other, operator.mul)

    def __truediv__(self, other) -> 'Container':
        return self.math_operation(other, operator.truediv)


class DataContainer(Container, ABC):
    def __init__(self, df: pd.DataFrame):
        super().__init__(
            {k: HierarchicalVariable(pm.Data, value=df.loc[:, k], name=f'{k}_data', group='obs', mutable=False) for k in
             df.columns}, )


class VariableContainer(Container, ABC):
    def __init__(self, components: Iterable[ModelComponent]):
        super().__init__({c.name: c for c in components})


class Likelihood(HierarchicalVariable):
    def __init__(self, distribution: Callable, data: ModelComponent, **kwargs):
        super().__init__(distribution, **kwargs)
        self.data = data
        self.group = 'obs'
        self.name = 'likelihood'


class HierarchicalModel:
    def __init__(self, df: pd.DataFrame, likelihood: Likelihood):
        self.likelihood = likelihood
        self.df = df

        self.groupset = init_groupset(self.df, likelihood.retrieve_groups(), likelihood.retrieve_names())

    def build(self) -> pm.Model:
        with pm.Model(coords=self.groupset.coords()) as model:
            self.likelihood.params['observed'] = self.likelihood.data.build(self.groupset)
            _ = self.likelihood.build(self.groupset)
        return model

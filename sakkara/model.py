import abc
import operator
from abc import ABC
from typing import Callable, Set, Any, Dict, List

import numpy as np
import pandas as pd
import pymc as pm
import aesara.tensor as at

from sakkara.relation.compositegroup import CompositeGroupPair
from sakkara.relation.groupset import GroupSet, init


class ModelComponent:
    def __init__(self, name: str = None, group_name: str = None):
        self.group_name = group_name
        self.name = name
        self.variable = None
        self.group = None

    @abc.abstractmethod
    def retrieve_group_names(self) -> Set[str]:
        raise NotImplementedError

    def clear(self):
        self.variable = None
        self.group = None

    @abc.abstractmethod
    def build(self, groupset: GroupSet) -> None:
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

    def clear(self):
        super(CompositeComponent, self).clear()
        self.a.clear()
        self.b.clear()

    def build(self, groupset: GroupSet):
        if self.a.variable is None:
            self.a.build(groupset)
        if self.b.variable is None:
            self.b.build(groupset)

        self.group = CompositeGroupPair(self.a.group, self.b.group)

        a_mapping = list(map(lambda m: m.index, self.group.map_from(self.a.group)))
        b_mapping = list(map(lambda m: m.index, self.group.map_from(self.b.group)))

        if 1 < len(np.unique(a_mapping)) and 1 < len(np.unique(b_mapping)):
            self.variable = self.op(self.a.variable[a_mapping], self.b.variable[b_mapping])
        else:
            self.variable = self.op(self.a.variable, self.b.variable)

    def retrieve_group_names(self) -> Set[str]:
        return self.a.retrieve_group_names().union(self.b.retrieve_group_names())


class Container(ModelComponent, ABC):
    def __init__(self, components: List[ModelComponent], keys: List[str], name: str = None, group_name: str = None):
        super().__init__(name, group_name)
        self.keys = keys
        self.key_mapping = {k: i for i, k in enumerate(keys)}
        self.components = components

    def clear(self):
        super(Container, self).clear()
        for c in self.components:
            c.clear()

    def build(self, groupset: GroupSet):
        build_components = []
        self.group = groupset[self.group_name]

        for k, c in zip(self.keys, self.components):
            if c.variable is None:
                c.name = f'{self.name}_{k}'
                c.build(groupset)
            build_components.append(c.variable)

        self.variable = at.stack(build_components)

    def retrieve_group_names(self) -> Set[str]:
        return {self.group_name}

    def __getitem__(self, item) -> ModelComponent:
        return self.components[self.key_mapping[item]]


class HierarchicalVariable(ModelComponent):

    def __init__(self, distribution: Callable, group_name: str = 'global', name=None, **kwargs):
        super().__init__(name, group_name)
        self.distribution = distribution
        self.params = kwargs
        self.variable = None
        self.group = None

    def build_parameter(self, param_name: str, groupset: GroupSet, component: ModelComponent) -> at.TensorVariable:
        if component.variable is None:
            component.name = f'{param_name}_{self.name}' if component.name is None else component.name
            component.build(groupset)
        if component.group_name is not None:
            mapping = list(map(lambda m: m.index, self.group.map_from(component.group)))
            if 1 < len(np.unique(mapping)):
                return component.variable[mapping]

        return component.variable

    def build(self, groupset: GroupSet) -> None:

        built_params = {}

        if self.group_name is None or self.group_name == 'global':
            dims = None
            self.group = groupset['global']
        else:
            dims = self.group_name
            self.group = groupset[self.group_name]

        for param_name, component in self.params.items():
            if isinstance(component, ModelComponent):
                built_params[param_name] = self.build_parameter(param_name, groupset, component)
            else:
                built_params[param_name] = component

        self.variable = self.distribution(self.name, **built_params, dims=dims)
        return self.variable

    def retrieve_group_names(self) -> Set[str]:
        groups = set()
        for k, v in self.params.items():
            if isinstance(v, ModelComponent):
                parent_groups = v.retrieve_group_names()
                groups = groups.union(parent_groups)
        if self.group_name is not None:
            groups.add(self.group_name)

        return groups


class DataContainer(Container, ABC):
    def __init__(self, df: pd.DataFrame):
        keys = list(df.columns)
        components = [
            HierarchicalVariable(pm.Data, value=df.loc[:, k], name=f'{k}_data', group_name='obs', mutable=False) for k
            in keys]
        super().__init__(components, keys, group_name='obs')


class Likelihood(HierarchicalVariable):
    def __init__(self, distribution: Callable, data: ModelComponent, name=None, **kwargs):
        super().__init__(distribution, **kwargs)
        self.data = data
        self.group_name = 'obs'
        self.name = 'likelihood' if name is None else name

    def build(self, groupset):
        self.params['observed'] = self.data.build(groupset)
        super(Likelihood, self).build(groupset)


def build(df: pd.DataFrame, likelihood: Likelihood):
    likelihood.clear()

    tmp_df = df.copy()
    tmp_df['global'] = 'global'
    tmp_df['obs'] = np.arange(len(df))

    groupset = init(tmp_df.loc[:, list(likelihood.retrieve_group_names())])

    with pm.Model(coords=groupset.coords()) as model:
        likelihood.build(groupset)
    return model

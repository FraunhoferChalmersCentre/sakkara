import abc
import operator
from abc import ABC
from copy import deepcopy
from typing import Callable, Set, Any, List, Optional

import numpy as np
import pandas as pd
import pymc as pm
import aesara.tensor as at

from sakkara.relation.groupset import GroupSet


class ModelComponent:
    """
        Abstract class for all model components (variables, data, parameters)
        """

    def __init__(self):
        self.node = None
        self.variable = None

    @abc.abstractmethod
    def get_name(self) -> Optional[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_name(self, name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clear variable and group for this and underlying components
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_variable(self) -> None:
        """
        Build the variable, performed after prebuild and build_group.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_columns(self) -> Set[str]:
        """
        Retrieve group names for this and underlying components
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prebuild(self, groupset: GroupSet) -> None:
        """
        All operations to be performed before building group and variable, e.g., building the underlying components.

        Parameters
        ----------
        groupset: Groups to be used for building all components of the model.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_node(self, groupset: GroupSet) -> None:
        """
        Build the group of this component, performed after prebuild.
        Parameters
        ----------
        groupset: Groups to be used for building all components of the model.
        """
        raise NotImplementedError

    def build(self, groupset: GroupSet) -> None:
        self.prebuild(groupset)
        self.build_node(groupset)
        self.build_variable()

    @abc.abstractmethod
    def __add__(self, other) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __sub__(self, other) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __mul__(self, other) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __rmul__(self, other) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __truediv__(self, other) -> 'ModelComponent':
        raise NotImplementedError


class FixedComponent(ModelComponent):
    """
    Class for fixed variables. This class is intended for internal usage, to specify deterministic values of a variables use the Deterministic class instead.
    """

    def __init__(self, value: Any, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.values = value

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
        self.node = groupset['global']

    def build_variable(self) -> None:
        self.variable = deepcopy(self.values)

    def retrieve_columns(self) -> Set[str]:
        return {'global'}

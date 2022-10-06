import abc
import operator
from abc import ABC
from typing import Callable, Set, Any, List

import numpy as np
import pandas as pd
import pymc as pm
import aesara.tensor as at

from sakkara.relation.groupset import GroupSet


class ModelComponent:
    """
    Base class for all model components (variables, data, parameters)
    """
    def __init__(self, name: str = None, group_name: str = None):
        """

        Parameters
        ----------
        name: Name of the component to used for naming in PyMC variables.
        group_name: Name of group for component to be found in groupset when building the model.
        """
        self.group_name = group_name
        self.name = name
        self.variable = None
        self.group = None

    @abc.abstractmethod
    def retrieve_group_names(self) -> Set[str]:
        """
        Retrieve group names for this and underlying components
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        """
        Clear variable and group for this and underlying components
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
    def build_group(self, groupset: GroupSet) -> None:
        """
        Build the group of this component, performed after prebuild.
        Parameters
        ----------
        groupset: Groups to be used for building all components of the model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_variable(self) -> None:
        """
        Build the variable, performed after prebuild and build_group.
        """
        raise NotImplementedError

    def build(self, groupset: GroupSet) -> None:
        self.prebuild(groupset)
        self.build_group(groupset)
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

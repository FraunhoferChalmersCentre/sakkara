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
    def __init__(self, name: str = None, group_name: str = None):
        self.group_name = group_name
        self.name = name
        self.variable = None
        self.group = None

    @abc.abstractmethod
    def retrieve_group_names(self) -> Set[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError

    @abc.abstractmethod
    def prebuild(self, groupset: GroupSet) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def build_group(self, groupset: GroupSet) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def build_variable(self) -> None:
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

import abc
from typing import Set, Optional, Any

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
    def __add__(self, other: Any) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __sub__(self, other: Any) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __mul__(self, other: Any) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __rmul__(self, other: Any) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __truediv__(self, other: Any) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __rtruediv__(self, other: Any) -> 'ModelComponent':
        raise NotImplementedError

    @abc.abstractmethod
    def __neg__(self) -> 'ModelComponent':
        raise NotImplementedError

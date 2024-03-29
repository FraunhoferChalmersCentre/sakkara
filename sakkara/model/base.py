import abc
from typing import Set, Optional, Any

from sakkara.relation.groupset import GroupSet


class ModelComponent:
    """
        Abstract class for all model components
    """

    def __init__(self):
        self.representation = None
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
    def retrieve_groups(self) -> Set[str]:
        """
        Retrieve group names for this and underlying components
        """
        raise NotImplementedError

    @abc.abstractmethod
    def prebuild(self, groupset: GroupSet) -> None:
        """
        All operations to be performed before building group and variable, e.g., building the underlying components.

        :param groupset: Groups to be used for building all components of the model.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_representation(self, groupset: GroupSet) -> None:
        """
        Build the group of this component, performed after prebuild.

        :param groupset: Groups to be used for building all components of the model.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_minibatch(self, batch_size: int, group: str) -> 'ModelComponent':
        """
        Convert the component to counterpart suitable for mini-batches
        """
        raise NotImplementedError

    def build(self, groupset: GroupSet) -> None:
        """
        Method to call for building variables from the component. Will chronological order call
        :meth:`ModelComponent.prebuild`, :meth:`ModelComponent.build_representation`, and
        :meth:`ModelComponent.build_variable`
        """
        self.prebuild(groupset)
        self.build_representation(groupset)
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

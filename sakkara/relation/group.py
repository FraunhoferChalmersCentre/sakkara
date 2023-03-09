from typing import Set, Any, Tuple, Callable, Union, List

import numpy as np
import numpy.typing as npt
import pandas as pd


class Group:
    """
    Class corresponding to a data frame column, with capabilities of handling relations to other groups.

    :param name: Name of the group, should be same as the column
    :param members: The names of the unique column values, the members, ordered by their first appearance.
    """
    def __init__(self, name: str, members: npt.NDArray[Any]):
        self.name = name
        self.members = members
        self.parents = set()
        self.children = set()
        self.twins = {self}
        self.mapping = pd.DataFrame(index=members, data={name: np.arange(len(members))})

    def add_child(self, child: 'Group') -> None:
        """
        Add a child relation to this group

        :param child: The child group
        """
        self.children.add(child)

    def add_parent(self, parent: 'Group', parent_mapping: npt.NDArray[int]) -> None:
        """
        Add a parent relation to this group

        :param parent: The parent group
        :param parent_mapping: Indices of parent members (of the parent group), ordered by the corresponding member of
            this group
        """
        self.parents.add(parent)
        self.mapping[parent.name] = parent_mapping

    def add_twin(self, twin: 'Group') -> None:
        """
        Add a twin relation to this group

        :param twin: The twin group
        """
        self.twins.add(twin)
        self.mapping[twin.name] = np.arange(len(self.mapping))

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.mapping)

    def __lt__(self, other) -> bool:
        return str(self) < str(other)

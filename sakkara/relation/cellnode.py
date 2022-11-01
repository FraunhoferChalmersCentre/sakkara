from sakkara.relation.atomicnode import AtomicNode
from typing import Set, Any
import numpy as np


class CellNode(AtomicNode):
    def __init__(self, name: Any, parents: Set[AtomicNode]):
        super().__init__(name, np.array(list()), parents)

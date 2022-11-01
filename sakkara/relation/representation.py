from sakkara.relation.node import Node
from typing import Iterable
import numpy as np


class Representation:

    def __init__(self, nodes: set['Node', ...]):
        self.nodes = nodes
        self.updated = True
        self.before_reduce = len(self.nodes)

    def get_order(self):
        if self.updated:
            self.reduce()
        list_sed_nodes = list(self.nodes)
        list_sed_nodes.sort()
        return list_sed_nodes

    def reduce(self):
        self.nodes = {n_i for n_i in self.nodes if not np.any([n_i.is_parent_to(n_j) for n_j in self.nodes])}
        lnodes = list(self.nodes)
        has_twin = [[j for j in range(i+1, len(lnodes)) if lnodes[i].is_twin_to(lnodes[j])] for i in range(len(lnodes)-1)] + [[]]
        index = np.ones((len(lnodes)), dtype=bool)
        for i in range(len(lnodes)):
            if len(has_twin[i]) > 0 and index[i]:
                index[has_twin[i]] = False
        self.nodes = {n_i for i, n_i in enumerate(lnodes) if index[i]}
        self.updated = False

    def set_nodes(self, nodes: set[Node, ...]):
        self.nodes = nodes
        self.comp_expected_len()

    def get_nodes(self):
        if self.updated:
            self.reduce()
        return self.nodes

    def __eq__(self, other: 'Representation') -> bool:
        if self.updated:
            self.reduce()
        nodes_eq = self.nodes == other.nodes
        if not nodes_eq:
            remaining_nodes = list(self.nodes.union(other.nodes) - self.nodes.intersection(other.nodes))
            n_nodes = len(remaining_nodes)
            matches = np.zeros((n_nodes, n_nodes), dtype=bool)
            for i in range(n_nodes-1):
                node_matches = [remaining_nodes[i].is_twin_to(remaining_nodes[j]) for j in range(i, n_nodes)]
                matches[i, i:] = node_matches
                matches[i:, i] = node_matches

            if np.all(np.any(matches, axis=0)):
                nodes_eq = True
        return nodes_eq

    def __copy__(self):
        if self.updated:
            self.reduce()
        return Representation(self.nodes.copy())

    def __iter__(self):
        if self.updated:
            self.reduce()
        return iter(self.nodes)

    def __str__(self) -> str:
        if self.updated:
            self.reduce()
        return '{' + ', '.join(map(str, self.nodes)) + '}'

    def __len__(self) -> int:
        if self.updated:
            self.reduce()
        return len(self.nodes)

    def get_expected_len(self):
        return self.before_reduce

    def comp_expected_len(self):
        self.before_reduce = len(self.nodes)

    # Set Operations
    def add(self, element: Node) -> None:
        self.nodes.add(element)
        self.updated = True
        self.comp_expected_len()

    def update(self, *s: Iterable[Node]) -> None:
        self.nodes.update(*s)
        self.comp_expected_len()
        self.updated = True

    def discard(self, element: Node) -> None:
        self.nodes.discard(element)
        self.comp_expected_len()
        self.updated = True

    def pop(self) -> Node:
        self.updated = True
        node = self.nodes.pop()
        self.comp_expected_len()
        return node

    def remove(self, element: Node) -> None:
        self.nodes.remove(element)
        self.updated = True
        self.comp_expected_len()

    def clear(self) -> None:
        self.nodes.clear()
        self.updated = True
        self.before_reduce = 0

    def union(self, *s: Iterable[Node]) -> 'Representation':
        copy_rep = self.__copy__()
        copy_rep.set_nodes(self.nodes.union(*s))
        return copy_rep

    def intersection(self, *s: Iterable[Node]) -> 'Representation':
        copy_rep = self.__copy__()
        copy_rep.set_nodes(self.nodes.intersection(*s))
        return copy_rep

    def difference(self, *s: Iterable[Node]) -> 'Representation':
        copy_rep = self.__copy__()
        copy_rep.set_nodes(self.nodes.difference(*s))
        return copy_rep

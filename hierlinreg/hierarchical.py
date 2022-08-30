from typing import Callable, Set

from hierlinreg.relation import GroupSet


class HierarchicalVariable:

    def __init__(self, distribution: Callable, group_name: str = 'global', **kwargs):
        self.distribution = distribution
        self.group_name = group_name
        self.params = kwargs
        self.variable = None

    def retrieve_variable_groups(self) -> Set[str]:
        groups = set()
        for k, v in self.params.items():
            if type(v) is HierarchicalVariable:
                parent_groups = v.retrieve_variable_groups()
                groups = groups.union(parent_groups)
        if self.group_name is not None:
            groups.add(self.group_name)

        return groups

    def build(self, name: str, groupset: GroupSet):
        built_params = {}

        if self.group_name is not None:
            dims = self.group_name
        else:
            dims = None

        for k, v in self.params.items():
            if type(v) in {float, int}:
                built_params[k] = v
            if type(v) is HierarchicalVariable:
                v.build(f'{k}_{name}', groupset)
                if v.group_name is not None:
                    group = groupset[self.group_name]
                    parent_mapping = group.get_parent_mapping(v.group_name)[f'{v.group_name}_id']
                    built_params[k] = v.variable[parent_mapping]
                else:
                    built_params[k] = v.variable

        self.variable = self.distribution(f'{name}_{self.group_name}', **built_params, dims=dims)


class Likelihood(HierarchicalVariable):
    def __init__(self, distribution: Callable, target_kw: str, **kwargs):
        super().__init__(distribution, 'observation', **kwargs)
        self.target_kw = target_kw

    def set_estimated(self, estimated):
        self.params[self.target_kw] = estimated

    def set_data(self, data):
        self.params['data'] = data

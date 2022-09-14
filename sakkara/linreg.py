import aesara.tensor as at

from sakkara.model import HierarchicalModel


class LinearRegression(HierarchicalModel):
    def __init__(self, target: str = None, **kwargs):
        super().__init__(**kwargs)
        self.target = target

    def build_likelihood(self):
        estimate = at.zeros(len(self.df))

        for k, v in self.spec.items():
            mapping = self.groupset['obs'].get_parent_mapping(v.group)[f'{v.group}_id'].values
            estimate += v.variable[mapping] * self.df.loc[:, k]

        self.likelihood.set_data(self.df[self.target].values)
        self.likelihood.set_estimated(estimate)

        self.likelihood.build('likelihood', self.groupset)

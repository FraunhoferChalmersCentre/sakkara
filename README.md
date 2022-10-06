![Sakkara logo](logo.png)

Welcome to Sakkara, a research framework for speeding up Bayesian hierarchical and graphical modelling using PyMC.

## Installation

    poetry install

## Minimal linear regression example

    import pandas as pd
    import numpy as np
    import pymc as pm
    from sakkara.model.components import Distribution, Deterministic, Stacked
    from sakkara.model.utils import Likelihood, Data, build
    
    x = np.random.randn(10)
    group = np.repeat([0, 1], 5)
    k = np.array([1, -1])
    y = x * k[group] + np.random.randn() * 1e-3
    df = pd.DataFrame({'x': x, 'y': y, 'group': group})
    
    data = Data(df)
    coeff = Distribution(pm.Normal, name='x', group_name='group', mu=Distribution(pm.Normal), sigma=Distribution(pm.Exponential, lam=1))
    intercept = Distribution(pm.Normal, name='intercept')
    
    likelihood = Likelihood(pm.Normal, name='est', mu=coeff * data['x'] + intercept, sigma=Distribution(pm.Exponential, lam=1),
                          data=data['y'])
    
    
    with build(df, likelihood):
        idata = pm.sample()

## More examples

See notebooks in the [/examples](/examples) directory.
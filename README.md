![Sakkara logo](logo.png)

Welcome to Sakkara, a framework for speeding up Bayesian hierarchical modelling using PyMC.

## Installation

    poetry install

## Minimal linear regression example

    import pandas as pd
    import numpy as np
    import pymc as pm
    from sakkara.model import HierarchicalVariable as HV, VariableContainer, DataContainer, HierarchicalModel, Likelihood
    
    x = np.random.randn(10)
    group = np.repeat([0, 1], 5)
    k = np.array([1, -1])
    y = x * k[group] + np.random.randn() * 1e-3
    df = pd.DataFrame({'x': x, 'y': y, 'group': group})
    
    data = DataContainer(df)
    coeff = HV(pm.Normal, name='x', group='group', mu=HV(pm.Normal), sigma=HV(pm.Exponential, lam=1))
    intercept = HV(pm.Normal, name='intercept')
    
    estimate = Likelihood(pm.Normal, name='est', mu=coeff * data['x'] + intercept, sigma=HV(pm.Exponential, lam=1),
                          data=data['y'])
    
    model = HierarchicalModel(df, likelihood=estimate)
    
    with model.build():
        idata = pm.sample()

## More examples

See notebooks in the [/examples](/examples) directory.
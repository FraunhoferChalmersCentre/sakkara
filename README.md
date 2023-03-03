![Sakkara logo](logo.png)

Welcome to Sakkara, a research framework for speeding up Bayesian hierarchical and graphical modelling using PyMC.

## Installation
Recommended is to first install PyMC via Conda, and then install Sakkara with
    
    pip install sakkara
    

## Small linear regression example

    import pandas as pd
    import numpy as np
    import pymc as pm
    from sakkara.model import DistributionComponent as DC, Likelihood, build, data_components
    
    x = np.random.randn(10)
    group = np.repeat([0, 1], 5)
    k = np.array([1, -1])
    y = x * k[group] + np.random.randn() * 1e-3
    df = pd.DataFrame({'x': x, 'y': y, 'group': group})
    
    data = data_components(df)
    coeff = DC(pm.Normal, name='x', group='group', mu=DC(pm.Normal),
               sigma=DC(pm.Exponential, lam=1))
    intercept = DC(pm.Normal, name='intercept')
    
    likelihood = Likelihood(pm.Normal, name='est', mu=coeff * data['x'] + intercept,
                            sigma=DC(pm.Exponential, lam=1),
                            observed=data['y'])
    
    with build(df, likelihood):
        idata = pm.sample()

## More examples

See notebooks in [the notebook examples repository](https://github.com/FraunhoferChalmersCentre/sakkara-examples).


## Docs

To be published...

## License

The project is licensed under the MIT license.
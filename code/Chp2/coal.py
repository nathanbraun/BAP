"""
Read about the coal mining disaster model that is part of the PyMC3
documentation:

http://pymc-devs.github.io/pymc3/notebooks/getting_started.html#Case-study-2:-Coal-mining-disasters.

Try to implement and run this model by yourself.
"""

import pandas as pd
import arviz as az
import pymc3 as pm
import numpy as np
from pandas import DataFrame, Series

# Consider the following time series of recorded coal mining disasters in the
# UK from 1851 to 1962 (Jarrett, 1979). The number of disasters is thought to
# have been affected by changes in safety regulations during this period.
# Unfortunately, we also have pair of years with missing data, identified as
# missing by a nan in the pandas Series. These missing values will be
# automatically imputed by PyMC3.

disaster_data = Series([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3,
                        5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4,
                        2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0,
                        0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
                        2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, np.nan, 2, 1, 1, 1,
                        1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                        0, 0, 1, 0, 1], index=range(1851, 1962),
                       name='naccidents')

# Occurrences of disasters in the time series is thought to follow a Poisson
# process with a large rate parameter in the early part of the time series, and
# from one with a smaller rate in the later part. We are interested in locating
# the change point in the series, which perhaps is related to changes in mining
# safety regulations.

with pm.Model() as model:
    alpha = 1.0/disaster_data.mean()

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)

    tau = pm.DiscreteUniform("tau", lower=disaster_data.index.min(),
                             upper=disaster_data.index.max())

    # idx = np.arange(n_count_data) # Index
    lambda_ = pm.math.switch(tau > disaster_data.index, lambda_1, lambda_2)

    observation = pm.Poisson("obs", lambda_, observed=disaster_data)

    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000,step=step)

pm.plot_trace(trace)
pm.plot_posterior(trace)

y_pred = pm.sample_posterior_predictive(trace, 100, model)
data_ppc = az.from_pymc3(trace=trace, posterior_predictive=y_pred)
az.plot_ppc(data_ppc, kind='kde')

ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
ax[0].legend(fontsize=15)

# don't actually need az, can just do in pm
# pm will fill in missing values for you
# plot_ppc doesn't work if data has missing values
# pm.math has some nice functions, including switch

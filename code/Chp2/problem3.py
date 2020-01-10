"""
Modify model_g, change the prior for the mean to a Gaussian distribution
centered at the empirical mean, and play with a couple of reasonable values for
the standard deviation of this prior. How robust/sensitive are the inferences
to these changes? What do you think of using a Gaussian, which is an unbounded
distribution (goes from -∞ to ∞), to model bounded data such as this? Remember
that we said it is not possible to get values below 0 or above 100.

- doesn't appear very sensitive at all
"""

import pymc3 as pm
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

data = Series(np.loadtxt('../data/chemical_shifts.csv'), name='shift')
data_mean = data.mean()

#########
# in book
#########
with pm.Model() as model_g:
  mu = pm.Uniform('mu', lower=40, upper=70)
  sigma = pm.HalfNormal('sigma', sd=10)
  y = pm.Normal('y', mu=mu, sd=sigma, observed=data)

  trace_g = pm.sample(1000)

pm.plot_trace(trace_g)

################
# modification 1
################
with pm.Model() as model1:
  mu = pm.Normal('mu', data_mean, 5)
  sigma = pm.HalfNormal('sigma', sd=10)
  y = pm.Normal('y', mu=mu, sd=sigma, observed=data)
  trace1 = pm.sample(1000)

pm.plot_trace(trace1)

################
# modification 2
################
with pm.Model() as model1:
  mu = pm.Normal('mu', data_mean, 10)
  sigma = pm.HalfNormal('sigma', sd=10)
  y = pm.Normal('y', mu=mu, sd=sigma, observed=data)
  trace2 = pm.sample(1000)

pm.plot_trace(trace2)
pm.plot_posterior(trace2)

"""
Using the data in the chemical_shifts.csv file, compute the empirical mean and
the standard deviation with and without outliers. Compare those results to the
Bayesian estimation using the Gaussian and Student's t-distribution. Repeat
the exercise by adding more outliers.
"""

import pymc3 as pm
import seaborn as sns
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

data = Series(np.loadtxt('../data/chemical_shifts.csv'), name='shift')
data_mean = data.mean()

data.loc[data < 60].describe()
# count    46.000000
# mean     52.952609
# std       2.219286
# min      47.720000
# 25%      51.527500
# 50%      52.760000
# 75%      54.595000
# max      57.480000

data.describe()
# count    48.000000
# mean     53.496458
# std       3.456198
# min      47.720000
# 25%      51.582500
# 50%      52.875000
# 75%      54.960000
# max      68.580000

# normal
with pm.Model() as model_g:
    mu = pm.Uniform('mu', lower=40, upper=70)
    sigma = pm.HalfNormal('sigma', sd=10)
    y = pm.Normal('y', mu=mu, sd=sigma, observed=data)
    trace_g = pm.sample(1000)

# students t
with pm.Model() as model_t:
    mu = pm.Uniform('mu', 40, 70)
    sigma = pm.HalfNormal('sigma', sd=10)
    v = pm.Exponential('v', 1/30)
    y = pm.StudentT('y', mu=mu, sd=sigma, nu=v, observed=data)
    trace_t = pm.sample(1000)

data2 = Series(data, copy=True)
data2[48] = 65
data2[49] = 63
data2[50] = 69

data2.loc[data2 < 60].describe()
data2.describe()

# add some outliers
with pm.Model() as model_g2:
    mu = pm.Uniform('mu', lower=40, upper=70)
    sigma = pm.HalfNormal('sigma', sd=10)
    y = pm.Normal('y', mu=mu, sd=sigma, observed=data2)
    trace_g2 = pm.sample(1000)

# students t
with pm.Model() as model_t2:
    mu = pm.Uniform('mu', 40, 70)
    sigma = pm.HalfNormal('sigma', sd=10)
    v = pm.Exponential('v', 1/30)
    y = pm.StudentT('y', mu=mu, sd=sigma, nu=v, observed=data2)
    trace_t2 = pm.sample(1000)

pm.summary(trace_g)
pm.summary(trace_g2)

pm.summary(trace_t)
pm.summary(trace_t2)

"""
- takeaways:
    - student's t better approximates sd of data once you remove outliers
    - normal better approximates sd of data leaving outliers in

- what's better? if trying to get at overall gaussian shape w/ some outliers,
students t i guess
"""

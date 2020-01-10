import pandas as pd
import scipy.stats as stats
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm
import seaborn as sns

################################################
## Confounding variables and redundant variables
################################################

np.random.seed(42)
N = 100
x_1 = np.random.normal(size=N)
x_2 = x_1 + np.random.normal(size=N, scale=1)
# x_2 = x_1 + np.random.normal(size=N, scale=0.01)
y = x_1 + np.random.normal(size=N)
X = np.vstack((x_1, x_2)).T

df = DataFrame(X, columns=['x0', 'x1'])
df['y'] = y

sns.pairplot(df)

# model 1: two independent variables x1 and x2
with pm.Model() as m_x1x2:
    b0 = pm.Normal('b0', mu=0, sd=10)
    b1 = pm.Normal('b1', mu=0, sd=10)
    b2 = pm.Normal('b2', mu=0, sd=10)
    e = pm.HalfCauchy('e', 5)

    mu = b0 + b1 * X[:, 0] + b2 * X[:, 1]

    y_pred = pm.Normal('y_pred', mu=mu, sd=e, observed=y)

    trace_x1x2 = pm.sample(2000)


# model 2: just x1
with pm.Model() as m_x1:
    b0 = pm.Normal('b0', mu=0, sd=10)
    b1 = pm.Normal('b1', mu=0, sd=10)
    e = pm.HalfCauchy('e', 5)

    mu = b0 + b1 * X[:, 0]

    y_pred = pm.Normal('y_pred', mu=mu, sd=e, observed=y)

    trace_x1 = pm.sample(2000)

# model 3: just x2
with pm.Model() as m_x2:
    b0 = pm.Normal('b0', mu=0, sd=10)
    b2 = pm.Normal('b2', mu=0, sd=10)
    e = pm.HalfCauchy('e', 5)

    mu = b0 + b2 * X[:, 1]

    y_pred = pm.Normal('y_pred', mu=mu, sd=e, observed=y)

    trace_x2 = pm.sample(2000)

az.plot_forest([trace_x1x2, trace_x1, trace_x2],
               model_names=['m_x1x2', 'm_x1', 'm_x2'],
               var_names=['b1', 'b2'],
               combined=False, colors='cycle', figsize=(8, 3))

##################################################################
# repeating the code from above, but with a lower value of `scale`
##################################################################

np.random.seed(42)
N = 100
x_1 = np.random.normal(size=N)
# x_2 = x_1 + np.random.normal(size=N, scale=1)
x_2 = x_1 + np.random.normal(size=N, scale=0.01)
y = x_1 + np.random.normal(size=N)
X = np.vstack((x_1, x_2)).T

df = DataFrame(X, columns=['x0', 'x1'])
df['y'] = y

sns.pairplot(df)

with pm.Model() as model_red:
    b0 = pm.Normal('b0', mu=0, sd=10)
    b = pm.Normal('b', mu=0, sd=10, shape=2)
    e = pm.HalfCauchy('e', 5)

    mu = b0 + pm.math.dot(X, b)
    y_pred = pm.Normal('y_pred', mu=mu, sd=e, observed=y)

    trace_red = pm.sample(2000)

az.plot_forest(trace_red, var_names=['b'], combined=True, figsize=(8, 2))
az.plot_pair(trace_red, var_names=['b'])

##########################
# Masking effect variables
##########################

np.random.seed(42)
N = 126
r = 0.8
x_1 = np.random.normal(size=N)
x_2 = np.random.normal(x_1, scale=(1 - r ** 2) ** 0.5)
y = np.random.normal(x_1 - x_2)
X = np.vstack((x_1, x_2)).T

df = DataFrame(X, columns=['x0', 'x1'])
df['y'] = y

sns.pairplot(df)

# model 1: two independent variables x1 and x2
with pm.Model() as m_x1x2:
    b0 = pm.Normal('b0', mu=0, sd=10)
    b1 = pm.Normal('b1', mu=0, sd=10)
    b2 = pm.Normal('b2', mu=0, sd=10)
    e = pm.HalfCauchy('e', 5)

    mu = b0 + b1 * X[:, 0] + b2 * X[:, 1]

    y_pred = pm.Normal('y_pred', mu=mu, sd=e, observed=y)

    trace_x1x2 = pm.sample(1000)

# model 2: just x1
with pm.Model() as m_x1:
    b0 = pm.Normal('b0', mu=0, sd=10)
    b1 = pm.Normal('b1', mu=0, sd=10)
    e = pm.HalfCauchy('e', 5)

    mu = b0 + b1 * X[:, 0]

    y_pred = pm.Normal('y_pred', mu=mu, sd=e, observed=y)

    trace_x1 = pm.sample(1000)

# model 3: just x2
with pm.Model() as m_x2:
    b0 = pm.Normal('b0', mu=0, sd=10)
    b2 = pm.Normal('b2', mu=0, sd=10)
    e = pm.HalfCauchy('e', 5)

    mu = b0 + b2 * X[:, 1]

    y_pred = pm.Normal('y_pred', mu=mu, sd=e, observed=y)

    trace_x2 = pm.sample(1000)


# In[54]:

az.plot_forest([trace_x1x2, trace_x1, trace_x2],
               model_names=['m_x1x2', 'm_x1', 'm_x2'],
               var_names=['b1', 'b2'],
               combined=True, colors='cycle', figsize=(8, 3))

plt.savefig('b11197_03_27.png', dpi=300, bbox_inches='tight')

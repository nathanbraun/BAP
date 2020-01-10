"""
Modify the tips example to make it robust to outliers. Try with one shared for
all groups and also with one per group. Run posterior predictive checks to
assess these three models.
"""
import pandas as pd
import random
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm
import numpy as np

tips = pd.read_csv('../data/tips.csv')

tip = tips['tip'].values
idx = pd.Categorical(tips['day'], categories=['Thur', 'Fri', 'Sat', 'Sun']).codes
groups = len(np.unique(idx))

# original version
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=10, shape=groups)
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)
    y = pm.Normal('y', mu=mu[idx], sd=sigma[idx], observed=tip)
    trace = pm.sample(5000)

"""
- apparently in pymc can only index groups by numbers (0-3) not other things
(thu-sun)
- need to pass mu=mu[idx] to llh

"""

# robust to outliers version
"""
- let's set it as an exponential distribution with a mean of 30
- allows for many small values up through 120 ish
"""

with pm.Model() as model1:
    mu = pm.Normal('mu', mu=0, sd=10, shape=groups)
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)
    v = pm.Exponential('v', 1/30)
    y = pm.StudentT('y', mu=mu[idx], sd=sigma[idx], nu=v, observed=tip)
    trace1 = pm.sample(5000)


# outliers, but own can vary
with pm.Model() as model2:
    mu = pm.Normal('mu', mu=0, sd=10, shape=groups)
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)
    v = pm.Exponential('v', 1/30, shape=groups)
    y = pm.StudentT('y', mu=mu[idx], sd=sigma[idx], nu=v[idx], observed=tip)
    trace2 = pm.sample(5000)

y_pred = pm.sample_posterior_predictive(trace, 100, model)
data_ppc = az.from_pymc3(trace=trace, posterior_predictive=y_pred)
ax0 = az.plot_ppc(data_ppc, kind='kde', mean=False)
plt.xlim(-2, 8)

y_pred1 = pm.sample_posterior_predictive(trace1, 100, model1)
data_ppc1 = az.from_pymc3(trace=trace, posterior_predictive=y_pred1)
az.plot_ppc(data_ppc1, kind='kde', mean=False)
plt.xlim(-2, 8)

# works best by far
y_pred2 = pm.sample_posterior_predictive(trace2, 100, model2)
data_ppc2 = az.from_pymc3(trace=trace, posterior_predictive=y_pred2)
az.plot_ppc(data_ppc2, kind='kde', mean=False)
plt.xlim(-2, 8)

"""
Compute the probability of superiority directly from the posterior (without
computing Cohen's d first). You can use the pm.sample_posterior_predictive()
function to take a sample from each group. Is it really different from the
calculation assuming normality? Can you explain the result?
"""

df = DataFrame(y_pred2['y'].T)

thu = df.loc[tips['day'] == 'Thur']
fri = df.loc[tips['day'] == 'Fri']
sat = df.loc[tips['day'] == 'Sat']
sun = df.loc[tips['day'] == 'Sun']

def compare2(df1, df2):
    nrows = min(len(df1), len(df2))
    return (df1.sample(nrows).reset_index(drop=True) >
            df2.sample(nrows).reset_index(drop=True)).mean().mean()

compare2(thu, fri)
compare2(thu, sat)
compare2(thu, sun)

compare2(fri, sat)
compare2(fri, sun)

compare2(sat, sun)

"""
Create a hierarchical version of the tips example by partially pooling across
the days of the week. Compare the results to those obtained without the
hierarchical structure.
"""

with pm.Model() as model_h:
    # hyper_priors
    mu_mu = pm.Normal('mu_mu', mu=0, sd=10)
    sigma_mu = pm.HalfNormal('sigma_mu', 10)

    # priors
    mu = pm.Normal('mu', mu=mu_mu, sd=sigma_mu, shape=groups)
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)

    y = pm.Normal('y', mu=mu[idx], sd=sigma[idx], observed=tip)
    trace = pm.sample(5000)

# with pm.Model() as model_h:
#     # hyper_priors
#     mu_mu = pm.Normal('mu_mu', mu=0, sd=10)
#     sigma_mu = pm.HalfNormal('sigma_mu', 10)

#     # priors
#     mu = pm.Normal('mu', mu=mu_mu, sd=sigma_mu, shape=groups)
#     sigma = pm.HalfNormal('sigma', sd=10, shape=groups)

#     y = pm.Normal('y', mu=mu[idx], sd=sigma[idx], observed=diff)

#     trace_cs_h = pm.sample(1000)

y_pred3 = pm.sample_posterior_predictive(trace, 100, model_h)
data_ppc3 = az.from_pymc3(trace=trace, posterior_predictive=y_pred3)
az.plot_ppc(data_ppc3, kind='kde', mean=False)
plt.xlim(-2, 8)

with pm.Model() as model4:
    # hyper_priors
    mu_mu = pm.Normal('mu_mu', mu=0, sd=10)
    sigma_mu = pm.HalfNormal('sigma_mu', 10)

    # priors
    mu = pm.Normal('mu', mu=mu_mu, sd=sigma_mu, shape=groups)
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)
    v = pm.Exponential('v', 1/30, shape=groups)

    y = pm.StudentT('y', mu=mu[idx], sd=sigma[idx], nu=v[idx], observed=tip)
    trace4 = pm.sample(5000)

# with pm.Model() as model_h:
#     # hyper_priors
#     mu_mu = pm.Normal('mu_mu', mu=0, sd=10)
#     sigma_mu = pm.HalfNormal('sigma_mu', 10)

#     # priors
#     mu = pm.Normal('mu', mu=mu_mu, sd=sigma_mu, shape=groups)
#     sigma = pm.HalfNormal('sigma', sd=10, shape=groups)

#     y = pm.Normal('y', mu=mu[idx], sd=sigma[idx], observed=diff)

#     trace_cs_h = pm.sample(1000)

y_pred4 = pm.sample_posterior_predictive(trace, 100, model_h)
data_ppc4 = az.from_pymc3(trace=trace, posterior_predictive=y_pred4)
az.plot_ppc(data_ppc4, kind='kde', mean=False)
plt.xlim(-2, 8)

with pm.Model() as model5:
    alpha = pm.Exponential('alpha', 1/30, shape=groups)
    beta = pm.Exponential('beta', 1/30, shape=groups)
    y = pm.Gamma('y', alpha=alpha[idx], beta=beta[idx], observed=tip)
    trace5 = pm.sample(5000)

y_pred5 = pm.sample_posterior_predictive(trace, 100, model5)
data_ppc5 = az.from_pymc3(trace=trace5, posterior_predictive=y_pred5)
az.plot_ppc(data_ppc5, kind='kde', mean=False)
plt.xlim(0, 8)

import pymc3 as pm
import seaborn as sns
import numpy as np
import scipy.stats as stats
import pandas as pd
import theano.tensor as tt
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(42)

cs = pd.read_csv('../data/chemical_shifts_theo_exp.csv')
cs_exp = cs['exp']

az.plot_kde(cs_exp)
plt.hist(cs_exp, density=True, bins=30, alpha=0.3)
plt.yticks([])

plt.show()

clusters = 2

# reparameterized
# says this should have issue w/ parameter non-identifiablity, but seems fine
with pm.Model() as model_mg:
    p = pm.Dirichlet('p', a=np.ones(clusters))
    means = pm.Normal('means', mu=cs_exp.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
    trace_mg = pm.sample(random_seed=123)

varnames = ['means', 'p']
az.plot_trace(trace_mg, varnames)
az.summary(trace_mg, varnames)

ppc_mg = pm.sample_posterior_predictive(trace_mg, 2000, model=model_mg)
data_ppc = az.from_pymc3(trace=trace_mg, posterior_predictive=ppc_mg)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)

clusters = 2
with pm.Model() as model_mgp:
    p = pm.Dirichlet('p', a=np.ones(clusters))
    means = pm.Normal('means', mu=np.array([.9, 1]) * cs_exp.mean(),
                      sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)

    # Potential adds a constraint to the model
    # sayi
    order_means = pm.Potential('order_means',
                               tt.switch(means[1]-means[0] < 0,
                                         -np.inf, 0))
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
    trace_mgp = pm.sample(1000, random_seed=123)

az.plot_trace(trace_mgp, varnames)
az.summary(trace_mgp, varnames)

clusters = [3, 4, 5, 6]
models = []
traces = []
for cluster in clusters:
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means', mu=cs_exp.mean(), sd=10, shape=cluster)
        # means = pm.Normal('means',
        #                   mu=np.linspace(cs_exp.min(), cs_exp.max(), cluster),
        #                   sd=10, shape=cluster,
        #                   transform=pm.distributions.transforms.ordered)
        sd = pm.HalfNormal('sd', sd=10)
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
        trace = pm.sample(1000, tune=2000, random_seed=123)
        traces.append(trace)
        models.append(model)

comp = az.compare(dict(zip(clusters, traces)), method='BB-pseudo-BMA')

ppc_mg = pm.sample_posterior_predictive(traces[-1], 2000, model=models[-1])
data_ppc = az.from_pymc3(trace=traces[-1], posterior_predictive=ppc_mg)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)

# above, just for cluster 6
cluster = 6
with pm.Model() as model_6:
    # seting equal prob of being assigned each cluster
    p = pm.Dirichlet('p', a=np.ones(cluster))

    # all of these have the same prior
    means = pm.Normal('means', mu=cs_exp.mean(), sd=10, shape=cluster)
    sd = pm.HalfNormal('sd', sd=10)

    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
    trace_6 = pm.sample(1000, tune=2000, random_seed=123)

ppc_mg = pm.sample_posterior_predictive(trace_6, 2000, model=model_6)
data_ppc = az.from_pymc3(trace=trace_6, posterior_predictive=ppc_mg)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)

# Dirichlet processes
K = 20

def stick_breaking(alpha, K):
    beta = pm.Beta('beta', 1., alpha, shape=K)
    w = beta * pm.math.concatenate([[1.], tt.extra_ops.cumprod(1. - beta)[:-1]])
    return w

with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1, 1.)
    w = pm.Deterministic('w', stick_breaking(alpha, K))
    means = pm.Normal('means',
                      mu=np.linspace(cs_exp.min(), cs_exp.max(), K),
                      sd=10, shape=K)

    sd = pm.HalfNormal('sd', sd=10, shape=K)
    obs = pm.NormalMixture('obs', w, means, sd=sd, observed=cs_exp.values)
    trace = pm.sample(1000, tune=2000, nuts_kwargs={'target_accept':0.85})


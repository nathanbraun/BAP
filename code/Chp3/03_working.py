import pandas as pd
import scipy.stats as stats
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm
import seaborn as sns

##########################
# Simple linear regression
##########################

# generating some fake data
np.random.seed(1)
N = 100
alpha_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alpha_real + beta_real * x
y = y_real + eps_real

# plot it
_, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(x, y, 'C0.')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0)
ax[0].plot(x, y_real, 'k')
az.plot_kde(y, ax=ax[1])
ax[1].set_xlabel('y')
plt.tight_layout()

# pymc it
with pm.Model() as model_g:
    alpha = pm.Normal('alpha', mu=0, sd=10)  # aka b0
    beta = pm.Normal('beta', mu=0, sd=1)  # aka b1
    epsilon = pm.HalfCauchy('epsilon', 5)

    mu = pm.Deterministic('mu', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=y)

    # could also do
    # y_pred = pm.Normal('y_pred', mu=alpha + beta*x, sd=epsilon, observed=y)

    trace_g = pm.sample(2000, tune=1000)

az.plot_trace(trace_g)
az.plot_trace(trace_g, var_names=['alpha', 'beta', 'epsilon'])


# looking at pairplot, very correlated
az.plot_pair(trace_g, var_names=['alpha', 'beta'], plot_kwargs={'alpha': 0.1})

# plotting confidence intervals and lines
# v1 - plotting a bunch of lines randomly drawn from trace
plt.plot(x, y, 'C0.')

alpha_m = trace_g['alpha'].mean()
beta_m = trace_g['beta'].mean()

draws = range(0, len(trace_g['alpha']), 10)
plt.plot(x, trace_g['alpha'][draws] + trace_g['beta'][draws]
         * x[:, np.newaxis], c='gray', alpha=0.5)

plt.plot(x, alpha_m + beta_m * x, c='k',
         label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')

plt.xlabel('x')
plt.ylabel('y', rotation=0)

# v2 - plotting hpd
plt.plot(x, alpha_m + beta_m * x, c='k',
         label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
sig = az.plot_hpd(x, trace_g['mu'], credible_interval=0.98, color='k')

plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

# v3 - plotting interval for future data
ppc = pm.sample_posterior_predictive(trace_g, samples=2000, model=model_g)

# way to get r2
az.r2_score(y, ppc['y_pred'])

# In[11]:

plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m * x, c='k',
         label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')

az.plot_hpd(x, ppc['y_pred'], credible_interval=0.5, color='gray', smooth=False)
az.plot_hpd(x, ppc['y_pred'], color='gray', smooth=False)

plt.xlabel('x')
plt.ylabel('y', rotation=0)

###############################################################################
# doing univariate linear regression via modeling as a multinomial distribution
###############################################################################
data = np.stack((x, y)).T
with pm.Model() as pearson_model:
    μ = pm.Normal('μ', mu=data.mean(0), sd=10, shape=2)
    σ_1 = pm.HalfNormal('σ_1', 10)
    σ_2 = pm.HalfNormal('σ_2', 10)
    ρ = pm.Uniform('ρ', -1., 1.)
    r2 = pm.Deterministic('r2', ρ**2)
    cov = pm.math.stack(([σ_1**2, σ_1*σ_2*ρ],
                         [σ_1*σ_2*ρ, σ_2**2]))
    y_pred = pm.MvNormal('y_pred', mu=μ, cov=cov, observed=data)
    trace_p = pm.sample(1000)

az.plot_trace(trace_p, var_names=['r2'])

##########################
# robust linear regression
##########################
ans = pd.read_csv('../data/anscombe.csv')
ans3 = ans.loc[ans['group'] == 'III']

# look at it
sns.regplot(ans3['x'], ans3['y'])
sns.kdeplot(ans3['y'])

# x_3 = ans[ans.group == 'III']['x'].values
# y_3 = ans[ans.group == 'III']['y'].values
x3_mean = ans3['x'].mean()
y3_mean = ans3['y'].mean()
ans3['x_centered'] = ans3['x'] - x3_mean

with pm.Model() as model_t:
    alpha = pm.Normal('alpha', mu=y3_mean, sd=1)  # aka b0
    beta = pm.Normal('beta', mu=0, sd=1)  # aka b1
    epsilon = pm.HalfNormal('epsilon', 5)
    v_ = pm.Exponential('v_', 1/29)
    v = pm.Deterministic('v', v_ + 1)

    y_pred = pm.StudentT('y_pred', mu=alpha + beta*ans3['x_centered'],
                         sd=epsilon, nu=v, observed=ans3['y'])

    # could also do
    # y_pred = pm.Normal('y_pred', mu=alpha + beta*x, sd=epsilon, observed=y)

    trace_t = pm.sample(2000)

plt.clf()
beta_c, alpha_c = stats.linregress(ans3['x_centered'], ans3['y'])[:2]

plt.plot(ans3['x_centered'], (alpha_c + beta_c * ans3['x_centered']), 'k',
         label='non-robust', alpha=0.5)
plt.plot(ans3['x_centered'], ans3['y'], 'C0o')
alpha_m = trace_t['alpha'].mean()
beta_m = trace_t['beta'].mean()
plt.plot(ans3['x_centered'], alpha_m + beta_m * ans3['x_centered'], c='k',
         label='robust')

plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend(loc=2)
plt.tight_layout()

# posterior predictive check
# note: this is weird
ppc = pm.sample_posterior_predictive(trace_t, samples=200, model=model_t,
                                     random_seed=2)


data_ppc = az.from_pymc3(trace=trace_t, posterior_predictive=ppc)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
plt.xlim(5.5, 8)

################################
# hierarchical linear regression
################################

# first, make some data
N = 20
M = 8
idx = np.repeat(range(M-1), N)
idx = np.append(idx, 7)
np.random.seed(314)

# drawing alpha, beta, from some parameters <- differ by group
# eps varies by observation
alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(6, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))

df = DataFrame(idx, columns=['group'])
df['x'] = np.random.normal(10, 1, len(idx))
df['eps'] = np.random.normal(0, 0.5, size=len(idx))


# generate x and y based on idx and draws
y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real

# do it pandas style
def get_y(row):
    group_ = int(row['group'])
    x_ = row['x']
    eps_ = row['eps']
    return alpha_real[group_] + beta_real[group_]*x_ + eps_
df['y'] = df.apply(get_y, axis=1)

sns.relplot(x='x', y='y', data=df, col='group', col_wrap=4, height=2)

# now center
df['x_centered'] = df['x'] - df['x'].mean()

idx = df['group'].values
y_m = df['y'].values
x_centered = df['x_centered'].values

with pm.Model() as unpooled_model:
    alpha_temp = pm.Normal('alpha_temp', mu=0, sd=10, shape=M)
    beta = pm.Normal('beta', mu=0, sd=10, shape=M)
    epsilon = pm.HalfCauchy('epsilon', 5)
    v = pm.Exponential('v', 1/30)
    y_pred = pm.StudentT('y_pred', mu=alpha_temp[idx] + beta[idx] * x_centered,
                         sd=epsilon, nu=v, observed=y_m)

    alpha = pm.Deterministic('alpha', alpha_temp - beta * df['x'].mean())
    trace_up = pm.sample(2000)

az.plot_forest(trace_up, var_names=['alpha', 'beta'], combined=True)

with pm.Model() as hierarchical_model:
    # hyper-priors
    alpha_mu_temp = pm.Normal('alpha_mu_temp', mu=0, sd=10)
    alpha_sigma_temp = pm.HalfNormal('alpha_sigma_temp', 10)
    beta_mu = pm.Normal('beta_mu', mu=0, sd=10)
    beta_sigma = pm.HalfNormal('beta_sigma', sd=10)

    # priors
    alpha_temp = pm.Normal('alpha_temp', mu=alpha_mu_temp, sd=alpha_sigma_temp, shape=M)
    beta = pm.Normal('beta', mu=beta_mu, sd=beta_sigma, shape=M)
    epsilon = pm.HalfCauchy('epsilon', 5)
    v = pm.Exponential('v', 1/30)

    y_pred = pm.StudentT('y_pred', mu=alpha_temp[idx] + beta[idx] * x_centered,
                         sd=epsilon, nu=v, observed=y_m)

    alpha = pm.Deterministic('alpha', alpha_temp - beta * x_m.mean())
    alpha_mu = pm.Deterministic('alpha_mu', alpha_mu_temp - beta_mu * x_m.mean())
    alpha_sigma = pm.Deterministic('alpha_sigma', alpha_sigma_temp - beta_mu * x_m.mean())

    trace_hm = pm.sample(1000)

az.plot_forest([trace_up, trace_hm], var_names=['alpha', 'beta'], combined=True)

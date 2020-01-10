#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
import arviz as az

az.style.use('arviz-darkgrid')

np.random.seed(123)
trials = 4
theta_real = 0.35  # unknown value in a real experiment
data = stats.bernoulli.rvs(p=theta_real, size=trials)
data

with pm.Model() as our_first_model:
    # a priori
    theta = pm.Beta('theta', alpha=1., beta=1.)
    # likelihood
    y = pm.Bernoulli('y', p=theta, observed=data)
    trace = pm.sample(3000, random_seed=123)


# ### Summarizing the posterior
az.plot_trace(trace)
plt.savefig('B11197_02_01.png')

az.summary(trace)
az.plot_posterior(trace)
plt.savefig('B11197_02_02.png', dpi=300)

az.plot_posterior(trace, rope=[0.45, .55])
plt.savefig('B11197_02_03.png', dpi=300)

az.plot_posterior(trace, ref_val=0.5)
plt.savefig('B11197_02_04.png', dpi=300)


grid = np.linspace(0, 1, 200)
theta_pos = trace['theta']
lossf_a = [np.mean(abs(i - θ_pos)) for i in grid]
lossf_b = [np.mean((i - θ_pos)**2) for i in grid]

for lossf, c in zip([lossf_a, lossf_b], ['C0', 'C1']):
    mini = np.argmin(lossf)
    plt.plot(grid, lossf, c)
    plt.plot(grid[mini], lossf[mini], 'o', color=c)
    plt.annotate('{:.2f}'.format(grid[mini]),
                 (grid[mini], lossf[mini] + 0.03), color=c)
    plt.yticks([])
    plt.xlabel(r'$\hat \theta$')
# plt.savefig('B11197_02_05.png', dpi=300)

def qloss(theta, pos):
    return np.mean([(theta - x) ** 2 for x in pos])

def aloss(theta, pos):
    return np.mean([np.abs(theta - x) for x in pos])

# In[11]:

qloss(.05, theta_pos)
qloss(.32, theta_pos)
qloss(.33, theta_pos)
qloss(.34, theta_pos)
qloss(.40, theta_pos)
qloss(.70, theta_pos)

aloss(.05, theta_pos)
aloss(.31, theta_pos)
aloss(.32, theta_pos)
aloss(.33, theta_pos)
aloss(.34, theta_pos)
aloss(.40, theta_pos)
aloss(.70, theta_pos)

np.mean(theta_pos), np.median(theta_pos)


# In[12]:


lossf = []
for i in grid:
    if i < 0.5:
        f = np.mean(np.pi * theta_pos / np.abs(i - theta_pos))
    else:
        f = np.mean(1 / (i - theta_pos))
    lossf.append(f)

mini = np.argmin(lossf)
plt.plot(grid, lossf)
plt.plot(grid[mini], lossf[mini], 'o')
plt.annotate('{:.2f}'.format(grid[mini]),
             (grid[mini] + 0.01, lossf[mini] + 0.1))
plt.yticks([])
plt.xlabel(r'$\hat \theta$')
plt.savefig('B11197_02_06.png', dpi=300)


# ## Gaussian inferences

# In[13]:


data = np.loadtxt('../data/chemical_shifts.csv')

# remove outliers using the interquartile rule
#quant = np.percentile(data, [25, 75])
#iqr = quant[1] - quant[0]
#upper_b = quant[1] + iqr * 1.5
#lower_b = quant[0] - iqr * 1.5
#data = data[(data > lower_b) & (data < upper_b)]
#print(np.mean(data), np.std(data))

az.plot_kde(data, rug=True)
plt.yticks([0], alpha=0)
plt.savefig('B11197_02_07.png', dpi=300)


#  <img src="B11197_02_08.png" width="500">

# In[14]:

with pm.Model() as model_g:
    μ = pm.Uniform('μ', lower=40, upper=70)
    σ = pm.HalfNormal('σ', sd=10)
    y = pm.Normal('y', mu=μ, sd=σ, observed=data)
    trace_g = pm.sample(3000)

az.plot_trace(trace_g)

# plt.savefig('B11197_02_09.png', dpi=300)
az.plot_joint(trace_g, kind='kde', fill_last=False)

# plt.savefig('B11197_02_10.png', dpi=300)

az.summary(trace_g)

y_pred_g = pm.sample_posterior_predictive(trace_g, 100, model_g)

data_ppc = az.from_pymc3(trace=trace_g, posterior_predictive=y_pred_g)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
ax[0].legend(fontsize=15)

plt.savefig('B11197_02_11.png', dpi=300)


# ### Robust inferences

# In[20]:


plt.figure(figsize=(10, 6))
x_values = np.linspace(-10, 10, 500)
for df in [1, 2, 30]:
    distri = stats.t(df)
    x_pdf = distri.pdf(x_values)
    plt.plot(x_values, x_pdf, label=fr'$\nu = {df}$', lw=3)

x_pdf = stats.norm.pdf(x_values)
plt.plot(x_values, x_pdf, 'k--', label=r'$\nu = \infty$')
plt.xlabel('x')
plt.yticks([])
plt.legend()
plt.xlim(-5, 5)
plt.savefig('B11197_02_12.png', dpi=300)


#  <img src="B11197_02_13.png" width="500">

# In[21]:


with pm.Model() as model_t:
    μ = pm.Uniform('μ', 40, 75)
    σ = pm.HalfNormal('σ', sd=10)
    ν = pm.Exponential('ν', 1/30)
    y = pm.StudentT('y', mu=μ, sd=σ, nu=ν, observed=data)
    trace_t = pm.sample(1000)

az.plot_trace(trace_t)
plt.savefig('B11197_02_14.png', dpi=300)


# In[23]:


az.summary(trace_t)


# In[24]:


y_ppc_t = pm.sample_posterior_predictive(
    trace_t, 100, model_t, random_seed=123)
y_pred_t = az.from_pymc3(trace=trace_t, posterior_predictive=y_ppc_t)
az.plot_ppc(y_pred_t, figsize=(12, 6), mean=False)
ax[0].legend(fontsize=15)
plt.xlim(40, 70)

plt.savefig('B11197_02_15.png', dpi=300)


# # Tips example

# In[25]:


tips = pd.read_csv('../data/tips.csv')
tips.tail()


# In[26]:


sns.violinplot(x='day', y='tip', data=tips)
plt.savefig('B11197_02_16.png', dpi=300)


# In[27]:


tip = tips['tip'].values
idx = pd.Categorical(tips['day'],
                     categories=['Thur', 'Fri', 'Sat', 'Sun']).codes
groups = len(np.unique(idx))


# In[28]:


with pm.Model() as comparing_groups:
    mu = pm.Normal('mu', mu=0, sd=10, shape=groups)
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)
    y = pm.Normal('y', mu=mu[idx], sd=sigma[idx], observed=tip)
    trace_cg = pm.sample(5000)

az.plot_trace(trace_cg)
plt.savefig('B11197_02_17.png', dpi=300)


# In[29]:


dist = stats.norm()

_, ax = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True)

comparisons = [(i, j) for i in range(4) for j in range(i+1, 4)]
pos = [(k, l) for k in range(3) for l in (0, 1)]

for (i, j), (k, l) in zip(comparisons, pos):
    means_diff = trace_cg['μ'][:, i] - trace_cg['μ'][:, j]
    d_cohen = (means_diff / np.sqrt((trace_cg['σ'][:, i]**2 + trace_cg['σ'][:, j]**2) / 2)).mean()
    ps = dist.cdf(d_cohen/(2**0.5))
    az.plot_posterior(means_diff, ref_val=0, ax=ax[k, l])
    ax[k, l].set_title(f'$\mu_{i}-\mu_{j}$')
    ax[k, l].plot(
        0, label=f"Cohen's d = {d_cohen:.2f}\nProb sup = {ps:.2f}", alpha=0)
    ax[k, l].legend()
plt.savefig('B11197_02_18.png', dpi=300)


# # Hierarchical Models

#  <img src="B11197_02_19.png" width="500">

# In[30]:


N_samples = [30, 30, 30]
G_samples = [18, 18, 18]  # [3, 3, 3]  [18, 3, 3]

group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

with pm.Model() as model_h:
    mu = pm.Beta('mu', 1., 1.)
    k = pm.HalfNormal('k', 10)
    theta = pm.Beta('theta', alpha=mu*k, beta=(1.0-mu)*k, shape=len(N_samples))
    y = pm.Bernoulli('y', p=theta[group_idx], observed=data)
    trace_h = pm.sample(2000)


sum30 = az.summary(trace_h)
sum3 = az.summary(trace_h)
sum_m = az.summary(trace_h)

az.plot_trace(trace_h)
plt.savefig('B11197_02_20.png', dpi=300)

# basically going through and plotting individual curves from all various
# traces

# use this to feed through the PDF and plot it
x = np.linspace(0, 1, 100)

for i in np.random.randint(0, len(trace_h), size=100):
    mu = trace_h['mu'][i]
    k = trace_h['k'][i]

    # picking one mu, k at a time, plotting curve with those in the beta
    pdf = stats.beta(mu*k, (1.0-mu)*k).pdf(x)
    plt.plot(x, pdf,  'C1', alpha=0.2)

# then plot with the mean
u_mean = trace_h['mu'].mean()
k_mean = trace_h['k'].mean()
dist = stats.beta(u_mean*k_mean, (1.0-u_mean)*k_mean)
pdf = dist.pdf(x)
mode = x[np.argmax(pdf)]
mean = dist.moment(1)
plt.plot(x, pdf, lw=3, label=f'mode = {mode:.2f}\nmean = {mean:.2f}')
plt.yticks([])

plt.legend()
plt.xlabel('$θ_{prior}$')
plt.tight_layout()
plt.savefig('B11197_02_21.png', dpi=300)


cs_data = pd.read_csv('../data/chemical_shifts_theo_exp.csv')
diff = cs_data.theo.values - cs_data.exp.values
idx = pd.Categorical(cs_data['aa']).codes
groups = len(np.unique(idx))


# In[36]:


with pm.Model() as cs_nh:
    mu = pm.Normal('mu', mu=0, sd=10, shape=groups)
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)
    y = pm.Normal('y', mu=mu[idx], sd=sigma[idx], observed=diff)
    trace_cs_nh = pm.sample(1000)

# In[37]:
with pm.Model() as cs_h:
    # hyper_priors
    mu_mu = pm.Normal('mu_mu', mu=0, sd=10)
    sigma_mu = pm.HalfNormal('sigma_mu', 10)

    # priors
    mu = pm.Normal('mu', mu=mu_mu, sd=sigma_mu, shape=groups)
    sigma = pm.HalfNormal('sigma', sd=10, shape=groups)

    y = pm.Normal('y', mu=mu[idx], sd=sigma[idx], observed=diff)

    trace_cs_h = pm.sample(1000)


# In[38]:

_, axes = az.plot_forest([trace_cs_nh, trace_cs_h],
                         model_names=['n_h', 'h'],
                         var_names='mu', combined=False, colors='cycle')
y_lims = axes[0].get_ylim()
axes[0].vlines(trace_cs_h['mu_mu'].mean(), *y_lims)

plt.savefig('B11197_02_22.png', dpi=300)


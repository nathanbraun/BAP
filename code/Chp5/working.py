import pymc3 as pm
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame, Series
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

order = 2

dummy_data = np.loadtxt('../data/dummy.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]


# alt
dd = pd.read_csv('../data/dummy.csv', sep=' ', header=None, names=['x', 'y'])
for i in range(2, order+1):
    dd[f'x{i}'] = dd['x']**i

# normalize
dd_norm = (dd - dd.mean())/dd.std()

# adding in a column with x^i for i = 1 through order
x_1p = np.vstack([x_1**i for i in range(1, order+1)])

x_1s = ((x_1p - x_1p.mean(axis=1, keepdims=True)) /
        x_1p.std(axis=1, keepdims=True))
y_1s = (y_1 - y_1.mean()) / y_1.std()

plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('B11197_05_01.png', dpi=300)

# basic linear modell
with pm.Model() as model_l:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=10)
    epsilon = pm.HalfNormal('epsilon', 5)

    # mu = alpha + beta * x_1s[0]
    mu = alpha + beta * dd_norm['x']

    y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=dd_norm['y'])
    trace_l = pm.sample(2000)

with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=10, shape=order)
    epsilon = pm.HalfNormal('epsilon', 5)

    # mu = alpha + pm.math.dot(beta, x_1s)

    # note: pandas OK here, but need to transpose it
    mu = alpha + pm.math.dot(beta, dd_norm[['x', 'x2']].T)

    y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=dd_norm['y'])

    trace_p = pm.sample(2000)


# note: 2k simulations, each of which is a row
# each sim/row has 33 columns, which is size of original dataset
y_l = pm.sample_posterior_predictive(trace_l, 2000, model=model_l)['y_pred']
y_p = pm.sample_posterior_predictive(trace_p, 2000, model=model_p)['y_pred']

####################################
# interquartile range error bar plot
####################################
plt.figure(figsize=(8, 3))
data = [y_1s, y_l, y_p]
labels = ['data', 'linear model', 'order 2']
for i, d in enumerate(data):
    mean = d.mean()
    err = np.percentile(d, [25, 75])
    plt.errorbar(mean, -i, xerr=[[-err[0]], [err[1]]], fmt='o')
    plt.text(mean, -i+0.2, labels[i], ha='center', fontsize=14)
plt.ylim([-i-0.5, 0.5])
plt.yticks([])
# plt.savefig('B11197_05_03.png', dpi=300)

################################
# distribution of mean, iqr plot
################################

fig, ax = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

# just the 75% percentile - 25% percentile
def iqr(x, a=0):
    return np.subtract(*np.percentile(x, [75, 25], axis=a))

def iqr2(x, a=0):
    ss = dd_norm['y'].describe()
    return ss['75%'] - ss['25%']

((DataFrame(y_l).quantile(.75, axis=1) - DataFrame(y_l).quantile(.25, axis=1)) >= iqr(dd_norm['y'])).mean()
((DataFrame(y_p).quantile(.75, axis=1) - DataFrame(y_p).quantile(.25, axis=1)) >= iqr(dd_norm['y'])).mean()

stats_l = pd.concat(
    [DataFrame(y_l).mean(axis=1),
     (DataFrame(y_l).quantile(.75, axis=1) - DataFrame(y_l).quantile(.25, axis=1))], axis=1)
stats_l.columns = ['mean', 'iqr']
stats_l['model'] = 'C1'

stats_p = pd.concat(
    [DataFrame(y_p).mean(axis=1),
     (DataFrame(y_p).quantile(.75, axis=1) - DataFrame(y_p).quantile(.25, axis=1))], axis=1)
stats_p.columns = ['mean', 'iqr']
stats_p['model'] = 'C2'

stats = pd.concat([stats_l, stats_p], ignore_index=True)

stats2 = pd.concat([
    stats[['mean', 'model']].rename(columns={'mean': 'value'}).assign(stat = 'mean'),
    stats[['iqr', 'model']].rename(columns={'iqr': 'value'}).assign(stat = 'iqr')
], ignore_index=True)

g = sns.FacetGrid(stats2, hue='model', col='stat').map(sns.kdeplot, 'value',
                                                       shade='True')

for idx, func in enumerate([np.mean, iqr]):
    T_obs = func(y_1s)
    ax[idx].axvline(T_obs, 0, 1, color='k', ls='--')
    for d_sim, c in zip([y_l, y_p], ['C1', 'C2']):
        T_sim = func(d_sim, 1)
        p_value = np.mean(T_sim >= T_obs)
        az.plot_kde(T_sim, plot_kwargs={'color': c},
                    label=f'p-value {p_value:.2f}', ax=ax[idx])
    ax[idx].set_title(func.__name__)
    ax[idx].set_yticks([])
    ax[idx].legend()


###########################################
# Computing information criteria with PyMC3
###########################################

waic_l = az.waic(trace_l)
waic_l


cmp_df = az.compare({'model_l':trace_l, 'model_p':trace_p},
                    method='BB-pseudo-BMA')
cmp_df

########################
# computing bayes factor
########################

coins = 30 # 300
heads = 9 # 90
y_d = np.repeat([0, 1], [coins-heads, heads])

with pm.Model() as model_BF_0:
    theta = pm.Beta('theta', 4, 8)
    y = pm.Bernoulli('y', theta, observed=y_d)
    trace_BF_0 = pm.sample(2500, step=pm.SMC())

with pm.Model() as model_BF_1:
    theta = pm.Beta('theta', 8, 4)
    y = pm.Bernoulli('y', theta, observed=y_d)
    trace_BF_1 = pm.sample(2500, step=pm.SMC())


# In[19]:


model_BF_0.marginal_likelihood / model_BF_1.marginal_likelihood

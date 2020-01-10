import pymc3 as pm
import numpy as np
import seaborn as sns
import pandas as pd
from theano import shared
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

## Variable variance

data = pd.read_csv('../data/babies.csv')
data.plot.scatter('Month', 'Lenght')

with pm.Model() as model_vv:
    b0 = pm.Normal('b0', sd=10)
    b1 = pm.Normal('b1', sd=10)
    gamma = pm.HalfNormal('gamma', sd=10)
    delta = pm.HalfNormal('delta', sd=10)

    # what is this?
    # shared: way to change values of x after fitting without needing to refit
    x_shared = shared(data.Month.values * 1.)

    # note: function of sqrt of x
    # says it's a way to fit it to a curve
    mu = pm.Deterministic('mu', b0 + b1 * x_shared**0.5)
    e = pm.Deterministic('e', gamma + delta * x_shared)

    y_pred = pm.Normal('y_pred', mu=mu, sd=e, observed=data.Lenght)
    trace_vv = pm.sample(1000, tune=1000)


# In[57]:

plt.plot(data.Month, data.Lenght, 'C0.', alpha=0.1)

mu_m = trace_vv['mu'].mean(0)
e_m = trace_vv['e'].mean(0)

plt.plot(data.Month, mu_m, c='k')
plt.fill_between(data.Month, mu_m + 1 * e_m, mu_m -
                 1 * e_m, alpha=0.6, color='C1')
plt.fill_between(data.Month, mu_m + 2 * e_m, mu_m -
                 2 * e_m, alpha=0.4, color='C1')

plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.savefig('B11197_03_29.png', dpi=300)


# In[58]:

x_shared.set_value([0.5])
ppc = pm.sample_posterior_predictive(trace_vv, 2000, model=model_vv)
y_ppc = ppc['y_pred'][:, 0]

sns.kdeplot(y_ppc)

# In[59]:


ref = 53
density, l, u = az._fast_kde(y_ppc)
x_ = np.linspace(l, u, 200)
plt.plot(x_, density)
percentile = int(sum(y_ppc <= ref) / len(y_ppc) * 100)
plt.fill_between(x_[x_ < ref], density[x_ < ref],
                 label='percentile = {:2d}'.format(percentile))
plt.xlabel('length')
plt.yticks([])
plt.legend()
plt.savefig('B11197_03_30.png', dpi=300)


# In[ ]:


x_4 = ans[ans.group == 'IV']['x'].values
y_4 = ans[ans.group == 'IV']['y'].values

with pm.Model() as model_t2:
    b0 = pm.Normal('b0', mu=0, sd=100)
    b = pm.Normal('b', mu=0, sd=1)
    e = pm.HalfCauchy('e', 5)
    ν = pm.Exponential('ν', 1/30)
    #ν = pm.Gamma('ν', mu=20, sd=15)
    #ν = pm.Gamma('ν', 2, 0.1)

    y_pred = pm.StudentT('y_pred', mu=b0 + b * x_4, sd=e, nu=ν, observed=y_4)
    trace_t2 = pm.sample(2000)


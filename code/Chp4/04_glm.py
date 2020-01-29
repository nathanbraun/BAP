#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import seaborn as sns
import scipy.stats as stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import arviz as az


# In[2]:


az.style.use('arviz-darkgrid')


# ## Logistic regression

# In[3]:


z = np.linspace(-8, 8)
plt.plot(z, 1 / (1 + np.exp(-z)))
plt.xlabel('z')
plt.ylabel('logistic(z)')
plt.savefig('B11197_04_01.png', dpi=300);


# ## The iris dataset

# In[4]:


iris = pd.read_csv('../data/iris.csv')
iris.head()


# In[5]:


sns.stripplot(x="species", y="sepal_length", data=iris, jitter=True)
plt.savefig('B11197_04_02.png', dpi=300);


# In[6]:


sns.pairplot(iris, hue='species', diag_kind='kde')
plt.savefig('B11197_04_03.png', dpi=300, bbox_inches='tight');


# ### The logistic model applied to the iris dataset

# In[7]:


df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()


# In[8]:


with pm.Model() as model_0:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    mu = alpha + pm.math.dot(x_c, beta)
    theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
    bd = pm.Deterministic('bd', -alpha/beta)

    yl = pm.Bernoulli('yl', p=theta, observed=y_0)

    trace_0 = pm.sample(1000)


# In[9]:


varnames = ['alpha', 'beta', 'bd']
az.summary(trace_0, varnames)


# In[10]:


theta = trace_0['theta'].mean(axis=0)
idx = np.argsort(x_c)
plt.plot(x_c[idx], theta[idx], color='C2', lw=3)
plt.vlines(trace_0['bd'].mean(), 0, 1, color='k')
bd_hpd = az.hpd(trace_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)

plt.scatter(x_c, np.random.normal(y_0, 0.02),
            marker='.', color=[f'C{x}' for x in y_0])
az.plot_hpd(x_c, trace_0['theta'], color='C2')

plt.xlabel(x_n)
plt.ylabel('theta', rotation=0)
# use original scale for xticks
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))
plt.savefig('B11197_04_04.png', dpi=300)


# # Multiple logistic regression

# In[11]:


df = iris.query("species == ('setosa', 'versicolor')")
y_1 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_1 = df[x_n].values


# In[12]:


with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_1, beta)
    theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
    bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:,0])

    yl = pm.Bernoulli('yl', p=theta, observed=y_1)

    trace_1 = pm.sample(2000)


# In[13]:


varnames = ['alpha', 'beta']
az.plot_forest(trace_1, var_names=varnames);


# In[14]:


idx = np.argsort(x_1[:,0])
bd = trace_1['bd'].mean(0)[idx]
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in y_0])
plt.plot(x_1[:,0][idx], bd, color='k');

az.plot_hpd(x_1[:,0], trace_1['bd'], color='k')

plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
plt.savefig('B11197_04_05.png', dpi=300);


# ## Interpreting the coefficients of a logistic regression

# In[15]:


probability = np.linspace(0.01, 1, 100)
odds = probability / (1 - probability)

_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(probability, odds, 'C0')
ax2.plot(probability, np.log(odds), 'C1')

ax1.set_xlabel('probability')
ax1.set_ylabel('odds', color='C0')
ax2.set_ylabel('log-odds', color='C1')
ax1.grid(False)
ax2.grid(False)
plt.savefig('B11197_04_06.png', dpi=300);


# In[16]:


df = az.summary(trace_1, var_names=varnames)
df


# In[17]:


x_1 = 4.5  # sepal_length
x_2 = 3   # sepal_width

log_odds_versicolor_i = (df['mean'] * [1, x_1, x_2]).sum()
probability_versicolor_i = logistic(log_odds_versicolor_i)


log_odds_versicolor_f = (df['mean'] * [1, x_1 + 1, x_2]).sum()
probability_versicolor_f = logistic(log_odds_versicolor_f)

log_odds_versicolor_f - log_odds_versicolor_i, probability_versicolor_f - probability_versicolor_i


# ## Dealing with correlated variables

# In[18]:


corr = iris[iris['species'] != 'virginica'].corr()
mask = np.tri(*corr.shape).T
sns.heatmap(corr.abs(), mask=mask, annot=True, cmap='viridis')
plt.savefig('B11197_04_07.png', dpi=300, bbox_inches='tight');


# ## Dealing with unbalanced classes

# In[19]:


df = iris.query("species == ('setosa', 'versicolor')")
df = df[45:]
y_3 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_3 = df[x_n].values


# In[20]:


with pm.Model() as model_3:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_3, beta)
    theta = 1 / (1 + pm.math.exp(-mu))
    bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_3[:,0])

    yl = pm.Bernoulli('yl', p=theta, observed=y_3)

    trace_3 = pm.sample(1000)


# In[21]:


#az.plot_trace(trace_3, varnames);


# In[22]:


idx = np.argsort(x_3[:,0])
bd = trace_3['bd'].mean(0)[idx]
plt.scatter(x_3[:,0], x_3[:,1], c= [f'C{x}' for x in y_3])
plt.plot(x_3[:,0][idx], bd, color='k')

az.plot_hpd(x_3[:,0], trace_3['bd'], color='k')

plt.xlabel(x_n[0])
plt.ylabel(x_n[1])

plt.savefig('B11197_04_08.png', dpi=300);


# ## Softmax regression

# In[23]:


iris = sns.load_dataset('iris')
y_s = pd.Categorical(iris['species']).codes
x_n = iris.columns[:-1]
x_s = iris[x_n].values
x_s = (x_s - x_s.mean(axis=0)) / x_s.std(axis=0)


# In[24]:


with pm.Model() as model_s:
    alpha = pm.Normal('alpha', mu=0, sd=5, shape=3)
    beta = pm.Normal('beta', mu=0, sd=5, shape=(4,3))
    mu = pm.Deterministic('mu', alpha + pm.math.dot(x_s, beta))
    theta = tt.nnet.softmax(mu)
    yl = pm.Categorical('yl', p=theta, observed=y_s)
    trace_s = pm.sample(2000)


# In[25]:


#az.plot_forest(trace_s, var_names=['alpha', 'beta']);


# In[26]:


data_pred = trace_s['mu'].mean(0)

y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0)
          for point in data_pred]

f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}'


# In[27]:


with pm.Model() as model_sf:
    alpha = pm.Normal('alpha', mu=0, sd=2, shape=2)
    beta = pm.Normal('beta', mu=0, sd=2, shape=(4,2))
    alpha_f = tt.concatenate([[0] ,alpha])
    beta_f = tt.concatenate([np.zeros((4,1)) , beta], axis=1)
    mu = alpha_f + pm.math.dot(x_s, beta_f)
    theta = tt.nnet.softmax(mu)
    yl = pm.Categorical('yl', p=theta, observed=y_s)
    trace_sf = pm.sample(1000)


# ## Discriminative and generative models

# In[28]:


with pm.Model() as lda:
    mu = pm.Normal('mu', mu=0, sd=10, shape=2)
    σ = pm.HalfNormal('σ', 10)
    setosa = pm.Normal('setosa', mu=mu[0], sd=σ, observed=x_0[:50])
    versicolor = pm.Normal('versicolor', mu=mu[1], sd=σ,
                           observed=x_0[50:])
    bd = pm.Deterministic('bd', (mu[0] + mu[1]) / 2)
    trace_lda = pm.sample(1000)


# In[29]:


plt.axvline(trace_lda['bd'].mean(), ymax=1, color='C1')
bd_hpd = az.hpd(trace_lda['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='C1', alpha=0.5)

plt.plot(x_0, np.random.normal(y_0, 0.02), '.', color='k')
plt.ylabel('theta', rotation=0)
plt.xlabel('sepal_length')
plt.savefig('B11197_04_09.png', dpi=300)


# In[30]:


az.summary(trace_lda)


# ### The Poisson distribution

# In[31]:


mu_params = [0.5, 1.5, 3, 8]
x = np.arange(0, max(mu_params) * 3)
for mu in mu_params:
    y = stats.poisson(mu).pmf(x)
    plt.plot(x, y, 'o-', label=f'mu = {mu:3.1f}')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('B11197_04_10.png', dpi=300);


# ## The Zero-Inflated Poisson model

# In[32]:


n = 100
theta_real = 2.5
psi = 0.1

# Simulate some data
counts = np.array([(np.random.random() > (1-psi)) *
                   np.random.poisson(theta_real) for i in range(n)])


# In[33]:


with pm.Model() as ZIP:
    psi = pm.Beta('psi', 1, 1)
    theta = pm.Gamma('theta', 2, 0.1)
    y = pm.ZeroInflatedPoisson('y', psi, theta,
                               observed=counts)
    trace = pm.sample(1000)


# In[34]:


az.plot_trace(trace)
plt.savefig('B11197_04_11.png', dpi=300);


# In[35]:


#az.summary(trace)


# ## Poisson regression and ZIP regression

# In[36]:


fish_data = pd.read_csv('../data/fish.csv')


# In[37]:


with pm.Model() as ZIP_reg:
    psi = pm.Beta('psi', 1, 1)
    alpha = pm.Normal('alpha', 0, 10)
    beta = pm.Normal('beta', 0, 10, shape=2)
    theta = pm.math.exp(alpha + beta[0] * fish_data['child'] + beta[1] * fish_data['camper'])
    yl = pm.ZeroInflatedPoisson('yl', psi, theta, observed=fish_data['count'])
    trace_ZIP_reg = pm.sample(1000)
az.plot_trace(trace_ZIP_reg);


# In[38]:


az.summary(trace_ZIP_reg)


# In[39]:


children = [0, 1, 2, 3, 4]
fish_count_pred_0 = []
fish_count_pred_1 = []
for n in children:
    without_camper = trace_ZIP_reg['alpha'] + trace_ZIP_reg['beta'][:,0] * n
    with_camper = without_camper + trace_ZIP_reg['beta'][:,1]
    fish_count_pred_0.append(np.exp(without_camper))
    fish_count_pred_1.append(np.exp(with_camper))


plt.plot(children, fish_count_pred_0, 'C0.', alpha=0.01)
plt.plot(children, fish_count_pred_1, 'C1.', alpha=0.01)

plt.xticks(children);
plt.xlabel('Number of children')
plt.ylabel('Fish caught')
plt.plot([], 'C0o', label='without camper')
plt.plot([], 'C1o', label='with camper')
plt.legend()
plt.savefig('B11197_04_12.png', dpi=300);


# ## Robust logistic regression

# In[40]:


iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values
y_0 = np.concatenate((y_0, np.ones(6, dtype=int)))
x_0 = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.4]))
x_c = x_0 - x_0.mean()
plt.plot(x_c, y_0, 'o', color='k');


# In[41]:


with pm.Model() as model_rlg:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    mu = alpha + x_c *  beta
    theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
    bd = pm.Deterministic('bd', -alpha/beta)

    pi = pm.Beta('pi', 1., 1.)
    p = pi * 0.5 + (1 - pi) * theta

    yl = pm.Bernoulli('yl', p=p, observed=y_0)

    trace_rlg = pm.sample(1000)


# In[42]:


az.plot_trace(trace_rlg, varnames);


# In[43]:


varnames = ['alpha', 'beta', 'bd']
az.summary(trace_rlg, varnames)


# In[44]:


theta = trace_rlg['theta'].mean(axis=0)
idx = np.argsort(x_c)
plt.plot(x_c[idx], theta[idx], color='C2', lw=3);
plt.vlines(trace_rlg['bd'].mean(), 0, 1, color='k')
bd_hpd = az.hpd(trace_rlg['bd'])

plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)

plt.scatter(x_c, np.random.normal(y_0, 0.02), marker='.', color=[f'C{x}' for x in y_0])
theta_hpd = az.hpd(trace_rlg['theta'])[idx]
plt.fill_between(x_c[idx], theta_hpd[:,0], theta_hpd[:,1], color='C2', alpha=0.5)

plt.xlabel(x_n)
plt.ylabel('theta', rotation=0)
# use original scale for xticks
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))
plt.savefig('B11197_04_13.png', dpi=300);


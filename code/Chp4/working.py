import pandas as pd
import numpy as np
import theano.tensor as tt
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt


iris = pd.read_csv('../data/iris.csv')

#######################
# single variable logit
#######################
df = iris.query("species == ('setosa', 'versicolor')")

y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()

# logistic regression
with pm.Model() as model_0:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    mu = alpha + pm.math.dot(x_c, beta)
    theta = pm.Deterministic('theta', pm.math.sigmoid(mu))

    bd = pm.Deterministic('bd', -alpha/beta)

    yl = pm.Bernoulli('yl', p=theta, observed=y_0)

    trace_0 = pm.sample(1000)

varnames = ['alpha', 'beta', 'bd']
pm.summary(trace_0, varnames)
pm.plot_trace(trace_0, varnames)

#######################
# multi variable logit
#######################

df = iris.query("species == ('setosa', 'versicolor')")
y_1 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
# note: not centering this time
x_1 = df[x_n].values

with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_1, beta)

    # this time manually did it intead of pm.math.sigmoid
    theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))

    bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:,0])

    yl = pm.Bernoulli('yl', p=theta, observed=y_1)

    trace_1 = pm.sample(2000)

varnames = ['alpha', 'beta']
pm.plot_forest(trace_1, var_names=varnames);

pm.summary(trace_1, ['alpha', 'beta'])

############################################
# multi variable logit w/ unbalanced classes
############################################

df = iris.query("species == ('setosa', 'versicolor')")
df = df[45:]
y_3 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_3 = df[x_n].values


with pm.Model() as model_3:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_3, beta)
    theta = 1 / (1 + pm.math.exp(-mu))
    bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_3[:,0])

    yl = pm.Bernoulli('yl', p=theta, observed=y_3)

    trace_3 = pm.sample(1000)

###########################################
# softmax (logit but with multiple classes)
###########################################

iris = sns.load_dataset('iris')
y_s = pd.Categorical(iris['species']).codes
x_n = iris.columns[:-1]
x_s = iris[x_n].values
x_s = (x_s - x_s.mean(axis=0)) / x_s.std(axis=0)

with pm.Model() as model_s:
    # note alpha shape: 3 potential outputs
    alpha = pm.Normal('alpha', mu=0, sd=5, shape=3)
    # note beta shape: 4 explanatory variables, 3 potential outputs
    beta = pm.Normal('beta', mu=0, sd=5, shape=(4,3))

    mu = pm.Deterministic('mu', alpha + pm.math.dot(x_s, beta))

    theta = tt.nnet.softmax(mu)

    yl = pm.Categorical('yl', p=theta, observed=y_s)
    trace_s = pm.sample(2000)

data_pred = trace_s['mu'].mean(0)

y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0)
          for point in data_pred]

f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}'

###############################
# softmax with fewer parameters
###############################

with pm.Model() as model_sf:
    alpha = pm.Normal('alpha', mu=0, sd=2, shape=2)
    beta = pm.Normal('beta', mu=0, sd=2, shape=(4,2))
    alpha_f = tt.concatenate([[0] ,alpha])
    beta_f = tt.concatenate([np.zeros((4,1)) , beta], axis=1)
    mu = alpha_f + pm.math.dot(x_s, beta_f)
    theta = tt.nnet.softmax(mu)
    yl = pm.Categorical('yl', p=theta, observed=y_s)
    trace_sf = pm.sample(1000)

#####
# LDA
#####

with pm.Model() as lda:
    mu = pm.Normal('mu', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', 10)
    # assuming sepal length normally distributed
    # put mu and sigma parameters on it - note: same sigma
    setosa = pm.Normal('setosa', mu=mu[0], sd=sigma, observed=x_0[:50])
    versicolor = pm.Normal('versicolor', mu=mu[1], sd=sigma,
                           observed=x_0[50:])
    # identifying bd, average of two means
    bd = pm.Deterministic('bd', (mu[0] + mu[1]) / 2)
    trace_lda = pm.sample(1000)

#######################
# zero-inflated poisson
#######################

n = 100
theta_real = 2.5
psi = 0.1

# Simulate some data
counts = np.array([(np.random.random() > (1-psi)) *
                   np.random.poisson(theta_real) for i in range(n)])

with pm.Model() as ZIP:
    # psi is prob being modeled by poisson
    # 1 - psi is prob of extra zeros
    psi = pm.Beta('psi', 1, 1)
    theta = pm.Gamma('theta', 2, 0.1)
    y = pm.ZeroInflatedPoisson('y', psi, theta,
                               observed=counts)
    trace = pm.sample(1000)

#######################################
# Poisson regression and ZIP regression
#######################################

fish_data = pd.read_csv('../data/fish.csv')

with pm.Model() as ZIP_reg:
    psi = pm.Beta('psi', 1, 1)
    alpha = pm.Normal('alpha', 0, 10)
    beta = pm.Normal('beta', 0, 10, shape=2)
    theta = pm.math.exp(alpha + beta[0] * fish_data['child'] + beta[1] *
                        fish_data['camper'])
    yl = pm.ZeroInflatedPoisson('yl', psi, theta, observed=fish_data['count'])
    trace_ZIP_reg = pm.sample(1000)

pm.plot_trace(trace_ZIP_reg)
pm.summary(trace_ZIP_reg)


############################
# Robust logistic regression
############################

iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values
y_0 = np.concatenate((y_0, np.ones(6, dtype=int)))
x_0 = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.4]))
x_c = x_0 - x_0.mean()

with pm.Model() as model_rlg:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    mu = alpha + x_c *  beta
    theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
    bd = pm.Deterministic('bd', -alpha/beta)

    pi = pm.Beta('pi', 1., 1.)

    # saying prob is random (0.5) with prob pi, or logistic with 1-pi
    # way to do a robust logit
    p = pi * 0.5 + (1 - pi) * theta

    yl = pm.Bernoulli('yl', p=p, observed=y_0)

    trace_rlg = pm.sample(2000)

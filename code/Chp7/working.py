import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import arviz as az


np.random.seed(42)

# thing with this one:  each point is completely independent of others
x = np.linspace(0, 1, 10)
# y = np.random.normal(0, 1, len(x))
y = np.random.multivariate_normal(np.zeros_like(x), np.eye(len(x)))

first_one = list(zip(x, y))

# y is distributed as a series of draws from draws from N(-1, 1), N(-2
# thing with this one: mean of y_i+1 is value of y_i
# but more general approach to capturing dependencies, and not just beteween
# consecutive points
y2 = np.zeros_like(x)
for i in range(len(x)):
    y2[i] = np.random.normal(y2[i-1], 1)
    print(f'{y2[i]} draw from N({y2[i-1]}, 1)')

second_one = list(zip(x, y))

# doing it with a more flexible framework
np.random.multivariate_normal(np.zeros_like(x), np.eye(len(x)))

def exp_quad_kernel(x, knots, l=1):
    """exponentiated quadratic kernel"""
    return np.array([np.exp(-(x-k)**2 / (2*l**2)) for k in knots])

#############################
# Gaussian Process regression
#############################

np.random.seed(42)
x = np.random.uniform(0, 10, size=15)
y = np.random.normal(np.sin(x), 0.1)

plt.plot(x, y, 'o')
true_x = np.linspace(0, 10, 200)
plt.plot(true_x, np.sin(true_x), 'k--')
plt.xlabel('x')
plt.ylabel('f(x)', rotation=0)
plt.savefig('B11197_07_04.png')

# A one dimensional column vector of inputs.
X = x[:, None]

# 100 evenly spaced numbers from 0 to 10 (based on data)
X_new = np.linspace(np.floor(x.min()), np.ceil(x.max()), 100)[:,None]

with pm.Model() as model_reg:
    # hyperprior for lengthscale kernel parameter
    l = pm.Gamma('l', 2, 0.5)
    # instanciate a covariance function
    cov = pm.gp.cov.ExpQuad(1, ls=l)
    # instanciate a GP prior
    gp = pm.gp.Marginal(cov_func=cov)
    # prior
    ϵ = pm.HalfNormal('ϵ', 25)
    # likelihood
    y_pred = gp.marginal_likelihood('y_pred', X=X, y=y, noise=ϵ)
    trace_reg = pm.sample(2000)

    f_pred = gp.conditional('f_pred', X_new)

az.plot_trace(trace_reg)

pred_samples = pm.sample_posterior_predictive(trace_reg, vars=[f_pred],
                                              samples=82, model=model_reg)


_, ax = plt.subplots(figsize=(12,5))
ax.plot(X_new, pred_samples['f_pred'].T, 'C1-', alpha=0.3)
ax.plot(X, y, 'ko')
ax.set_xlabel('X')

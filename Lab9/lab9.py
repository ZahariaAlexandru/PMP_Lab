from random import random

import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
#import theano
#theano.config.blas__ldflags = ''


if __name__ == '__main__':
    az.style.use('arviz-darkgrid')
    dummy_data = np.loadtxt('date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    with pm.Model() as model_p:
        alpha = pm.Normal('α', mu=0, sd=1)
        beta = pm.Normal('β', mu=0, sd=10, shape=order)
        eps = pm.HalfNormal('ε', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=miu, sd=eps, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()

    plt.show()

# for sd = 100

    with pm.Model() as model_p1:
        alpha1 = pm.Normal('α', mu=0, sd=1)
        beta1 = pm.Normal('β', mu=0, sd=100, shape=order)
        eps1 = pm.HalfNormal('ε', 5)
        miu1 = alpha1 + pm.math.dot(beta1, x_1s)
        y_pred1 = pm.Normal('y_pred', mu=miu1, sd=eps1, observed=y_1s)
        idata_p1 = pm.sample(2000, return_inferencedata=True)

    x_new1 = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_p_post1 = idata_p.posterior1['α'].mean(("chain", "draw")).values
    β_p_post1 = idata_p.posterior1['β'].mean(("chain", "draw")).values
    idx1 = np.argsort(x_1s[0])
    y_p_post1 = α_p_post1 + np.dot(β_p_post1, x_1s)
    plt.plot(x_1s[0][idx], y_p_post1[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()

    plt.show()

# for sd = sd=np.array([10, 0.1, 0.1, 0.1, 0.1])

    with pm.Model() as model_p2:
        alpha2 = pm.Normal('α', mu=0, sd=1)
        beta2 = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        eps2 = pm.HalfNormal('ε', 5)
        miu2 = alpha2 + pm.math.dot(beta2, x_1s)
        y_pred2 = pm.Normal('y_pred', mu=miu2, sd=eps2, observed=y_1s)
        idata_p2 = pm.sample(2000, return_inferencedata=True)

    x_new2 = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_p_post2 = idata_p.posterior2['α'].mean(("chain", "draw")).values
    β_p_post2 = idata_p.posterior2['β'].mean(("chain", "draw")).values
    idx2 = np.argsort(x_1s[0])
    y_p_post2 = α_p_post1 + np.dot(β_p_post2, x_1s)
    plt.plot(x_1s[0][idx], y_p_post2[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()

    plt.show()

#2)
    dummy_data = []
    for i in range(1,500):
        dummy_data.append([random.uniform(0, 1), random.uniform(0, 1)])
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    with pm.Model() as model_p:
        alpha = pm.Normal('α', mu=0, sd=1)
        beta = pm.Normal('β', mu=0, sd=10, shape=order)
        eps = pm.HalfNormal('ε', 5)
        miu = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=miu, sd=eps, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)
    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()

#3)

cmp_df = az.compare({'model_l':idata_l, 'model_p':idata_p},

method='BB-pseudo-BMA', ic="waic", scale="deviance")
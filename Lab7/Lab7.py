import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

np.random.seed(1)

df = pd.read_csv('Prices.csv')

x = np.array(df["Speed"],np.log(df["HardDrive"]))
x_mean = x.mean(axis=0, keepdims=True)
x_centered = x - x_mean
y = df["Price"]

with pm.Model() as model:
    α_tmp = pm.Normal('α', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    ε = pm.HalfCauchy('ε', 5)
    μ = α_tmp + df["Speed"]beta1 + np.log(df["HardDrive"]*beta2)

    α = pm.Deterministic('α', α_tmp - pm.math.dot(x_mean, beta1))
    y_pre = pm.Normal('y_pre', mu=μ, sd=ε, observed=df["Price"])
    idata_model = pm.sample(2000, tune=2000, return_inferencedata=True)
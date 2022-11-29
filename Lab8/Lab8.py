import az as az
import pymc3 as pm
import pandas as pd
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

#We use the following model
df = pd.read_csv('Admission.csv')

with pm.Model() as model:
    β0 = pm.Normal('β0', mu=0, sigma=100)
    β1 = pm.Normal('β1', mu=0, sigma=100)
    β2 = pm.Normal('β2', mu=0, sigma=100)
    pi = pm.Deterministic('pi', pm.math.invlogit(β0 + β1*df['GRE'] + β2*df['GPA']))
    y = pm.Bernoulli('y', pi, observed=df['Admission'])
    idata_model = pm.sample(400, tune=2000, target_accept=0.94, return_inferencedata=True)

az.plot_hdi(df['GRE'], df['GPA'], idata_model.posterior['pi'].mean(dim=('chain', 'draw')), hdi_prob=0.94)


idx = np.argsort(df['Admission'].values[:, 0])
bd = idata_model.posterior['bd'].mean(("chain", "draw"))[idx]
plt.scatter(df['Admission'].values[:, 0], df['Admission'].values[:, 1], c=[f'C{x}' for x in df['Admission'].values[:, 0]])
plt.plot(df['Admission'].values[:, 0][idx], bd, color='k');
az.plot_hdi(df['Admission'].values[:, 0], idata_model.posterior['bd'], color='k')
plt.xlabel('GRE')
plt.ylabel('GPA'

posterior_0 = predictive.posterior.stack(samples=("chain", "draw"))
theta = posterior_0['pi'].mean("samples")
idx = np.argsort(data['GPA'].values)
plt.plot(df['GPA'].values[idx], theta[idx], color='C2', lw=3)
plt.vlines(posterior_0['bd2'].mean(), 0, 1, color='k')
bd_hpd = az.hdi(posterior_0['bd2'].values)
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)
plt.scatter(df['GPA'].values, np.random.normal(df['Admission'].values, 0.02),
            marker='.', color=[f'C{x}' for x in df['Admission'].values])
az.plot_hdi(df['GPA'].values, posterior_0['pi'].T, color='C2', smooth=False)
plt.xlabel(df['GPA'].values)
plt.ylabel('pi', rotation=0)
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + df['GPA'].values.mean(), 1))

posterior_0 = predictive.posterior.stack(samples=("chain", "draw"))
theta = posterior_0['pi'].mean("samples")
idx = np.argsort(data['GRE'].values)
plt.plot(df['GRE'].values[idx], theta[idx], color='C2', lw=3)
plt.vlines(posterior_0['bd2'].mean(), 0, 1, color='k')
bd_hpd = az.hdi(posterior_0['bd2'].values)
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)
plt.scatter(df['GRE'].values, np.random.normal(df['Admission'].values, 0.02),
            marker='.', color=[f'C{x}' for x in df['Admission'].values])
az.plot_hdi(df['GRE'].values, posterior_0['pi'].T, color='C2', smooth=False)
plt.xlabel(df['GRE'].values)
plt.ylabel('pi', rotation=0)
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + df['GPA'].values.mean(), 1))














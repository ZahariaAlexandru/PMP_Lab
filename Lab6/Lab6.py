# importat un fisier care are 400 de intrari

import csv
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm

with open(r'data.csv', 'r') as file:
    csvreader = csv.reader(file)
    ppvt = []
    educ_cat = []
    mom_age = []
    ok = 0
    for row in csvreader:
        if ok == 0:
            ok = 1
        elif ok == 1:
            ppvt.append(row[1])
            educ_cat.append(row[2])
            mom_age.append(row[3])
        for i in range(0, len(ppvt)):
            ppvt[i] = int(ppvt[i])
            mom_age[i] = int(mom_age[i])
ppvt = np.array(ppvt)
educ_cat = np.array(educ_cat)
mom_age = np.array(mom_age)
mom_age.sort()
#1
plt.scatter(ppvt, mom_age)
plt.xlabel('ppvt')
plt.ylabel('momage')
plt.show()

#2
# bayesian regression line
with pm.Model() as model_g:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=1)
    ε = pm.HalfCauchy('ε', 5)
    μ = pm.Deterministic('μ', α + β * ppvt)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=mom_age)
    idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)
    az.plot_trace(idata_g, var_names=['α', 'β', 'ε'])
    plt.scatter(x, y, color="m",
                marker="o", s=30)
    X1 = np.linspace(0, 100, 100)
    X2 = np.linspace(0, 100, 100)
    y_pred = X1+ X2 * mom_age
    plt.plot(ppvt, y_pred, color="g")
    plt.xlabel('ppvt')
    plt.ylabel('momage')
    plt.show()



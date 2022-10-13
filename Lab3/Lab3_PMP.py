import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    cutremur = pm.Bernoulli('C',0.0005)
    incediu  = pm.Deterministic('I',pm.math.switch(cutremur,0.03,0.01))
    alarma   = pm.Deterministic('A',pm.math.switch((cutremur and incediu),0.98,pm.math.switch(incediu,0.95,pm.math.switch(cutremur,0.02,0.0001))))
    trace = pm.sample(20000)

dictionary = {
              'cutremur' : trace['C'].tolist(),
              'incediu' : trace['C'].tolist(),
              'alarma' : trace['C'].tolist()
             }
df = pd.DataFrame(dictionary)

p_cutremur = df[((df['cutremur'] == 1) & (df['incediu'] == 1))].shape[0] / df[df['incediu'] == 1].shape[0]

print(p_cutremur)

p_incediu = df[((df['incediu'] == 1) & (df['alarma'] == 0))].shape[0] / df[df['incediu'] == 0].shape[0]

print(p_incediu)

az.plot_posterior(trace)
plt.show()
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

s1 = stats.gamma.rvs(4, 0, 1 / 3, size=10000)
s2 = stats.gamma.rvs(4, 0, 1 / 2, size=10000)
s3 = stats.gamma.rvs(5, 0, 1 / 2, size=10000)
s4 = stats.gamma.rvs(5, 0, 1 / 3, size=10000)
list = []
for i in range(10000):
    z = stats.uniform.rvs(0, 1)
    if z < s1[i]:
        list.append(s1[i])
    elif s1[i] <= z < s2[i]:
        list.append(s2[i])
    elif s2[i] <= z < s3[i]:
        list.append(s3[i])
    elif s3[i] <= z < s4[i]:
        list.append(s4[i])

az.plot_posterior({'list':list})
print("Probabilitatea este: " + str(np.mean(s4)))
plt.show()
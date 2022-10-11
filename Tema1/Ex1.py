import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

z = []
s = 0
x = []
for i in range(0, 10000):
    x.append(stats.binom.rvs(1, 0.4, size=1))
    if x[i] == 0:
        z.append(stats.expon.rvs(0.25))
    else:
        z.append(stats.expon.rvs(0.16))
s = np.sum(z)
media = s / 10000
d = np.std(z)
print((s))
print(media)
az.plot_posterior({'z':z})

plt.show()
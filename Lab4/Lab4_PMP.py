import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from random import random, randint

with pm.Model():

    numar_clienti = pm.Poisson('C', mu=1/3) # distributie poisson cu 1/3 oameni pe minut

    for i in numar_clienti:
        timp_plasare_plata = pm.Normal('x', mu=1, sigma=0.5)# distributie normala cu media 1 minut si deviata standard 0.5

        alpha = randint(0,30) # numar necunoscut pentru subpunctul 1
        timp_comanda = pm.Exponential(alpha)# distributia exponent
print(numar_clienti)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import pymc3 as pm

with pm.Model():
    nr_clienti = pm.poisson(20, 1000)

    for i in range(len(nr_clienti)):
        timp_plasare_plata = pm.Normal('x', mu=1, sigma=0.5)

        timp_comanda = pm.Exponential('y', mu=2)

        timp_consum = pm.Normal('z', mu=10, sigma=2)
        trace = pm.sample(nr_clienti[i], chains=1, model=md)
        dictionary = {
            'comanda': trace['N'].tolist(),
            'gatit': trace['G'].tolist(),
            'mancat': trace['M'].tolist()
        }

        for elem in dictionary['comanda']:
            timp_plasare_plata += elem
        for elem in dictionary['gatit']:
            timp_comanda += elem
        for elem in dictionary['mancat']:
            timp_consum += elem
       nr_statii_gatit = pm.random.randint(1, 10)
       nr_statii_casa = pm.random.randint(1, 10)
       nr_mese = pm.random.randint(1, 10)
       timp_comanda /= nr_statii_gatit
       timp_plasare_plata /= nr_statii_casa
       timp_consum /= nr_mese
       if timp_comanda + timp_plasare_plata <= 60:
           nr_statii_gatit += 1
       if timp_consum <= 60:
           nr_mese += 1
    print(nr_statii_casa/100)
    print(nr_mese/100)







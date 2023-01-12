import matplotlib.pyplot as plt
import arviz as az
import pandas as pd


centered = az.load_arviz_data("centered_eight")
non_centered = az.load_arviz_data("non_centered_eight")


#Exercitiul 1
print(centered['posterior'])
info= az.plot_trace(centered, divergences='top', compact=False)
print(non_centered['posterior'])
info = az.plot_trace(non_centered, divergences='top', compact=False)


#Exercitiul 2
Rhat1 = az.rhat(centered, var_names=["mu", "theta"])
Rhat2 = az.rhat(non_centered, var_names=["mu", "theta"])

print(Rhat1)
print(Rhat2)


#Exercitiul 3
centered.sample_stats.diverging.sum()
non_centered.sample_stats.diverging.sum()

_, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5), constrained_layout=True)

for idx, tr in enumerate([centered, non_centered]):
    az.plot_pair(tr, var_names=['mu', 'tau'], kind='scatter',
                 divergences=True, divergences_kwargs={'color':'C1'},
                 ax=ax[idx])

    ax[idx].set_title(['centered', 'non-centered'][idx])

plt.show()
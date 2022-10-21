# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sys
import pandas as pd
import numpy as np
from leakconfound.simulation import (
    create_correlated_array
)
from scipy.stats import pearsonr, spearmanr
import os
from pathlib import Path

np.random.seed(189747)
data_dir = sys.argv[1]
# %%
base_dataset_dir = f'{data_dir}/uci_datasets/'
save_base_dir = f'{data_dir}/uci_sim_conf_datasets/'

if not os.path.exists(save_base_dir):
    os.makedirs(save_base_dir)

all_datasets = [dataset.replace('.csv', '')
                for dataset in os.listdir(base_dataset_dir)
                ]

for dataset_name in all_datasets:
    data = pd.read_csv(base_dataset_dir + dataset_name + '.csv')
    data = data.filter(regex='.*(?<!_confound)$')
    target = data.filter(regex='.*_target').iloc[:, 0]

    for corr in np.arange(0.2, 1, .2):
        corr_method = (spearmanr if target.name.endswith('__binary_target')
                       else pearsonr
                       )
        confound = create_correlated_array(
            target, corr, corr_method=corr_method, abs_tol=.01)
        df_sim = data.copy()
        save_dir = save_base_dir + dataset_name + \
            f'_confoundCorr_{int(10*corr)}' + '.csv'
        df_sim['simulated__continuous_confound'] = confound
        df_sim.to_csv(save_dir, index=False)


# %%
y = np.random.normal(size=500)
X = create_correlated_array(y, .5)
y = pd.Series(y).apply(lambda x: x < np.median(y))
df = pd.DataFrame(dict(X__continuous=X, y__binary_target=y))
noise = np.random.normal(size=500)
df["Xn__continuous"] = df["X__continuous"] + noise
df["noise__continuous_confound"] = noise

path = f"{data_dir}/sim_supression/"
Path(path).mkdir(parents=True, exist_ok=True)
df.to_csv(path+'sim.csv', index=False)

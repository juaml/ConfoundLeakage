# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Download UCI Data
# Here, I will read in data from the UCI repository and do some minor preprocessing.

# %%
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from leakconfound.prepare_datasets import (balance_shuffle_data, encode_cat)
from sklearn.preprocessing import StandardScaler
np.random.seed(4224)
data_dir = sys.argv[1]

Path(f'{data_dir}/realistic/audio_data').mkdir(exist_ok=True, parents=True)

# %%


def convert_stupid_german_values(val):
    return (val.replace(",", ".").replace("E", "e")
            if isinstance(val, str)
            else val
            )


df_bdi_labels = pd.read_csv(f'{data_dir}/raw/audio_data/label_adhd_all_bdi.csv', sep=';')
df_features = pd.read_csv(f'{data_dir}/raw/audio_data/features_6K.csv', sep=";")

df_features = df_features.applymap(convert_stupid_german_values).astype(float)

confounds = ['Gender', 'Education', 'Age', 'BDI']


df = df_features.copy()
df.columns = [str(i)+'__continuous' for i, _ in enumerate(df.columns)]
X = df.columns.to_list()
df['ATT_Task__binary_target'] = df_bdi_labels.ATT_TaskValue

df[['Gender__categorical_confound', 'Education__categorical_confound',
    'Age__continuous_confound']] = df_bdi_labels[confounds[:-1]]
df['BDI__continuous_confound'] = df_bdi_labels['BDI']

# only session one to prevent dependence between train test split in KFold
df['Session'] = df_bdi_labels['Session']
df = df.query('Session == "a"').copy()
df = df.drop(columns=['Session'])

df = df.replace({999: np.nan}).dropna()
df = balance_shuffle_data(df)
df, df_cat_encoders = encode_cat(df)


mask_drop = (df[X].apply(np.var) <
             2.2e-16) | df[X].apply(lambda x: len(np.unique(x)) < 100)

X_not_dropped = list(np.array(X)[~mask_drop])
print(f'{len(X_not_dropped)} of {len(X)}')
X = X_not_dropped

confounds = ['Gender__categorical_confound', 'Education__categorical_confound',
             'Age__continuous_confound', 'BDI__continuous_confound']
df = df.loc[:, [*X, *confounds, "ATT_Task__binary_target"]].copy()

df.to_csv(f'{data_dir}/realistic/audio_data/audio_data.csv', index=False)
Path(f'{data_dir}/realistic_non_TaCo/audio_data/').mkdir(parents=True, exist_ok=True)
df.to_csv(f'{data_dir}/realistic_non_TaCo/audio_data/audio_data.csv', index=False)
for conf in confounds:
    _df = df.copy().drop(columns=[c for c in confounds if c != conf])
    _df.to_csv(
        f'{data_dir}/realistic_non_TaCo/audio_data/audio_data_{conf.split("__")[0]}.csv',
        index=False)


noise = np.random.normal(scale=.25, size=(len(df), 1))
for i in range(9):
    noise += np.random.normal(scale=.25, size=(len(df), 1))
noise = np.random.normal(scale=1, size=(len(df), 1))
# noise_per_feat = np.random.uniform(
#     low=-1, high=1, size=(len(df), len(df[X].columns)))
df[X] = StandardScaler().fit_transform(df[X])
df[X] += noise  # + noise_per_feat

df = df.drop(columns=confounds)

df["noise__categorical_confound"] = noise

df.to_csv(
    f'{data_dir}/realistic/audio_data/audio_suppression_data.csv', index=False)

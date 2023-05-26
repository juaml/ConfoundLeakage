---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++
# Analysis 3d: Real World Datasets Audio Data Grouped CR

Imports, Functs and Paths

```{code-cell}
:tags: [hide-input, hide-output]
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from leakconfound.transformers import Shuffle
import os

import pandas as pd
import matplotlib.pyplot as plt
from julearn.transformers.confounds import DataFrameConfoundRemover

# plot styles
from sciplotlib import style

import matplotlib as mpl

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

base_save_paper = "./paper_val/"
results_base = "../../results/permutations"
base_dir = "../../"
colors = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2FF",
    "#DC0000",
    "#7E6148",
    "#B09C85",
]
red = colors[0]
blue = colors[1]
green = colors[2]
purple = colors[5]

np.random.seed(891236740)


def mm_to_inch(val_in_inch):
    mm = 0.1 / 2.54
    return val_in_inch * mm


base_save_paper = "./paper_val/"
results_base = "../../results/"
base_dir = "../../"
audio_data_non_TaCo_folder = f"{results_base}basic_non_TaCo/realistic_non_TaCo/models/"

audio_data_shuffled_non_TaCo_folder = (
    f"{results_base}" "shuffled_features_non_TaCo/realistic_non_TaCo/models/"
)


mpl.style.use(style.get_style("nature-reviews"))
mpl.rc("xtick", labelsize=11)
mpl.rc("ytick", labelsize=11)
mpl.rc("axes", labelsize=12, titlesize=12)
mpl.rc("figure", dpi=300)
mpl.rc("figure.subplot", wspace=mm_to_inch(8), hspace=0.7)
mpl.rc("lines", linewidth=1, markersize=2)

fig = plt.figure(
    figsize=[mm_to_inch(183), mm_to_inch(140)],
)


```

+++
Some info on the data:


```{code-cell}
:tags: [hide-input]
os.listdir(audio_data_non_TaCo_folder)

df_original = pd.read_csv("../../data/realistic_non_TaCo/audio_data/audio_data_BDI.csv")

X = df_original.filter(regex=".*__continuous$").columns.tolist()
y = "ATT_Task__binary_target"
confounds = ["BDI__continuous_confound"]
cv = RepeatedStratifiedKFold(random_state=2738).split(df_original[X], df_original[y])

scores_raw = []
scores_rm = []
scores_shuffled_raw = []
scores_shuffled_rm = []
for i_train, i_test in cv:
    df_i_train = df_original.iloc[i_train, :].copy()
    df_i_train_0 = df_i_train.query(f"{y} == 0").copy()
    df_i_test = df_original.iloc[i_test, :].copy()

    X_i_train = df_i_train.loc[:, X]
    Xc_i_train = df_i_train.loc[:, X + confounds]
    y_i_train = df_i_train.loc[:, y]

    Xc_i_train_0 = df_i_train_0.loc[:, X + confounds]
    y_i_train_0 = df_i_train_0.loc[:, y]

    X_i_test = df_i_test.loc[:, X]
    Xc_i_test = df_i_test.loc[:, X + confounds]
    y_i_test = df_i_test.loc[:, y]

    # Raw:
    sc = StandardScaler().fit(X_i_train)
    X_sc_train = sc.transform(X_i_train)
    X_sc_test = sc.transform(X_i_test)
    rf_X = RandomForestClassifier(n_estimators=300, random_state=78).fit(
        X_sc_train, y_i_train
    )
    scores_raw.append(rf_X.score(X_sc_test, y_i_test))

    # Remove trained on class 0 only:
    # But transform both
    sc_0 = StandardScaler().fit(Xc_i_train_0)
    Xc_sc_train_0 = pd.DataFrame(sc_0.transform(Xc_i_train), columns=Xc_i_train.columns)

    Xc_sc_test_0 = pd.DataFrame(sc_0.transform(Xc_i_test), columns=Xc_i_train.columns)

    remover = DataFrameConfoundRemover(confounds_match=confounds[0]).fit(Xc_sc_train_0)
    Xc_rem_train = remover.transform(Xc_sc_train_0)
    Xc_rem_test = remover.transform(Xc_sc_test_0)

    assert len(Xc_rem_train.columns) == len(X)
    rf_X = RandomForestClassifier(n_estimators=300, random_state=78).fit(
        Xc_rem_train, y_i_train
    )
    scores_rm.append(rf_X.score(Xc_rem_test, y_i_test))

    # Shuffled
    # Raw:
    shuffler = Shuffle().fit(X_i_train)
    X_i_train_shuffled = shuffler.transform(X_i_train)
    X_i_test_shuffled = shuffler.transform(X_i_test)
    Xc_i_train_shuffled = X_i_train_shuffled.copy()
    Xc_i_train_shuffled[confounds] = Xc_i_train[confounds]
    Xc_i_test_shuffled = X_i_test_shuffled.copy()
    Xc_i_test_shuffled[confounds] = Xc_i_test[confounds]
    Xc_i_train_shuffled_0 = Xc_i_train_shuffled.loc[y_i_train == 0]

    sc = StandardScaler().fit(X_i_train_shuffled)
    X_sc_train_shuffled = sc.transform(X_i_train_shuffled)
    X_sc_test = sc.transform(X_i_test)
    rf_X = RandomForestClassifier(n_estimators=300, random_state=78).fit(
        X_sc_train_shuffled, y_i_train
    )
    scores_shuffled_raw.append(rf_X.score(X_sc_test, y_i_test))

    # Remove train_shuffleded on class 0 only:
    # But transform both
    sc_0 = StandardScaler().fit(Xc_i_train_shuffled_0)
    Xc_sc_train_shuffled_0 = pd.DataFrame(
        sc_0.transform(Xc_i_train_shuffled), columns=Xc_i_train_shuffled.columns
    )

    Xc_sc_test_0 = pd.DataFrame(
        sc_0.transform(Xc_i_test), columns=Xc_i_train_shuffled.columns
    )

    remover = DataFrameConfoundRemover(confounds_match=confounds[0]).fit(
        Xc_sc_train_shuffled_0
    )
    Xc_rem_train_shuffled = remover.transform(Xc_sc_train_shuffled_0)
    Xc_rem_test = remover.transform(Xc_sc_test_0)

    assert len(Xc_rem_train_shuffled.columns) == len(X)
    rf_X = RandomForestClassifier(n_estimators=300, random_state=78).fit(
        Xc_rem_train_shuffled, y_i_train
    )
    scores_shuffled_rm.append(rf_X.score(Xc_rem_test, y_i_test))

df_scores = pd.DataFrame(
    dict(
        raw=scores_raw,
        rem=scores_rm,
        shuffled_raw=scores_shuffled_raw,
        shuffled_rem=scores_shuffled_rm,
        repeats=np.repeat(np.arange(10), 5),
        folds=np.tile(np.arange(5), 10),
    )
)
df_grouped = df_scores.groupby("repeats").mean()
print(df_grouped.describe())
```


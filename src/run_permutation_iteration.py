
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysis 3: Real World Datasets Audio Data

# Imports, Functs and Paths
# %% tags=["hide-input", "hide-output"]
import pandas as pd
import sys
from ast import literal_eval
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from julearn.transformers.confounds import DataFrameConfoundRemover
from leakconfound.transformers import Shuffle

repeat = int(sys.argv[1])
shuffled = literal_eval(sys.argv[2])


# %% tags=["hide-input"]
data = pd.read_csv('./data/realistic/audio_data/audio_data.csv')
result_dir = "./results/permutations/"
Path(result_dir).mkdir(parents=True, exist_ok=True)
result_dir += f'{repeat}_{shuffled}.csv'
confounds = data.filter(regex=".*confound", axis=1).columns.tolist()
confounds_renamer = {old_name: old_name+"__:type:__confound"
                     for old_name in confounds
                     }
data = data.rename(columns=confounds_renamer)
confounds = list(confounds_renamer.values())
y = data.filter(regex=".*target", axis=1).columns.tolist()[0]
X = data.iloc[:, :-5].columns.tolist()

cv = KFold(random_state=92345780, shuffle=True)
df_scores = pd.DataFrame()
df_comparisons = pd.DataFrame()
for fold, (train_idx, test_idx) in enumerate(cv.split(data.index)):
    print(f"Starting iteration {repeat=}, {fold=}")
    rng = repeat + fold
    df_train = data.iloc[train_idx, :]
    df_test = data.iloc[test_idx, :]
    # Standardize
    scaler = StandardScaler().fit(df_train[X])
    df_train.loc[:, X] = scaler.transform(df_train[X])
    df_test.loc[:, X] = scaler.transform(df_test[X])

    if shuffled:
        # Shuffled
        # original
        df_shuffled_train = df_train.copy()
        df_shuffled_test = df_test.copy()

        shuffler = Shuffle(random_state=rng).fit(df_train[X])
        df_shuffled_train.loc[:, X] = shuffler.transform(df_train[X]).values
        df_shuffled_test.loc[:, X] = shuffler.transform(df_test[X]).values

        rf_shuffled = RandomForestClassifier().fit(df_shuffled_train[X], df_shuffled_train[y])
        shuffled_score = rf_shuffled.score(df_shuffled_test[X], df_shuffled_test[y])

        # CR
        cr_shuffled = DataFrameConfoundRemover().fit(
            df_shuffled_train[X+confounds], df_shuffled_train[y]
        )
        df_shuffled_train_cr = cr_shuffled.transform(
            df_shuffled_train[X+confounds]
        )
        df_shuffled_train_cr.loc[:, y] = df_shuffled_train[y]
        df_shuffled_test_cr = cr_shuffled.transform(
            df_shuffled_test[X+confounds]
        )

        df_shuffled_test_cr.loc[:, y] = df_shuffled_test[y]

        rf_shuffled_cr = RandomForestClassifier().fit(
            df_shuffled_train_cr[X], df_shuffled_train_cr[y])
        shuffled_score_cr = rf_shuffled_cr.score(df_shuffled_test_cr[X], df_shuffled_test_cr[y])
        to_concat = [
            pd.DataFrame(
                dict(
                    score=shuffled_score, shuffled=True, confound_removed=False,
                    fold=fold, repeat=repeat
                ), index=[0]),
            pd.DataFrame(
                dict(
                    score=shuffled_score_cr, shuffled=True, confound_removed=True,
                    fold=fold, repeat=repeat
                ), index=[0]),
        ]

    else:
        # Actual Prediction:
        # original
        rf_orig = RandomForestClassifier().fit(df_train[X], df_train[y])
        orig_score = rf_orig.score(df_test[X], df_test[y])
        # CR
        cr_orig = DataFrameConfoundRemover().fit(df_train[X+confounds])
        df_train_cr = cr_orig.transform(df_train[X+confounds])
        df_train_cr.loc[:, y] = df_train[y]
        df_test_cr = cr_orig.transform(df_test[X+confounds])
        df_test_cr.loc[:, y] = df_test[y]
        rf_cr_orig = RandomForestClassifier().fit(df_train_cr[X], df_train_cr[y])
        orig_score_cr = rf_cr_orig.score(df_test_cr[X], df_test_cr[y])
        to_concat = [
            pd.DataFrame(dict(
                score=orig_score, shuffled=False, confound_removed=False,
                fold=fold, repeat=repeat
            ), index=[0]),
            pd.DataFrame(
                dict(
                    score=orig_score_cr, shuffled=False, confound_removed=True,
                    fold=fold, repeat=repeat
                ), index=[0]),
        ]

    df_iter = pd.concat(to_concat)
    df_scores = pd.concat([df_scores, df_iter])
df_scores.to_csv(result_dir, index=False)

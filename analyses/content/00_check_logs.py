# %%
import os
import pandas as pd
log_base = "./logs"

ignore = [
    "Hyperparameter search CV was specified, but no hyperparameters to tune",
    "/home/shamdan/LeakConfound/.venv/lib/python3.8/site-packages/"
    "julearn/utils/logging.py:169: RuntimeWarning: Hyperparameter search CV was specified,"
    "but no hyperparameters to tune",
    "warnings.warn(msg, category=category)"
]

# %%


def read_log(path):

    with open(path, "rb") as f:
        return list({line for line in f.readlines()})


def create_log_df(base_dir, experiment, datasets, data, file_name):
    path = f"{base_dir}/{experiment}/{datasets}/{data}/{file_name}"

    return pd.DataFrame(dict(
        experiment=experiment,
        datasets=datasets,
        data=data,
        file_name=file_name,
        log=read_log(path)
    ))


# %%
# All logs except for separation analyses
df_log = pd.concat([create_log_df(log_base, experiment, datasets, data, file_name)
                    for experiment in os.listdir(log_base)
                    for datasets in os.listdir(f"{log_base}/{experiment}")
                    if (
                        (not datasets.endswith(".log"))
                        and (not datasets.endswith(".err"))
                        and (not datasets.endswith(".out"))
)
    for data in os.listdir(f"{log_base}/{experiment}/{datasets}")
    for file_name in os.listdir(f"{log_base}/{experiment}/{datasets}/{data}")
    if file_name.endswith(".err")
])

# %%
print(df_log.log.unique())

# %%
print(
    df_log[df_log.log.astype(str).str.contains("FitFailedWarning")]
)

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from pathlib import Path


def encode_cat(df):
    """encodes only categorical variables as numbers"""
    _df = df.copy()

    cat_columns = (_df
                   .loc[:, ['__categorical' in col
                            for col in _df.columns]
                        ]
                   ).columns.to_list()

    confounds_column = (_df
                        .loc[:, ['__confound' in col
                                 for col in _df.columns]
                             ]
                        ).columns.to_list()

    cat_confounds = [col
                     for col in confounds_column
                     if type(_df.loc[:, col].iloc[0]) == str
                     ]

    cat_target = [col
                  for col in _df.columns
                  if (col.endswith('target')) and (type(_df.loc[:, col].iloc[0]) == str)]

    cat_columns += cat_confounds
    cat_columns += cat_target

    encoders = {col_name: LabelEncoder().fit(_df[col_name])
                for col_name in cat_columns}

    for col_name, encoder in encoders.items():
        _df[col_name] = encoder.transform(_df[col_name])

    return _df, encoders


def dump_data(df, encoders, dataset_name, base_dir):
    path = f'{base_dir}uci_datasets/'
    Path(path).mkdir(exist_ok=True, parents=True)
    df.to_csv(f'{path}{dataset_name}.csv', index=False)
    (Path(base_dir + 'encoders/')
     .mkdir(exist_ok=True, parents=True)
     )
    dump(encoders, open(base_dir + 'encoders/' + dataset_name, "wb"))


def balance_shuffle_data(df):
    target = [col for col in df.columns
              if col.endswith('target')]
    assert len(target) == 1
    target = target[0]
    if not target.endswith('regression_target'):
        min_size = df[target].value_counts().min()

        df_out = pd.concat([
            df[df[target] == val].sample(n=min_size, replace=False)
            for val in df[target].unique()
        ])
        print(f'The dataset was reduced from {len(df)} to {len(df_out)}')
    else:
        print('Regression problem so no resampling')
        print(f'Dataset length = {len(df)}')
        df_out = df.copy()
    print(df_out[target].value_counts())
    df_out = df_out.sample(len(df_out), replace=False)
    return df_out

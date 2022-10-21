from .. transformers import Shuffle
import pandas as pd
# Note:
# 1) confound removal will only be applied when deconfound is true
# 2) only continuous confounds are zscored even if zscoring is specified categorical stay same
experiments = {
    'basic_TaCo': dict(TaCo=True, convert_ordinal=True,
                       preprocess_X=['zscore', 'remove_confound'],
                       preprocess_confounds=['zscore'],
                       zscore__apply_to='continuous',
                       remove_confound__apply_to='all'
                       ),

    'basic_non_TaCo': dict(TaCo=False, convert_ordinal=True,
                           preprocess_X=['zscore', 'remove_confound'],
                           preprocess_confounds=['zscore'],
                           zscore__apply_to='continuous',
                           remove_confound__apply_to='all'
                           ),
    'basic_TaCo_threshold': dict(TaCo=True, convert_ordinal=True,
                                 preprocess_X=['zscore', 'remove_confound'],
                                 preprocess_confounds=['zscore'],
                                 zscore__apply_to='continuous',
                                 remove_confound__threshold=1e-5,
                                 remove_confound__apply_to='all'

                                 ),
    'shuffled_features_TaCo': dict(TaCo=True, convert_ordinal=True,
                                   preprocess_X=['zscore', 'remove_confound'],
                                   preprocess_confounds=['zscore'],
                                   zscore__apply_to='continuous',
                                   remove_confound__apply_to='all',
                                   shuffle='all_features',
                                   ),
    'shuffled_features_non_TaCo': dict(TaCo=False, convert_ordinal=True,
                                       preprocess_X=[
                                           'zscore', 'remove_confound'],
                                       preprocess_confounds=['zscore'],
                                       zscore__apply_to='continuous',
                                       remove_confound__apply_to='all',
                                       shuffle='all_features',
                                       ),
}


def get_input_data(df, TaCo):
    """get Features X, Target y and Confounds conf from the data
    considering whether it is TaCo

    Parameters
    ----------
    df : pd.DataFrame
        Dataset where features end with __categorical or __continuous
        Targets end with __*target and confounds with __confound
    TaCo : bool
        Should the Confound be equal to the Target

    """

    # Getting columns names of X, target and confounds from data
    X = [col
         for col in df.columns.to_list()
         if col.endswith('continuous')
         ]
    categorical = [col
                   for col in df.columns.to_list()
                   if (col.endswith('categorical') or col.endswith('nominal'))
                   ]
    categorical = None if categorical == [] else categorical
    y = [col
         for col in df.columns.to_list()
         if col.endswith('target')][0]

    if TaCo:
        # when target as a confound
        problem_type = y.split('__')[-1].split('_')[0]
        conf_name = y.split('__')[0]  # everything before the type
        conf_name += ('__continuous_confound' if problem_type == 'regression'
                      else '__categorical_confound')
        df[conf_name] = df.copy()[y]
        conf = [conf_name]
    else:

        conf = [col
                for col in df.columns.to_list()
                if col.endswith('confound')]

    return X, categorical, y, conf, df


def _get_experiment(df,
                    TaCo,
                    preprocess_X,
                    preprocess_confounds,
                    param_grid,
                    shuffle=None,
                    convert_ordinal=False,
                    shuffle_random_state=None,
                    **param_vals):

    X, categorical, y, conf, df = get_input_data(df, TaCo)

    if convert_ordinal and categorical is not None:
        # only non binary categorical variables will be one hot encoded (ohe)
        categorical_non_bi = [cat
                              for cat in categorical
                              if df[cat].unique().shape[0] > 2
                              ]
        df_ohe = pd.get_dummies(
            df[categorical], columns=categorical_non_bi)
        df.drop(columns=categorical, inplace=True)
        categorical = [correct_dummy_name(col)
                       for col in df_ohe.columns.to_list()
                       ]
        print(categorical)
        df[categorical] = df_ohe

    X_cat = X + categorical if categorical is not None else X
    if shuffle == 'all_features':
        df[X_cat] = (Shuffle(random_state=shuffle_random_state)
                     .fit_transform(df[X_cat])
                     )
    elif shuffle == 'confound':
        df[conf] = (Shuffle(random_state=shuffle_random_state)
                    .fit_transform(df[conf])
                    )
    for param, val in param_vals.items():
        param_grid[param] = val

    return X, categorical, y, conf, df, preprocess_X, preprocess_confounds, param_grid


def get_experiment(experiment, df, deconfound, param_grid, random_state):
    """Get all data and preprocessing for this experiment

    Parameters
    ----------
    experiment : str
        Experiment represented by a str. Has to be a key of the experiments dict in experiments.py
    df : pd.DataFrame
        Data that will be used preprocessing and prediction
    param_grid : dict
        Parameters used to tune in julearn.run_cross_validation

    Returns
    -------
    tuple
        (X, categorical, y, conf, df, preprocess_X, preprocess_confounds, param_grid)
    """

    X, categorical, y, conf, df, preprocess_X, preprocess_confounds, param_grid = _get_experiment(
        df, param_grid=param_grid, **experiments[experiment], shuffle_random_state=random_state)

    if not deconfound:
        # We want to remove preprocessing steps and their params when we do not deconfound
        preprocess_X = [x for x in preprocess_X if x != 'remove_confound']
        param_grid = {param: val for param, val in param_grid.items()
                      if not param.startswith('remove_confound')}

    if len(conf) == 1:
        if conf[0].endswith('categorical_confound'):
            # only continuous confounds should be zscored
            preprocess_confounds = [
                x for x in preprocess_confounds if x != 'zscore']
            preprocess_confounds = None if preprocess_confounds == [] else preprocess_confounds

    else:
        if 'zscore' in preprocess_confounds:
            print('zscoring confounds')
    if "zscore" in preprocess_X:
        if X == []:
            preprocess_X = [step
                            for step in preprocess_X
                            if step != "zscore"
                            ]
            param_grid = {param: val for param, val in param_grid.items()
                          if not param.startswith("zscore")
                          }

    return X, categorical, y, conf, df, preprocess_X, preprocess_confounds, param_grid


def correct_dummy_name(dummy_column):
    """names will be name__categorical_1, but should be name_1__categorical
    so thate __categorical as type is still valid
    """
    if dummy_column[-1].isdigit():
        name, column_type = dummy_column.split("__")
        column_type, number = column_type.split('_')
        return f"{name}_{number}__{column_type}"

    return dummy_column

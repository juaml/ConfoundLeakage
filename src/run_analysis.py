import sys
import joblib
from pathlib import Path
from ast import literal_eval
from itertools import tee

import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold, KFold)

from julearn import run_cross_validation
from julearn.scoring.metrics import r2_corr

from leakconfound.experiments.helper_func import get_model_grid
from leakconfound.experiments.experiments import get_experiment
from leakconfound.scores.score import assign_X_hat_scores

r2_corr_scorer = make_scorer(r2_corr)


def main(data, results_folder,  model_results_folder,
         cv_save_folder,
         model_name, deconfound, experiment,
         random_seed):

    # Define problem type based on target
    print(f"log {data = }, {model_name = }, {deconfound = }, {experiment = }")
    target = [col
              for col in data.columns.to_list()
              if col.endswith('target')][0]
    problem_type = target.split('__')[-1].split('_')[0]
    problem_type = (problem_type if problem_type == 'regression' else
                    problem_type + '_classification'
                    )

    # Defining pipeline and parameter grid from model and deconfound input
    model, param_grid = get_model_grid(model_name, problem_type)

    inputs = get_experiment(
        experiment=experiment, df=data,
        deconfound=deconfound, param_grid=param_grid,
        random_state=random_seed + 24)  # using random seed + some difference

    X, categorical, target, confounds, df, preprocess_X, preprocess_confounds, param_grid = inputs
    prediction_confounds = confounds if deconfound else None

    # Define scoring depended on the defined problem
    continuous_scorers = dict(
        r2='r2', neg_mean_squared_error='neg_mean_squared_error',
        neg_mean_absolute_error='neg_mean_absolute_error',
        r2_corr=r2_corr_scorer
    )
    categorical_scorers = dict(
        accuracy='accuracy', roc_auc='roc_auc',
    )
    scoring = (
        continuous_scorers if problem_type == 'regression' else
        categorical_scorers
    )

    if problem_type == 'regression':
        cv, cv_X_hat, cv_save = tee(RepeatedKFold().split(
            df.index, df[target]), 3)
        param_grid['cv'] = KFold()  # inner cv for tuning
    else:
        cv, cv_X_hat, cv_save = tee(RepeatedStratifiedKFold().split(
            df.index, df[target]), 3)  # outer cv
        param_grid['cv'] = StratifiedKFold()  # inner cv tuning

    # Analysis
    scores, model = run_cross_validation(
        X=X, categorical=categorical, y=target,
        confounds=prediction_confounds, data=df,
        model=model,
        preprocess_X=preprocess_X, problem_type=problem_type,
        return_estimator='all', model_params=param_grid, cv=cv,
        scoring=scoring, return_indices=True,
        seed=random_seed
    )

    if deconfound:
        scores = assign_X_hat_scores(df, confounds, target, scores, cv_X_hat, scoring)

    confounds_str = (
        confounds if isinstance(confounds, str)
        else "_".join(
            [(conf
              .replace("__categorical_confound", "")
              .replace("__continuous_confound", ""))
             for conf in confounds
             ])
    )
    results_file = '___'.join(
        [data_name, str(model_name), str(deconfound), str(random_seed), confounds_str]) + '.csv'

    scores.to_csv(results_folder + results_file)

    if hasattr(model, 'best_estimator_'):
        model = model.best_estimator_

    joblib.dump(model, model_results_folder +
                results_file.replace('.csv', '.joblib'))

    joblib.dump(list(cv_save), cv_save_folder +
                results_file.replace('.csv', '_cv.joblib'))
    return model


if __name__ == '__main__':

    data_path = sys.argv[1]
    model_name = sys.argv[2]
    deconfound = literal_eval(sys.argv[3])
    experiment = sys.argv[4]
    random_seed = literal_eval(sys.argv[5])

    data = pd.read_csv(data_path)
    data_name = data_path.split('/')[-1].split('.')[0]
    data_folder = data_path.split('/')[2]
    # Defining and if needed creating folder/file structure
    results_folder = f'./results/{experiment}/{data_folder}/'
    Path(results_folder).mkdir(exist_ok=True, parents=True)

    model_results_folder = results_folder + 'models/'
    Path(model_results_folder).mkdir(exist_ok=True, parents=True)

    cv_save_folder = results_folder + 'cv/'
    Path(cv_save_folder).mkdir(exist_ok=True, parents=True)

    main(data, results_folder, model_results_folder,
         cv_save_folder,
         model_name, deconfound, experiment, random_seed)

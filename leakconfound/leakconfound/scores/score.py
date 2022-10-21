import pandas as pd
from sklearn.base import clone
from sklearn.metrics import check_scoring


def assign_X_hat_scores(df, confounds, target,
                        scores, cv, scoring):
    _df = df.copy()
    pipelines = scores["estimator"]

    multi_scores = True if isinstance(scoring, dict) else False

    Xhat_scores = dict() if multi_scores else []
    for (train_idx, test_idx), pipe in zip(cv, pipelines):

        conf_train = _df.iloc[train_idx, :][confounds]
        conf_test = _df.iloc[test_idx, :][confounds]
        if isinstance(confounds, str) or len(confounds) == 1:
            conf_train = conf_train.values.reshape(-1, 1)
            conf_test = conf_test.values.reshape(-1, 1)

        y_train = _df.iloc[train_idx, :][target]
        y_test = _df.iloc[test_idx, :][target]

        pipe = (pipe.best_estimator_
                if hasattr(pipe, 'best_estimator_')
                else pipe
                )

        # get transformer for confound remvoal
        trans_cr = (pipe
                    .named_steps
                    .remove_confound
                    .models_confound_
                    )

        X_pred_train = (
            trans_cr.apply(
                lambda model: model.predict(conf_train))
            .pipe(
                lambda df: pd.DataFrame({row: col for row, col in df.iteritems()}))

        )

        X_pred_test = (
            trans_cr.apply(
                lambda model: model.predict(conf_test))
            .pipe(
                lambda df: pd.DataFrame({row: col for row, col in df.iteritems()}))

        )

        model = (clone(pipe.dataframe_pipeline.steps[-1][1])
                 .fit(X_pred_train, y_train)
                 )
        # scorers = prepare_scoring(model, scoring)
        if multi_scores:
            for score_name, scorer in scoring.items():
                if isinstance(scorer, str):
                    scorer = check_scoring(model, scorer)

                if f'test_X_hat_{score_name}' in Xhat_scores.keys():
                    Xhat_scores[f'test_X_hat_{score_name}'].append(
                        scorer(model, X_pred_test, y_test))
                else:
                    Xhat_scores[f'test_X_hat_{score_name}'] = [scorer(model, X_pred_test, y_test)]
        else:
            Xhat_scores.append(check_scoring(scoring)(model, X_pred_test, y_test))

    if multi_scores:
        for key, vals in Xhat_scores.items():
            scores[key] = vals
    else:
        scores["test_X_hat_score"] = Xhat_scores

    return scores

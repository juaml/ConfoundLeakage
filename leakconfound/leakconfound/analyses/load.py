import os
import pandas as pd


def gather_data(path, analysis_arguments, additional_filter=None):
    if additional_filter is None:
        def additional_filter(val):
            return True
    df_all = pd.concat(
        [(pd.read_csv(path + result_file_name, engine='python')
          .assign(**get_meta_information(result_file_name, arguments=analysis_arguments))
          )
         for result_file_name in os.listdir(path)
         if (result_file_name not in ['plots', 'models', 'cv'])
         if additional_filter(result_file_name)
         ]
    ).reset_index(drop=True)
    return df_all


def data_to_long(df_all, analysis_arguments, score_names):
    df_long = (df_all.copy()
               # convert to long fromat
               .melt(
        id_vars=['repeat', 'fold', *analysis_arguments],
        value_vars=score_names,
        var_name='score_name',
        value_name='score')
    )

    if 'is_deconfounded' in analysis_arguments:
        df_long = (df_long
                   .assign(confound=lambda df: df['is_deconfounded'].apply(
                       lambda x: 'removed' if x == "True" else 'not removed'
                   ))
                   )
    return df_long


def prepare_performance_data(path, analysis_arguments, score_names):
    df_all = gather_data(path, analysis_arguments)
    df_all['data'] = df_all.data.map(data_renamer)
    df_all['model_name'] = df_all.model_name.map(models_renamer)
    df_long = data_to_long(df_all, analysis_arguments,  score_names)

    return df_long


def get_meta_information(file_name, arguments):
    arguments = arguments
    return {argument: val.replace('.csv', '')
            for argument, val in zip(arguments, file_name.split('___'))
            }


def multi_cols_to_normal_cols(df):
    _df = df.copy()
    multi_cols = _df.columns
    normal_cols = ['_'.join(col) for col in multi_cols]
    normal_cols = [(col[:-1] if col.endswith('_')
                    else col)
                   for col in normal_cols]
    _df.columns = normal_cols
    return _df


# map data sets:

data_renamer = {
    'income': 'Income', 'bank': 'Bank',
    'heart': 'Heart', 'blood': 'Blood',
    'cancer': 'Cancer', 'student': 'Student',
    'abalone': 'Abalone', 'concrete': 'Concrete',
    'building': 'Building', 'real_estate': 'Real Estate'
}

models_renamer = {
    'dummy': 'Baseline Model', 'lin': 'Linear/Logistic',
    'linear_svm': 'Linear SVM', 'rbf_svm': 'RBF SVM',
    'gaussian': 'Gaussian Model', 'mlp': 'Neural Network',
    'decisiontree': 'Decision Tree', 'rf': 'Random Forest'

}
classification_datasets = ['Income', 'Bank',
                           'Heart', 'Blood',
                           'Breast Cancer']
regression_datasets = ['Student', 'Abalone',
                       'Concrete',
                       'Building', 'Real Estate']

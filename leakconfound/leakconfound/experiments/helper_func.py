import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR

c_vals = np.geomspace(1e-2, 1e2, 25)
tree_dict = {
}

# Dict returning for each model the julearn name + params
model_dict_classification = {
    'lin': ['logit', {'logit__C': c_vals, 'logit__max_iter': [100_000]}],
    'rbf_svm': ['svm', {'svm__kernel': ['rbf'],
                        'svm__C': c_vals,

                        }],
    'linear_svm': [LinearSVC(), {'linearsvc__C': c_vals, "linearsvc__max_iter": 1_000}],
    'decisiontree': [DecisionTreeClassifier(), {
        'decisiontreeclassifier__' + key: params
        for key, params in tree_dict.items()
    }],
    'rf': ['rf', {
        **{'rf__n_estimators': [500]},
        **{'rf__' + key: params
           for key, params in tree_dict.items()}
    }],
    'mlp': [MLPClassifier(), {'mlpclassifier__hidden_layer_sizes':
                              [[32], [64], [128], [256]],
                              'mlpclassifier__max_iter':[100_000]
                              }],
    'dummy': ['dummy', {'dummy__strategy': ['prior']}]
}

model_dict_regression = {

    'lin': ['linreg', {}],

    'rbf_svm': ['svm', {'svm__kernel': ['rbf'],
                        'svm__C': c_vals}
                ],
    'linear_svm': [LinearSVR(), {'linearsvr__C': c_vals, "linearsvr__max_iter": 1_000}
                   ],
    'decisiontree': [DecisionTreeRegressor(), {
        'decisiontreeregressor__' + key: params
        for key, params in tree_dict.items()}
    ],
    'rf': ['rf', {
        **{'rf__n_estimators': [500]},
        **{'rf__' + key: params
           for key, params in tree_dict.items()}
    }],
    'mlp': [MLPRegressor(), {
        'mlpregressor__hidden_layer_sizes':    [[32], [64], [128], [256]],
        'mlpregressor__max_iter':[100_000]
    }],
    'dummy': ['dummy', {}]
}


def get_model_grid(model_name, problem_type):
    if problem_type == 'regression':
        model, param_dict = model_dict_regression[model_name]
    else:
        model, param_dict = model_dict_classification[model_name]

    return model, param_dict

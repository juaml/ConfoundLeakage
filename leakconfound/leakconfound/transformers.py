from sklearn.base import BaseEstimator, TransformerMixin


class Shuffle(BaseEstimator, TransformerMixin):

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        original_idx = X.index
        return (X
                .copy()
                .sample(len(X), replace=False, random_state=self.random_state)
                .set_index(original_idx)
                )


class WithingGroupShuffle(BaseEstimator, TransformerMixin):

    def __init__(self, grouping_var, random_state=None):
        self.random_state = random_state
        self.grouping_var = grouping_var

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        original_idx = X.index
        return (X
                .copy()
                .reset_index()
                .groupby(self.grouping_var)
                .sample(frac=1, replace=False, random_state=self.random_state)
                .set_index("index")
                .loc[original_idx, :]  # sort by original index
                .copy()
                )

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

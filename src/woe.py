import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features, target="is_high_risk", eps=1e-6):
        self.features = features
        self.target = target
        self.eps = eps
        self.woe_maps_ = {}

    def fit(self, X, y):
        df = X.copy()
        df[self.target] = y

        for col in self.features:
            grouped = df.groupby(col)[self.target].agg(["count", "sum"])
            grouped.columns = ["total", "bad"]
            grouped["good"] = grouped["total"] - grouped["bad"]

            dist_good = grouped["good"] / grouped["good"].sum()
            dist_bad = grouped["bad"] / grouped["bad"].sum()

            woe = np.log((dist_good + self.eps) / (dist_bad + self.eps))
            self.woe_maps_[col] = woe.to_dict()

        return self

    def transform(self, X):
        X_new = X.copy()
        for col, mapping in self.woe_maps_.items():
            X_new[col] = X_new[col].map(mapping).fillna(0)
        return X_new

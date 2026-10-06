"""Load and preprocess the UCI Mushroom dataset."""

import pandas as pd
from ucimlrepo import fetch_ucirepo

RANDOM_STATE = 42
DROP_COLUMNS = ["veil-type"]


def load_raw():
    ds = fetch_ucirepo(id=73)
    X = ds.data.features.copy()
    y = ds.data.targets.iloc[:, 0].copy()
    return X, y


def preprocess(X, y):
    X = X.drop(columns=[c for c in DROP_COLUMNS if c in X.columns])
    X = X.fillna("unknown")
    X = pd.get_dummies(X)
    y = (y == "p").astype(int)
    return X, y


def load_mushroom():
    return preprocess(*load_raw())
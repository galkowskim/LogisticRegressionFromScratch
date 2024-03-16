import warnings

import openml
import pandas as pd

warnings.filterwarnings("ignore")


def prepare_data(df, target_variable):
    mapping = {}

    if target_variable == "binaryClass":
        mapping = {"N": 0, "P": 1}
    elif target_variable == "class:":
        mapping = {"g": 0, "h": 1}
    elif target_variable == "defect":
        mapping = {True: 1, False: 0}
    elif target_variable == "Target":
        mapping = {"Normal": 0, "Anomaly": 1}
    elif target_variable == "Class":
        df[target_variable] = df[target_variable].astype(int)
        mapping = {1: 0, 2: 1}

    if mapping != {}:
        df[target_variable] = df[target_variable].replace(mapping)

    # I want to assert that target variable have only 2 values: 0,1 -> create code for that
    assert df[target_variable].nunique() == 2 and sorted(
        list(df[target_variable].unique())
    ) == [0, 1], df[target_variable].value_counts()

    # Filling missing values if column have more than 10% of NaNs
    for col in df.columns:
        if df[col].isnull().mean() > 0.1:
            df[col].fillna(df[col].mean(), inplace=True)

    correlated = df.corr().abs().map(lambda x: x > 0.8 and x < 1)
    if correlated.any().any():
        collinear_vars = set()
        for col in correlated.columns:
            if col not in collinear_vars:
                correlated_columns = correlated.loc[correlated[col], col].index.tolist()
                collinear_vars.update(correlated_columns)
        collinear_vars.discard(target_variable)
        df.drop(collinear_vars, axis=1, inplace=True)

    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    return X.to_numpy(), y.to_numpy()

import pandas as pd
import openml
import warnings
warnings.filterwarnings("ignore")

def prepare_data(openml_id, target_variable):
    df = openml.datasets.get_dataset(openml_id).get_data()[0]
    if target_variable=='binaryClass':
        df[target_variable] = df[target_variable].map({'N': 0, 'P': 1})
    elif target_variable=='class:':
        df[target_variable] = df[target_variable].map({'g': 0, 'h': 1})
    else:
        df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce') - 1
    if df.isnull().mean().any() >= 0.1:
        for col in df.columns:
            if df[col].isnull().any():
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

def add_interactions(X):
    _, n_features = X.shape
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction = X[:, i] * X[:, j]
            interactions.append(interaction.reshape(-1, 1))
        X = np.hstack((X, *interactions))
    return X


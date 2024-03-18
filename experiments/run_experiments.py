import numpy as np
import openml
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from src.logistic_regression import LogisticRegression
from src.prepare_datasets import prepare_data

DATASETS = {
    # small datasets
    1462: "Class",
    871: "binaryClass",
    1120: "class:",
    # big datasets
    1510: "Class",
    1050: "c",
    1049: "c",
    833: "binaryClass",
    846: "binaryClass",
    879: "binaryClass",
}

classifiers = {
    "Logistic Regression (SGD) with interactions": {
        "add_interactions": True,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-7,
        "optimizer": "sgd",
    },
    "Logistic Regression (SGD)": {
        "add_interactions": False,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-7,
        "optimizer": "sgd",
    },
    "Logistic Regression (Adam) with interactions": {
        "add_interactions": True,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-7,
        "optimizer": "adam",
    },
    "Logistic Regression (Adam)": {
        "add_interactions": False,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-7,
        "optimizer": "adam",
    },
    "Logistic Regression (IRLS) with interactions": {
        "add_interactions": True,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-7,
        "optimizer": "irls",
    },
    "Logistic Regression (IRLS)": {
        "add_interactions": False,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-7,
        "optimizer": "irls",
    },
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis,
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
}


def compare_with_different_classifiers(no_iters=10, test_size=0.2):
    # Compare the classification performance of logistic regression (try all 3 methods: IWLS, SGD, ADAM) and LDA, QDA, Decision tree and Random Forest.
    results = []

    for i, (dataset_number, target_column) in enumerate(DATASETS.items(), start=1):
        df = openml.datasets.get_dataset(dataset_number).get_data()[0]

        X, y = prepare_data(df, target_column)
        print(f"Dataset {i}:")

        for name, params_or_model in list(classifiers.items()):
            accuracy = []

            if "Logistic Regression" in name:
                model = LogisticRegression(**params_or_model)
            else:
                model = params_or_model()
            print(f"Fitting model: {name}")
            for _ in range(no_iters):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=None
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy.append(balanced_accuracy_score(y_test, y_pred))
            avg_accuracy = np.mean(accuracy)
            results.append(
                {
                    "Dataset": f"Dataset_{i}",
                    "Classifier": name,
                    "Avg_Balanced_Accuracy": avg_accuracy,
                }
            )
            if "Logistic Regression" in name:
                model.plot_log_likelihood()

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = compare_with_different_classifiers(1)
    print(results)

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.logistic_regression import LogisticRegression
from src.prepare_datasets import add_interactions, prepare_data

datasets = {
    1462: "Class",
    871: "binaryClass",
    885: "binaryClass",
    1120: "class:",
    994: "binaryClass",
    1021: "binaryClass",
    847: "binaryClass",
}

classifiers = {
    "Logistic Regression (SGD) with interactions": LogisticRegression(
        add_interactions=True, learning_rate=0.01, max_iter=1000, tolerance=1e-10
    ),
    "Logistic Regression (SGD)": LogisticRegression(
        add_interactions=False, learning_rate=0.01, max_iter=100, tolerance=1e-10
    ),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
}


def compare_with_different_classifiers():
    # Compare the classification performance of logistic regression (try all 3 methods: IWLS, SGD, ADAM) and LDA, QDA, Decision tree and Random Forest.
    results = []

    for i, (dataset_number, target_column) in enumerate(datasets.items(), start=1):
        X, y = prepare_data(dataset_number, target_column)
        print(f"Dataset {i}:")

        for name, model in list(classifiers.items())[1:]:
            accuracy = []
            for split in np.arange(0.15, 0.41, 0.05):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=split, random_state=None
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy.append(balanced_accuracy_score(y_test, y_pred))
            avg_accuracy = np.mean(accuracy)
            # print(f'Avg of balanced_accuracy for {name} = {avg_accuracy:.3f}')
            results.append(
                {
                    "Dataset": f"Dataset_{i}",
                    "Classifier": name,
                    "Avg_Balanced_Accuracy": avg_accuracy,
                }
            )

    return pd.DataFrame(results)


def compare_w_wo_interactions():
    # for small datasets, compare logistic regression with and without interactions
    results = []

    for i, (dataset_number, target_column) in enumerate(
        list(datasets.items())[:3], start=1
    ):
        X, y = prepare_data(dataset_number, target_column)
        print(f"Dataset {i}:")

        for name, model in list(classifiers.items())[:2]:
            accuracy = []
            if model.add_interactions == True:
                X = add_interactions(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=None
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = balanced_accuracy_score(y_test, y_pred)
            # print(f'Balanced_accuracy for {name} = {accuracy:.3f}')
            results.append(
                {
                    "Dataset": f"Dataset_{i}",
                    "Classifier": name,
                    "Balanced_Accuracy": accuracy,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    results1 = compare_with_different_classifiers()
    print(results1)
    results2 = compare_w_wo_interactions()
    print(results2)

import os
import gzip
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --- Constants ---
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"
TRAIN_DATA_PATH = "files/input/train_data.csv.zip"
TEST_DATA_PATH = "files/input/test_data.csv.zip"



# --- Data Utilities ---
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in [0, 1, 2, 3, 4] else 4)
    return df


def split_features_labels(df):
    X = df.drop(columns=["default"])
    y = df["default"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


# --- Pipeline Utilities ---
def build_pipeline():
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )


def tune_hyperparameters(pipeline, X_train, y_train):
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
    }
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


# --- Persistence Utilities ---
def save_compressed_pickle(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as f:
        pickle.dump(obj, f)


# --- Evaluation Utilities ---
def compute_metrics(model, X, y, dataset):
    y_pred = model.predict(X)
    return {
        "dataset": dataset,
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
    }


def compute_confusion_matrix(model, X, y, dataset):
    cm = confusion_matrix(y, model.predict(X))
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


def save_metrics(metrics_list, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + "\n")


# --- Main ---
def main():
    # Data Preparation
    train_df = load_and_clean_data(TRAIN_DATA_PATH)
    test_df = load_and_clean_data(TEST_DATA_PATH)
    X_train, X_test, y_train, y_test = split_features_labels(train_df)

    # Model Training & Optimization
    pipeline = build_pipeline()
    best_model = tune_hyperparameters(pipeline, X_train, y_train)

    # Model Persistence
    save_compressed_pickle(best_model, MODEL_PATH)

    # Evaluation
    metrics = [
        compute_metrics(best_model.best_estimator_, X_train, y_train, "train"),
        compute_metrics(best_model.best_estimator_, X_test, y_test, "test"),
        compute_confusion_matrix(best_model.best_estimator_, X_train, y_train, "train"),
        compute_confusion_matrix(best_model.best_estimator_, X_test, y_test, "test"),
    ]
    save_metrics(metrics, METRICS_PATH)


if __name__ == "__main__":
    main()
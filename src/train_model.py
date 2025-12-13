import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import os

# Make sure folders exist
os.makedirs(".models", exist_ok=True)
os.makedirs("models/figures", exist_ok=True)
#mlflow.set_tracking_uri('file:../mlruns')
mlflow.set_experiment("Credit_Risk_Model_Experiment")


def eval_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc


def plot_conf_matrix(cm, labels, filename):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    idx = importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[idx])
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in idx], rotation=90)
    plt.tight_layout()

    path = "models/figures/feature_importances.png"
    plt.savefig(path)
    plt.close()
    return path


def train_model():

    data_path = "Data/Processed/customer_risk_data_final.csv"

    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)

    X = df.drop(columns=["is_high_risk", "CustomerId"])
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------------
    # Logistic Regression
    # -------------------------------------
    with mlflow.start_run(run_name="Logistic_Regression"):

        lr = LogisticRegression(max_iter=6000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        acc, prec, rec, f1, roc_auc = eval_metrics(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(sk_model=lr, name="model")

        print(f"LR F1={f1:.4f}, ROC_AUC={roc_auc:.4f}")

    # -------------------------------------
    # Random Forest with Tuning
    # -------------------------------------
    with mlflow.start_run(run_name="Random_Forest"):

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [8, 10, 12],
            "min_samples_split": [2, 5]
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        rf = grid.best_estimator_
        y_pred = rf.predict(X_test)

        acc, prec, rec, f1, roc_auc = eval_metrics(y_test, y_pred)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Confusion matrix (save to /reports/figures)
        cm = confusion_matrix(y_test, y_pred)
        cm_path = plot_conf_matrix(
            cm,
            ["Low Risk", "High Risk"],
            "models/figures/confusion_matrix.png"
        )
        mlflow.log_artifact(cm_path)

        # Feature importances
        fi_path = plot_feature_importance(rf, X.columns)
        mlflow.log_artifact(fi_path)

        # Save model
        joblib.dump(rf, "models/model.pkl")
        mlflow.sklearn.log_model(sk_model=rf, name="model")

        print(f"RF F1={f1:.4f}, ROC_AUC={roc_auc:.4f}")


if __name__ == "__main__":
    train_model()

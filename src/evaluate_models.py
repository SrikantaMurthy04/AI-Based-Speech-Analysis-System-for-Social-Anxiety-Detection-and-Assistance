import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.config import FEATURE_FILE, GRAPH_PATH, MODEL_PATH
from utils.helpers import load_model


def load_data():

    df = pd.read_csv(FEATURE_FILE)

    X = df.drop("label", axis=1)
    y = df["label"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def evaluate_all_models():

    X_train, X_test, y_train, y_test = load_data()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_names = [
        "logistic_regression",
        "svm",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "neural_network"
    ]

    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*50}")
        
        model = load_model(f"{MODEL_PATH}/{model_name}.pkl")
        predictions = model.predict(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title(f"{model_name.replace('_', ' ').title()} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.savefig(f"{GRAPH_PATH}/confusion_matrix_{model_name}.png")
        plt.close()

        print(f"Saved confusion matrix for {model_name}")

        # Classification Report
        report = classification_report(y_test, predictions)

        print(f"\nClassification Report for {model_name}:\n")
        print(report)


def model_accuracy_comparison():

    X_train, X_test, y_train, y_test = load_data()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model_names = [
        "logistic_regression",
        "svm",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "neural_network"
    ]

    from sklearn.metrics import accuracy_score
    results = {}

    for model_name in model_names:
        model = load_model(f"{MODEL_PATH}/{model_name}.pkl")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[model_name.replace("_", " ").title()] = accuracy

    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(12, 6))

    sns.barplot(x=names, y=scores, palette="viridis")

    plt.title("Model Accuracy Comparison (All Models)")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.0])

    # Add value labels on bars
    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

    plt.tight_layout()
    plt.savefig(f"{GRAPH_PATH}/model_comparison.png", dpi=300)
    plt.close()

    print("\nModel Accuracy Comparison:")
    for model, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {score:.4f}")
    
    print("\nSaved model comparison graph")


if __name__ == "__main__":

    evaluate_all_models()

    model_accuracy_comparison()
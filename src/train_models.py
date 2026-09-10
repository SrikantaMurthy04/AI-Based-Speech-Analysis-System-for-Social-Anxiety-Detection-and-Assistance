import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

from utils.config import FEATURE_FILE, MODEL_PATH
from utils.helpers import save_model


def load_features():

    df = pd.read_csv(FEATURE_FILE)

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y


def train_models():

    X, y = load_features()

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 🔹 Save the scaler so prediction uses the same scaling
    save_model(scaler, f"{MODEL_PATH}/scaler.pkl")
    print("Scaler saved.")

    models = {

        "logistic_regression": LogisticRegression(max_iter=1000),

        "svm": SVC(probability=True),

        "random_forest": RandomForestClassifier(n_estimators=200),

        "gradient_boosting": GradientBoostingClassifier(),

        "xgboost": XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss'),

        "lightgbm": LGBMClassifier(n_estimators=200),

        "neural_network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)

    }

    results = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        results[name] = accuracy

        print(f"{name} Accuracy: {accuracy:.4f}")

        # Save model
        model_path = f"{MODEL_PATH}/{name}.pkl"
        save_model(model, model_path)

        print(f"Model saved to {model_path}")

    print("\nModel Comparison:")
    for model, score in results.items():
        print(f"{model}: {score:.4f}")


if __name__ == "__main__":

    train_models()
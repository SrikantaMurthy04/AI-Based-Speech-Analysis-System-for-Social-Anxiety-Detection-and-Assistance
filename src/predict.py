import os
import sys
from collections import Counter
import numpy as np
import librosa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import MODEL_PATH
from utils.helpers import load_model
from src.suggestions import get_suggestion


def extract_features(file_path):

    audio, sr = librosa.load(file_path, duration=3)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    rms = np.mean(librosa.feature.rms(y=audio))

    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    features = np.hstack([mfcc, zcr, rms, centroid, bandwidth])

    return features


def predict_anxiety(audio_path):

    # load trained components
    models = {
        "logistic_regression": load_model(f"{MODEL_PATH}/logistic_regression.pkl"),
        "svm": load_model(f"{MODEL_PATH}/svm.pkl"),
        "random_forest": load_model(f"{MODEL_PATH}/random_forest.pkl"),
        "gradient_boosting": load_model(f"{MODEL_PATH}/gradient_boosting.pkl"),
        "xgboost": load_model(f"{MODEL_PATH}/xgboost.pkl"),
        "lightgbm": load_model(f"{MODEL_PATH}/lightgbm.pkl"),
        "neural_network": load_model(f"{MODEL_PATH}/neural_network.pkl"),
    }
    
    scaler = load_model(f"{MODEL_PATH}/scaler.pkl")
    encoder = load_model(f"{MODEL_PATH}/label_encoder.pkl")

    # extract features
    features = extract_features(audio_path)

    features = features.reshape(1, -1)

    # apply same scaling used during training
    features = scaler.transform(features)

    # Ensemble voting: get predictions and confidences from all models
    predictions = []
    model_results = {}

    for model_name, model in models.items():
        pred_idx = int(model.predict(features)[0])
        pred_label = encoder.inverse_transform([pred_idx])[0]
        confidence = None

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(features)[0]
                confidence = float(proba[pred_idx])
            except Exception:
                confidence = None

        model_results[model_name] = {
            "prediction": pred_label,
            "confidence": confidence
        }

        predictions.append(pred_idx)

    # Use majority voting for final prediction
    final_pred_idx = Counter(predictions).most_common(1)[0][0]
    anxiety_level = encoder.inverse_transform([final_pred_idx])[0]

    suggestion = get_suggestion(anxiety_level)

    return anxiety_level, suggestion, model_results


if __name__ == "__main__":

    audio_file = "dataset/ravdess/Actor_01/03-01-06-01-01-01-01.wav"

    anxiety, suggestion = predict_anxiety(audio_file)

    print("\nPrediction Result")
    print("------------------")
    print("Anxiety Level:", anxiety)
    print("Suggestion:", suggestion)
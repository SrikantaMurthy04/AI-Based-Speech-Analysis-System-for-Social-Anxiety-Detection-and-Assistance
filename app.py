import streamlit as st
import os
import tempfile
import pandas as pd

from src.predict import predict_anxiety
from utils.config import FEATURE_FILE, GRAPH_PATH, MODEL_PATH
from utils.helpers import load_model


def show_audio_model_predictions(model_results):
    table_data = []

    for model_name, result in model_results.items():
        confidence = result["confidence"]
        table_data.append(
            {
                "Model": model_name.replace("_", " ").title(),
                "Prediction": result["prediction"].upper(),
                "Confidence (%)": f"{confidence * 100:.2f}" if confidence is not None else "N/A"
            }
        )

    st.subheader("Model predictions for this audio")
    st.table(pd.DataFrame(table_data))



st.set_page_config(page_title="Speech Anxiety Detection", layout="wide")

st.title("🎤 AI-Based Speech Analysis System for Social Anxiety Detection and Assistance")

st.write(
    "Speak or upload an audio file and the system will analyze your anxiety level."
)


# Sidebar navigation
st.sidebar.title("Navigation")

option = st.sidebar.radio(
    "Select Option",
    ["Home", "Predict Anxiety", "Model Performance"]
)


# ---------------- HOME ----------------

if option == "Home":

    st.header("Project Overview")

    st.write("""
    This system analyzes speech and predicts anxiety levels using machine learning.

    **Ensemble Voting Models Trained:**
    - Logistic Regression
    - SVM
    - Random Forest
    - Gradient Boosting
    - XGBoost 
    - LightGBM 
    - Neural Network 

    **Prediction Method:** Ensemble voting combines all 7 models for improved accuracy and robustness.
    """)


# ---------------- PREDICTION ----------------

elif option == "Predict Anxiety":

    st.header("Analyze Speech")

    # ---- RECORD AUDIO ----

    st.subheader("🎤 Record Your Speech")

    audio_bytes = st.audio_input("Speak and record your voice")

    if audio_bytes is not None:

        st.audio(audio_bytes)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes.read())
            temp_path = f.name

        if st.button("Analyze Recorded Speech"):

            anxiety, suggestion, model_results = predict_anxiety(temp_path)
            show_audio_model_predictions(model_results)

            st.success(f"Predicted Anxiety Level: {anxiety.upper()}")
            st.info(f"Suggestion: {suggestion}")


    st.divider()

    # ---- UPLOAD AUDIO ----

    st.subheader("📂 Upload Audio File")

    uploaded_file = st.file_uploader(
        "Upload a WAV audio file",
        type=["wav"]
    )

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        st.audio(uploaded_file)

        if st.button("Analyze Uploaded Speech"):

            anxiety, suggestion, model_results = predict_anxiety(temp_path)
            show_audio_model_predictions(model_results)

            st.success(f"Predicted Anxiety Level: {anxiety.upper()}")
            st.info(f"Suggestion: {suggestion}")


# ---------------- MODEL PERFORMANCE ----------------

elif option == "Model Performance":

    st.header("Model Evaluation")

    model_names = [
        "logistic_regression",
        "svm",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "neural_network"
    ]

    st.subheader("📊 Model Accuracy Comparison")

    graph_path = os.path.join(GRAPH_PATH, "model_comparison.png")

    if os.path.exists(graph_path):
        st.image(graph_path)
    else:
        st.warning("Model comparison graph not found. Run evaluate_models.py to generate it.")

    st.subheader("📈 Confusion Matrices by Model")

    for model_name in model_names:
        cm_path = os.path.join(GRAPH_PATH, f"confusion_matrix_{model_name}.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f"Confusion Matrix - {model_name.replace('_', ' ').title()}")


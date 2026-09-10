# AI-Based Speech Analysis System for Social Anxiety Detection and Assistance

## Project Overview

This project explores how speech patterns can be used to identify
anxiety-related patterns in spoken audio.

I built a Python-based system that takes speech as input, preprocesses
the audio, extracts relevant features, and passes them through multiple
machine learning models.

The predictions from the individual models are combined using an
ensemble voting approach to determine the final anxiety level. The
application also displays the individual model predictions and
confidence scores, along with a supportive suggestion.

## Features

- Record speech directly through the application
- Upload WAV audio for analysis
- Preprocess and extract features from speech
- Predict anxiety levels using multiple ML models
- Compare predictions and confidence scores across models
- Ensemble voting for final prediction
- View model performance and evaluation results
- Provide supportive suggestions based on the prediction

## Technologies Used

- Python
- Streamlit
- Librosa
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib

## Project Workflow

```text
Speech Input
↓
Audio Preprocessing
↓
Feature Extraction
↓
Feature Scaling
↓
Multiple Machine Learning Models
↓
Ensemble Voting
↓
Final Anxiety Level
↓
Supportive Suggestions
```

## Machine Learning Models

The project evaluates multiple models:

- Logistic Regression
- SVM
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Neural Network

The predictions from these models are combined using ensemble voting
to obtain the final prediction.

## Dataset

The dataset is not included in this repository because of its size.
It should be downloaded separately and placed in the `dataset/` folder.

## Project Structure

```text
models/
outputs/
src/
ui/
utils/
app.py
count_features.py
requirements.txt
.gitignore
README.md
```

## How to Run

```bash
git clone https://github.com/SrikantaMurthy04/AI-Based-Speech-Analysis-System-for-Social-Anxiety-Detection-and-Assistance.git
cd AI-Based-Speech-Analysis-System-for-Social-Anxiety-Detection-and-Assistance
pip install -r requirements.txt
streamlit run app.py
```

## Results

The project includes confusion matrices and model comparison graphs
in the outputs/graphs/ directory.

## Author

Srikanta Murthy L



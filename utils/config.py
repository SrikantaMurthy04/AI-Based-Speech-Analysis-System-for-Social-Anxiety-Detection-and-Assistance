import os

# Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ravdess")

# Output folders
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs")
GRAPH_PATH = os.path.join(OUTPUT_PATH, "graphs")
PROCESSED_DATA_PATH = os.path.join(OUTPUT_PATH, "processed_data")
AUDIO_SAMPLE_PATH = os.path.join(OUTPUT_PATH, "audio_samples")

# Feature file
FEATURE_FILE = os.path.join(PROCESSED_DATA_PATH, "features.csv")

# Models
MODEL_PATH = os.path.join(BASE_DIR, "models")

# Ensure folders exist
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(AUDIO_SAMPLE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
import os
import pandas as pd

from utils.config import DATASET_PATH
from utils.helpers import list_audio_files


# Emotion mapping based on RAVDESS filename
EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}


def extract_emotion(file_name):
    """
    Extract emotion label from RAVDESS filename
    Example filename:
    03-01-06-01-02-01-12.wav
    Emotion code = 06
    """
    parts = file_name.split("-")
    emotion_code = int(parts[2])
    return EMOTION_MAP.get(emotion_code, "unknown")


def load_dataset():
    """
    Load dataset and return dataframe with file paths and emotions
    """

    audio_files = list_audio_files(DATASET_PATH)

    data = []

    for file_path in audio_files:

        file_name = os.path.basename(file_path)

        emotion = extract_emotion(file_name)

        data.append({
            "file_path": file_path,
            "emotion": emotion
        })

    df = pd.DataFrame(data)

    return df


if __name__ == "__main__":

    df = load_dataset()

    print("Dataset loaded successfully\n")

    print("Total samples:", len(df))

    print("\nEmotion distribution:\n")
    print(df["emotion"].value_counts())

    print("\nSample data:\n")
    print(df.head())
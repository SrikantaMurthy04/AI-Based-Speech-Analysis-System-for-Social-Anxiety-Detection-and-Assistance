from sklearn.preprocessing import LabelEncoder

from src.data_loader import load_dataset
from utils.helpers import save_model
from utils.config import MODEL_PATH


# Emotion → Anxiety mapping
ANXIETY_MAP = {
    "neutral": "low",
    "calm": "low",
    "happy": "low",
    "sad": "medium",
    "angry": "medium",
    "disgust": "medium",
    "fearful": "high",
    "surprised": "high"
}


def preprocess_dataset():
    """
    Load dataset and convert emotion to anxiety level
    """

    df = load_dataset()

    df["anxiety"] = df["emotion"].map(ANXIETY_MAP)

    return df


def encode_labels(df):
    """
    Convert anxiety labels into numeric values
    """

    encoder = LabelEncoder()

    df["anxiety_label"] = encoder.fit_transform(df["anxiety"])

    # 🔹 Save encoder so prediction uses the same mapping
    save_model(encoder, f"{MODEL_PATH}/label_encoder.pkl")

    print("Label encoder saved.")

    return df, encoder


if __name__ == "__main__":

    df = preprocess_dataset()

    df, encoder = encode_labels(df)

    print("\nDataset Preview:\n")
    print(df.head())

    print("\nAnxiety Distribution:\n")
    print(df["anxiety"].value_counts())

    print("\nLabel Mapping:\n")
    for label, value in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        print(f"{label} -> {value}")